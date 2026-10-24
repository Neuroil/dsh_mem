/**
 * Benchmark 共享基建：配置、介质适配器（memory / sqlite）、计时统计、
 * 复杂度拟合、记录工厂、确定性 LLM 替身。
 *
 * 原则：
 *  - 一切被测代码都是**生产代码**（../lib 下的构建产物），benchmark 只注入
 *    介质与模型替身（deterministic 嵌入 / LLM stub），不复制实现；
 *  - 计时一律 performance.now()，逐 op 采样 + 批量吞吐双口径；
 *  - 同种子语料跨机可复现（corpus.mjs）。
 */
import { existsSync } from 'node:fs'
import { rm, stat, mkdir } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Config as makeSchemaConfig } from '../../lib/config.js'
import { memoryUnitDescriptor } from '../../lib/memory-store.js'
import { epiLexical } from '../../lib/judges.js'

const HERE = dirname(fileURLToPath(import.meta.url))
export const BENCH_ROOT = join(HERE, '..')
export const TMP_DIR = join(BENCH_ROOT, 'tmp')

// ---------------------------------------------------------------- 配置

/**
 * 构造 benchmark 配置（schemastery 填默认值，与 src/config.ts 默认一致）。
 * 默认 deterministic 嵌入/重排（1024 维，与生产 bge-m3 同维数）。
 */
export function benchConfig(overrides = {}) {
  return makeSchemaConfig({
    storage: { backend: 'sqlite', unit: 'dsh_mem_bench', version: 1, workspaceScope: false },
    embedding: { mode: 'deterministic' },
    reranker: { mode: 'deterministic' },
    ...overrides,
  })
}

// ---------------------------------------------------------------- 介质适配器

/**
 * 内存介质 KvUnit（eval-ndcg 同款）+ 写计数：
 * puts/deletes/putChars 记录写入侧的介质压力（供与 sqlite 实测对齐）。
 */
export function memoryUnit() {
  const tables = { episodic: {}, semantic: {} }
  const counters = { puts: 0, deletes: 0, putChars: 0 }
  return {
    counters,
    async loadAll() {
      return { tables: structuredClone(tables), global: null }
    },
    async putRecord(table, key, value) {
      counters.puts++
      const text = JSON.stringify(value)
      counters.putChars += text.length
      tables[table][key] = JSON.parse(text)
    },
    async deleteRecord(table, key) {
      counters.deletes++
      delete tables[table][key]
    },
    async setGlobal() { throw new Error('no global slot') },
    async close() {},
  }
}

/**
 * sqlite 介质（生产推荐后端 @deepseek-ai/dsh-storage-sqlite，WAL）：
 * 一个实例独占一个库文件；sizes() 读三件套体积，inspect() 在 close 后
 * 只读打开取页统计与每表字节数，vacuum() 单独做压实实验。
 */
export class SqliteMedium {
  constructor(name) {
    this.path = join(TMP_DIR, `${name}.sqlite`)
    this.backend = null
    this.name = name
  }

  /**
   * 打开一个 KV unit。fresh=true（默认）在**首次**创建后端时清掉旧库文件
   * （每个 benchmark 场景独占新文件）；close 后重开同一文件必须传
   * fresh=false——后端重建不再隐式删库。
   */
  async openUnit(unitName, config, { fresh = true } = {}) {
    if (this.backend === null) {
      await mkdir(TMP_DIR, { recursive: true })
      if (fresh && existsSync(this.path)) await rm(this.path)
      const { SqliteStorageBackend } = await import('@deepseek-ai/dsh-storage-sqlite')
      this.backend = new SqliteStorageBackend({ path: this.path, journalMode: 'wal' })
    }
    return this.backend.kv.open(memoryUnitDescriptor(unitName, config))
  }

  async close() {
    if (this.backend !== null) {
      await this.backend.close()
      this.backend = null
    }
  }

  async sizes() {
    const out = { db: 0, wal: 0, shm: 0 }
    for (const [key, suffix] of [['db', ''], ['wal', '-wal'], ['shm', '-shm']]) {
      try { out[key] = (await stat(this.path + suffix)).size } catch { /* 尚未创建 */ }
    }
    return out
  }

  /** close 后调用：只读打开取 page 统计 + 每表行数/字节数（payload 口径）。 */
  async inspect(unitName) {
    const { DatabaseSync } = await import('node:sqlite')
    const db = new DatabaseSync(this.path, { readOnly: true })
    try {
      const page = db.prepare('PRAGMA page_size').get()
      const count = db.prepare('PRAGMA page_count').get()
      const free = db.prepare('PRAGMA freelist_count').get()
      const tables = {}
      for (const table of ['episodic', 'semantic']) {
        const physical = `u_${unitName}_${table}`
        try {
          const row = db.prepare(
            `SELECT COUNT(*) AS rows, COALESCE(SUM(LENGTH(key) + LENGTH(value)), 0) AS bytes FROM "${physical}"`,
          ).get()
          tables[table] = { rows: row.rows, payloadBytes: row.bytes }
        } catch { tables[table] = { rows: 0, payloadBytes: 0 } }
      }
      return {
        pageSize: page.page_size,
        pageCount: count.page_count,
        freelistPages: free.freelist_count,
        fileBytes: page.page_size * count.page_count,
        tables,
      }
    } finally {
      db.close()
    }
  }

  /** VACUUM 压实（close 后调用），返回压实后文件体积。 */
  async vacuum() {
    const { DatabaseSync } = await import('node:sqlite')
    const db = new DatabaseSync(this.path)
    try {
      db.exec('VACUUM')
    } finally {
      db.close()
    }
    return (await this.sizes()).db
  }

  async remove() {
    for (const suffix of ['', '-wal', '-shm']) {
      try { await rm(this.path + suffix) } catch { /* 不存在即可 */ }
    }
  }
}

// ---------------------------------------------------------------- 统计

/** 样本汇总（ms 口径）：mean/p50/p95/min/max/total + 吞吐。 */
export function summarize(samples) {
  const n = samples.length
  if (n === 0) return { n: 0, mean: 0, p50: 0, p95: 0, min: 0, max: 0, total: 0, opsPerSec: 0 }
  const sorted = Float64Array.from(samples).sort()
  const at = (q) => sorted[Math.min(n - 1, Math.floor(q * (n - 1)))]
  const total = samples.reduce((s, x) => s + x, 0)
  const mean = total / n
  return {
    n,
    mean,
    p50: at(0.5),
    p95: at(0.95),
    min: sorted[0],
    max: sorted[n - 1],
    total,
    opsPerSec: total > 0 ? (n / total) * 1000 : 0,
  }
}

/**
 * 幂律拟合 t = a·x^b（log-log 最小二乘）。
 * points: [{x, y}]，y > 0；返回 {a, b, r2}。b 即经验复杂度指数
 * （b≈1 → O(N)，b≈2 → O(N²)，b≈0 → O(1)）。
 */
export function fitPowerLaw(points) {
  const pts = points.filter((p) => p.x > 0 && p.y > 0)
  const n = pts.length
  if (n < 2) return { a: NaN, b: NaN, r2: NaN }
  const lx = pts.map((p) => Math.log(p.x))
  const ly = pts.map((p) => Math.log(p.y))
  const mx = lx.reduce((s, v) => s + v, 0) / n
  const my = ly.reduce((s, v) => s + v, 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) {
    sxy += (lx[i] - mx) * (ly[i] - my)
    sxx += (lx[i] - mx) ** 2
    syy += (ly[i] - my) ** 2
  }
  const b = sxx === 0 ? NaN : sxy / sxx
  const a = Math.exp(my - b * mx)
  const r2 = sxx === 0 || syy === 0 ? 1 : (sxy * sxy) / (sxx * syy)
  return { a, b, r2 }
}

/** 简单 markdown 表格。headers: string[]，rows: (string|number)[][] */
export function mdTable(headers, rows) {
  const line = (cells) => '| ' + cells.join(' | ') + ' |'
  return [line(headers), line(headers.map(() => '---')), ...rows.map(line)].join('\n')
}

export const f1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : '—')
export const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '—')
export const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : '—')
export const int = (x) => Math.round(x).toLocaleString('en-US')

export function log(msg) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`)
}

// ---------------------------------------------------------------- 记录工厂

let recordSeq = 0

/** 与 write-path 落库形状一致的全字段 MemoryRecord（epi 走真实 epiLexical）。 */
export function makeRecord({ id, text, embedding, imp = 6, now = Date.now() }) {
  return {
    id: id ?? `bench_${(recordSeq++).toString(36)}_${crypto.randomUUID()}`,
    text,
    embedding,
    imp,
    valence: 0,
    arousal: 0.1,
    epi: epiLexical(text),
    red: 0,
    r_m: 0,
    tier: imp >= 8 ? 'hot' : imp < 2 ? 'cold' : 'warm',
    con_route: { kind: 'pending' },
    links: [],
    S: imp,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: now, valid_at: now },
    flags: {},
    origin: { session_id: 'bench', turn: 0, seq: recordSeq },
  }
}

// ---------------------------------------------------------------- LLM 替身

/**
 * 确定性 InternalLlm 替身：按 system 提示词特征分流到各判定，返回合法 JSON。
 * calls 计数每类调用次数（LLM 调用次数是写入/睡眠路径的真实成本维度）。
 * behavior 可覆盖默认应答；candidates(user) 供给候选抽取（返回字符串数组）。
 */
export function stubLlm(behavior = {}) {
  const calls = { extract: 0, importance: 0, contradiction: 0, gist: 0, qaGen: 0, qaAnswer: 0, compat: 0 }
  const importance = behavior.importance ?? { importance: 6, valence: 0, arousal: 0.2, reasons: ['task_impact'] }
  return {
    calls,
    async complete(system, user) {
      if (system.includes('distill durable memories')) {
        calls.extract++
        const candidates = behavior.candidates?.(user) ?? []
        return JSON.stringify(candidates)
      }
      if (system.includes('score how strongly')) {
        calls.importance++
        return JSON.stringify(importance)
      }
      if (system.includes('judge how a NEW memory')) {
        calls.contradiction++
        return JSON.stringify(behavior.contradiction ?? { index: 0, verdict: 'unrelated' })
      }
      if (system.includes('distill the shared semantic core')) {
        calls.gist++
        return JSON.stringify({ gist: behavior.gist ?? 'The user has stable preferences around this topic.' })
      }
      if (system.includes('question-answer pairs')) {
        calls.qaGen++
        return JSON.stringify(behavior.qaPairs ?? [
          { question: 'What is the summary about?', answer: 'stable preferences' },
          { question: 'Is it durable?', answer: 'stable preferences' },
          { question: 'Whose preference?', answer: 'stable preferences' },
        ])
      }
      if (system.includes('using ONLY the provided summary')) {
        calls.qaAnswer++
        return JSON.stringify({ answer: behavior.qaAnswer ?? 'stable preferences' })
      }
      if (system.includes('folded into an EXISTING')) {
        calls.compat++
        return JSON.stringify({ compatible: behavior.compatible ?? true })
      }
      throw new Error(`stubLlm: 未识别的 system 提示词：${system.slice(0, 60)}…`)
    },
  }
}

/** 进度条式简报：每完成一个规模打印一行。 */
export function reportScale(name, n, extra) {
  log(`${name} N=${int(n)} ${extra}`)
}

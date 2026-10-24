/**
 * Benchmark C · 写入管线每轮成本（WritePipeline.processTurn）：
 *
 * 真实 WritePipeline（生产代码）+ deterministic 嵌入 + LLM 替身（零网络延迟，
 * 应答合法 JSON），隔离出**插件侧每轮开销**：候选去重、批量嵌入、M5 判分、
 * 矛盾近邻扫描（O(N) similar）、落库、邻域统计回写（O(N)）。
 * 生产中每轮还有 1+c 次 LLM 调用与 1 次批量嵌入的网络成本（README 给出量级）。
 *
 * 变体：候选数 c ∈ {4, 8}；矛盾判定 contradicts（触发软失效写）vs unrelated；
 * sqlite 介质（持久化份额）。
 *
 * 用法：node bench-write.mjs [--scales 0,1000,5000,10000] [--turns 20]
 */
import os from 'node:os'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import { WritePipeline } from '../lib/write-path.js'
import {
  benchConfig, memoryUnit, SqliteMedium, stubLlm, makeRecord, summarize, fitPowerLaw,
  mdTable, f2, f3, f1, int, log,
} from './support/harness.mjs'
import { generateCorpus, FAMILY_KEYS } from './support/corpus.mjs'

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))
const SCALES = (args.scales ?? '0,1000,5000,10000').split(',').map(Number)
const TURNS = Number(args.turns ?? 20)

const config = benchConfig()
const embedding = createEmbeddingProvider(config.embedding)

/** 建库到 n 条（putRecord 直写，形状与写入路径一致）。 */
async function seedStore(n, medium = 'memory', mediumObj = null) {
  const unit = medium === 'memory' ? memoryUnit() : await mediumObj.openUnit('dsh_mem_bench', config)
  const store = await MemoryStore.open(unit, createVectorIndex(), config)
  if (n > 0) {
    const { texts } = generateCorpus(n, 777)
    const vectors = await embedding.embed(texts)
    const now = Date.now()
    for (let i = 0; i < n; i++) {
      await store.putRecord(makeRecord({ text: texts[i], embedding: vectors[i], now }))
    }
  }
  return store
}

/** 一轮写入的候选文本：全新（不与库重复）、带族词汇（近邻扫描非退化）。 */
function freshCandidates(turn, c) {
  const out = []
  for (let k = 0; k < c; k++) {
    const family = FAMILY_KEYS[(turn + k) % FAMILY_KEYS.length]
    out.push(`Benchmark turn ${turn} candidate ${k}: a fresh durable note about ${family.replace('-', ' ')} habits and decisions, stated for the ${turn}x${k} experiment.`)
  }
  return out
}

async function runVariant(label, { n, candidates, verdict, medium, configOverrides = {} }) {
  let store
  let mediumObj = null
  if (medium === 'sqlite') {
    mediumObj = new SqliteMedium(`write_${n}_${label}`)
    store = await seedStore(n, 'sqlite', mediumObj)
  } else {
    store = await seedStore(n)
  }

  // 默认 contradictionMinCos=0.8 是给真实嵌入（bge-m3 相关文本 0.6-0.85）的；
  // deterministic 嵌入族间 p99≈0.45，走矛盾分支需降到 0.1（e2e-write 同款处理）。
  const cfg = Object.keys(configOverrides).length > 0 ? benchConfig(configOverrides) : config
  let turn = 0
  const llm = stubLlm({
    candidates: () => freshCandidates(turn, candidates),
    contradiction: verdict === 'contradicts'
      ? { index: 0, verdict: 'contradicts', temporal: 'overlapping' }
      : undefined,
  })
  const pipeline = new WritePipeline({ llm, embedding, store, config: cfg })

  const samples = new Float64Array(TURNS)
  const messages = [{ role: 'user', text: `bench turn ${'${turn}'}` }, { role: 'assistant', text: 'ok' }]
  for (let t = 0; t < TURNS; t++) {
    turn = t
    const t0 = performance.now()
    const outcome = await pipeline.processTurn('bench-session', t, messages)
    samples[t] = performance.now() - t0
    if (outcome.written.length !== candidates && outcome.dropped > 0) {
      log(`  注意：turn ${t} written=${outcome.written.length} dropped=${outcome.dropped}`)
    }
  }
  const stats = summarize(samples)
  const row = {
    label, n, candidates, verdict, medium, turns: TURNS,
    perTurn: stats,
    perCandidate: { mean: stats.mean / candidates },
    llmCalls: { ...llm.calls },
    records: store.allMemories().length,
  }
  await store.close()
  if (mediumObj !== null) {
    await mediumObj.close()
    await mediumObj.remove()
  }
  results.runs.push(row)
  log(`${label} N=${int(n)} c=${candidates} ${verdict}/${medium}: ${f1(stats.mean)}ms/轮（p95 ${f1(stats.p95)}）LLM 调用/轮 = ${1 + llm.calls.importance / TURNS + llm.calls.contradiction / TURNS}`)
  return row
}

const results = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    dims: config.embedding.dimensions,
    contradictionTopK: config.write.contradictionTopK,
    minImportance: config.write.minImportance,
    turns: TURNS,
    scales: SCALES,
  },
  runs: [],
}
log(`bench-write 启动：scales=${SCALES.join(',')} turns=${TURNS}`)

// 主扫描：c=4、unrelated、memory 介质，库规模 0 → 10k
for (const n of SCALES) {
  await runVariant('baseline', { n, candidates: 4, verdict: 'unrelated', medium: 'memory' })
}
// 变体（均以最大规模为参照）
const maxN = SCALES.at(-1)
await runVariant('c=8', { n: maxN, candidates: 8, verdict: 'unrelated', medium: 'memory' })
await runVariant('contradicts', {
  n: Math.min(maxN, 5000), candidates: 4, verdict: 'contradicts', medium: 'memory',
  configOverrides: { write: { contradictionMinCos: 0.1 } },
})
await runVariant('sqlite', { n: Math.min(maxN, 1000), candidates: 4, verdict: 'unrelated', medium: 'sqlite' })

// ---------------------------------------------------------------- 汇总

console.log('\n== 每轮写入成本（deterministic LLM 替身，插件侧开销；生产另加网络） ==')
console.log(mdTable(
  ['变体', '库规模 N', '候选 c', '每轮 mean(ms)', 'p95(ms)', '每候选(ms)'],
  results.runs.map((r) => [
    r.label, int(r.n), r.candidates, f1(r.perTurn.mean), f1(r.perTurn.p95), f2(r.perCandidate.mean),
  ]),
))

const baseline = results.runs.filter((r) => r.label === 'baseline' && r.n >= 1)
if (baseline.length >= 2) {
  const fit = fitPowerLaw(baseline.map((r) => ({ x: r.n, y: r.perTurn.mean })))
  console.log(`\n每轮成本随库规模的经验指数 b = ${f2(fit.b)}（仅 N≥1 点；b≈1 即每轮含 O(N) 扫描）`)
}
const withDrop = results.runs.find((r) => r.label === 'contradicts')
if (withDrop !== undefined) {
  console.log(`contradicts 变体软失效写：${int(withDrop.llmCalls.contradiction)} 次矛盾判定，落库 ${int(withDrop.records)} 条`)
}

writeFileSync(join(import.meta.dirname, 'results', 'write.json'), JSON.stringify(results, null, 2))
log('结果已写入 benchmark/results/write.json')

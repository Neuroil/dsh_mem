/**
 * Benchmark A · 批量插入（写入路径的存储侧成本）：
 *
 * 对 MemoryStore 逐条 putRecord（写入路径第 6 步）与 refreshNeighborhood
 * （第 7 步邻域统计回写，每条记忆的第二次介质写）在两种介质上按规模计时：
 *   - memory：内存 KvUnit，剥离持久化，测插件纯计算（索引维护、O(N) 扫描）；
 *   - sqlite：生产推荐后端（WAL，逐条 autocommit），测真实持久化吞吐。
 * 另测 MemoryStore.open 重开成本（同介质 loadAll + 镜像/索引重建，即宿主重启路径）。
 *
 * 用法：node bench-insert.mjs [--scales 1000,2000,...] [--sqlite-scales 500,...]
 */
import os from 'node:os'
import { writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import {
  benchConfig, memoryUnit, SqliteMedium, makeRecord, summarize, fitPowerLaw,
  mdTable, f2, f3, f1, int, log,
} from './support/harness.mjs'
import { generateCorpus } from './support/corpus.mjs'

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))
const MEMORY_SCALES = (args.scales ?? '1000,2000,5000,10000,20000').split(',').map(Number)
const SQLITE_SCALES = (args['sqlite-scales'] ?? '500,1000,2000,5000,10000').split(',').map(Number)

const config = benchConfig()
const embedding = createEmbeddingProvider(config.embedding)

/** 单规模一轮：建库 → 逐条 put → 逐条 refresh → close → 同介质重开。 */
async function runScale(medium, n, seedOffset) {
  const { texts } = generateCorpus(n, 42 + seedOffset)
  const t0 = performance.now()
  const vectors = await embedding.embed(texts)
  const embedMs = performance.now() - t0

  let unit
  let mediumObj = null
  if (medium === 'memory') {
    unit = memoryUnit()
  } else {
    mediumObj = new SqliteMedium(`insert_${n}`)
    unit = await mediumObj.openUnit('dsh_mem_bench', config)
  }
  let store = await MemoryStore.open(unit, createVectorIndex(), config)

  const putSamples = new Float64Array(n)
  const ids = new Array(n)
  const now = Date.now()
  for (let i = 0; i < n; i++) {
    const record = makeRecord({ text: texts[i], embedding: vectors[i], now })
    ids[i] = record.id
    const t = performance.now()
    await store.putRecord(record)
    putSamples[i] = performance.now() - t
  }

  // refresh 阶段只采样末尾 REFRESH_SAMPLE 条：索引此时已含全部 N 条，
  // 单次调用计时与"对全库逐条刷新"等价（后者正是睡眠周期第 7 步的模式，
  // 其全量成本 = 单次均值 × N，由 bench-sleep 单独覆盖）。
  const REFRESH_SAMPLE = Math.min(n, 200)
  const refreshSamples = new Float64Array(REFRESH_SAMPLE)
  for (let i = 0; i < REFRESH_SAMPLE; i++) {
    const t = performance.now()
    await store.refreshNeighborhood(ids[n - REFRESH_SAMPLE + i], config.retrieval.cslsK)
    refreshSamples[i] = performance.now() - t
  }

  // 重开：close 后对同一介质重新 open（sqlite 含全量读盘；memory 的 loadAll 走同进程 Map）
  await store.close()
  const t1 = performance.now()
  if (medium === 'sqlite') {
    unit = await mediumObj.openUnit('dsh_mem_bench', config)
  }
  const reopened = await MemoryStore.open(unit, createVectorIndex(), config)
  const reopenMs = performance.now() - t1
  const rows = reopened.allMemories().length
  await reopened.close()
  if (mediumObj !== null) {
    await mediumObj.close()
    await mediumObj.remove()
  }

  return { medium, n, rows, embedMs, put: summarize(putSamples), refresh: summarize(refreshSamples), reopenMs }
}

// ---------------------------------------------------------------- 执行

const results = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    dims: config.embedding.dimensions,
    cslsK: config.retrieval.cslsK,
    scales: { memory: MEMORY_SCALES, sqlite: SQLITE_SCALES },
  },
  runs: [],
}
log(`bench-insert 启动：memory scales=${MEMORY_SCALES.join(',')} / sqlite scales=${SQLITE_SCALES.join(',')}`)

for (const n of MEMORY_SCALES) {
  const r = await runScale('memory', n, 0)
  results.runs.push(r)
  log(`memory N=${int(r.n)}（rows=${r.rows}） put ${f3(r.put.mean)}ms refresh ${f3(r.refresh.mean)}ms reopen ${f1(r.reopenMs)}ms`)
}
for (const n of SQLITE_SCALES) {
  const r = await runScale('sqlite', n, 1000)
  results.runs.push(r)
  log(`sqlite N=${int(r.n)}（rows=${r.rows}） put ${f3(r.put.mean)}ms refresh ${f3(r.refresh.mean)}ms reopen ${f1(r.reopenMs)}ms`)
}

// ---------------------------------------------------------------- 汇总表

function tableFor(medium) {
  const rows = results.runs.filter((r) => r.medium === medium).map((r) => [
    int(r.n), f3(r.put.mean), f2(r.put.p95), int(r.put.opsPerSec),
    f3(r.refresh.mean), f2(r.refresh.p95), int(r.refresh.opsPerSec), f1(r.reopenMs),
  ])
  return mdTable(
    ['N', 'put mean(ms)', 'put p95(ms)', 'put ops/s', 'refresh mean(ms)', 'refresh p95(ms)', 'refresh ops/s', 'reopen(ms)'],
    rows,
  )
}

console.log('\n== memory 介质（纯计算：索引维护 + O(N) 近邻扫描） ==')
console.log(tableFor('memory'))
console.log('\n== sqlite 介质（生产后端：WAL，逐条 autocommit） ==')
console.log(tableFor('sqlite'))

console.log('\n== 经验复杂度指数 b（t = a·N^b；0≈O(1)，1≈O(N)，2≈O(N²)）与嵌入耗时 ==')
const fitRows = []
for (const medium of ['memory', 'sqlite']) {
  const runs = results.runs.filter((r) => r.medium === medium)
  const putFit = fitPowerLaw(runs.map((r) => ({ x: r.n, y: r.put.mean })))
  const refreshFit = fitPowerLaw(runs.map((r) => ({ x: r.n, y: r.refresh.mean })))
  const reopenFit = fitPowerLaw(runs.map((r) => ({ x: r.n, y: r.reopenMs })))
  fitRows.push([medium, f2(putFit.b), f2(refreshFit.b), f2(reopenFit.b), f1(runs.at(-1).embedMs)])
}
console.log(mdTable(['medium', 'put b', 'refresh b', 'reopen b', `embed maxN(ms)`], fitRows))

mkdirSync(join(import.meta.dirname, 'results'), { recursive: true })
writeFileSync(join(import.meta.dirname, 'results', 'insert.json'), JSON.stringify(results, null, 2))
log('结果已写入 benchmark/results/insert.json')

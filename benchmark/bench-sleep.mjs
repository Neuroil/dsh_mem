/**
 * Benchmark D · 睡眠巩固周期（离线批处理成本）：
 *
 *   1) buildClusters 随库规模：实现是全量 pairwise 余弦连边（O(N²·d)），
 *      直接量化其规模曲线与簇产出；
 *   2) SleepCycle.run 完整一轮（deterministic LLM 替身全放行）：
 *      聚类 → gist/QA/路由 → 迁移 → 第 7 步全库刷新（每条记录
 *      updateRecord + refreshNeighborhood，即 2N 次介质写 + N 次 O(N) 扫描）。
 *      memory/sqlite 双介质各跑一轮，拆分各阶段占比。
 *
 * clusterCos 校准：deterministic 嵌入的余弦分布与 bge-m3 不同（族内
 * p50≈0.40 vs 生产阈值 0.75），先采样族内/族间分布自动选阈值，保证
 * 聚类路径被真实执行。
 *
 * 用法：node bench-sleep.mjs [--scales 250,500,1000,2000] [--full 1000]
 */
import os from 'node:os'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import { SleepCycle, buildClusters } from '../lib/sleep-cycle.js'
import {
  benchConfig, memoryUnit, SqliteMedium, stubLlm, makeRecord, summarize,
  mdTable, f2, f3, f1, int, log,
} from './support/harness.mjs'
import { generateClusteredCorpus } from './support/corpus.mjs'

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))
const SCALES = (args.scales ?? '250,500,1000,2000').split(',').map(Number)
const FULL_N = Number(args.full ?? 1000)

// ------------------------------------------------ clusterCos 校准

/**
 * 采样组内/组间 pairwise 余弦分布，选能成簇的阈值（deterministic 嵌入口径）。
 * 关键约束：union-find 连通分量聚类对跨组边极敏感（少量随机边即把全库串成
 * 一个分量、随后被簇内均值闸门整体拒绝）——阈值须压到组间分布尾部（p99.9），
 * 同时不高于组内 p25（簇内均值闸门以组内均值衡量）。
 */
async function calibrateClusterCos(embedding) {
  const { texts, groupIdx } = generateClusteredCorpus(360, 999)
  const vectors = await embedding.embed(texts)
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }
  const intra = []
  const inter = []
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const c = dot(vectors[i], vectors[j])
      ;(groupIdx[i] === groupIdx[j] ? intra : inter).push(c)
    }
  }
  const q = (arr, p) => arr.sort((x, y) => x - y)[Math.floor(p * (arr.length - 1))]
  const floor = Math.max(q(inter, 0.999), q(inter, 0.995)) // 组间尾部：跨组边几乎清零
  const ceil = q(intra, 0.25) // 低于组内均值，保证簇内均值闸门可过
  const chosen = Math.round(Math.min(floor, ceil) * 100) / 100 // schema 要求 0.01 的倍数
  return { chosen, intraP50: q(intra, 0.5), intraP05: q(intra, 0.05), interP99: q(inter, 0.99), interTail: floor }
}

// ------------------------------------------------ 执行

const config = benchConfig()
const embedding = createEmbeddingProvider(config.embedding)
const cal = await calibrateClusterCos(embedding)
log(`clusterCos 校准：组内 p50=${f3(cal.intraP50)} 组间 p99=${f3(cal.interP99)}（尾部 ${f3(cal.interTail)}）→ 取 ${f3(cal.chosen)}（生产默认 0.75 对应真实嵌入）`)
const sleepCfg = benchConfig({ sleep: { clusterCos: cal.chosen } })

const results = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    dims: config.embedding.dimensions,
    sleepDefaults: { clusterCos: config.sleep.clusterCos, clusterMinSize: config.sleep.clusterMinSize, qaF1Threshold: config.sleep.qaF1Threshold },
    calibratedClusterCos: cal.chosen,
    calibration: cal,
    scales: SCALES,
    fullN: FULL_N,
  },
  clusterScaling: [],
  fullRuns: [],
}

// 1) buildClusters 规模曲线（memory 介质直连记录数组）
for (const n of SCALES) {
  const { texts } = generateClusteredCorpus(n, 555)
  const vectors = await embedding.embed(texts)
  const records = texts.map((t, i) => makeRecord({ text: t, embedding: vectors[i] }))
  const t = performance.now()
  const clusters = buildClusters(records, sleepCfg.sleep)
  const ms = performance.now() - t
  const sizes = clusters.map((c) => c.members.length)
  results.clusterScaling.push({
    n, ms,
    clusters: clusters.length,
    clusteredRecords: sizes.reduce((s, x) => s + x, 0),
    meanClusterSize: sizes.length === 0 ? 0 : sizes.reduce((s, x) => s + x, 0) / sizes.length,
    pairsPerSec: (n * (n - 1) / 2) / (ms / 1000),
  })
  log(`buildClusters N=${int(n)} ${f1(ms)}ms → ${clusters.length} 簇（均值 ${f1(results.clusterScaling.at(-1).meanClusterSize)} 条/簇，覆盖 ${results.clusterScaling.at(-1).clusteredRecords} 条）`)
}

// 2) SleepCycle.run 完整一轮：memory 与 sqlite 各一次
async function fullRun(medium) {
  const { texts } = generateClusteredCorpus(FULL_N, 555)
  const vectors = await embedding.embed(texts)
  let store
  let mediumObj = null
  let unitCounters = null
  if (medium === 'memory') {
    const unit = memoryUnit()
    unitCounters = unit.counters
    store = await MemoryStore.open(unit, createVectorIndex(), sleepCfg)
  } else {
    mediumObj = new SqliteMedium(`sleep_${FULL_N}`)
    const unit = await mediumObj.openUnit('dsh_mem_bench', sleepCfg)
    store = await MemoryStore.open(unit, createVectorIndex(), sleepCfg)
  }
  const now = Date.now()
  for (let i = 0; i < FULL_N; i++) {
    await store.putRecord(makeRecord({ text: texts[i], embedding: vectors[i], now }))
  }
  if (unitCounters !== null) unitCounters.puts = 0 // 从周期开始计介质写

  // 阶段拆分：聚类单独计时（run() 内部会再算一次，成本可忽略对比总时长）
  const pool = store.validMemories().filter((r) => r.flags.compressed_into === undefined)
  const t0 = performance.now()
  const clusters = buildClusters(pool, sleepCfg.sleep)
  const clusterMs = performance.now() - t0

  const llm = stubLlm()
  const cycle = new SleepCycle({ llm, embedding, store, config: sleepCfg, newId: (() => { let k = 0; return () => `node_bench_${k++}` })() })
  const t1 = performance.now()
  const outcome = await cycle.run()
  const totalMs = performance.now() - t1

  const writes = medium === 'memory' ? unitCounters.puts : null
  const row = {
    medium, n: FULL_N, totalMs, clusterMs,
    clustersFound: clusters.length,
    clustersProcessed: outcome.clusters.length,
    routes: tally(outcome.clusters.map((c) => c.route)),
    compressedMembers: outcome.clusters.filter((c) => c.route === 'new_node' || c.route === 'attached').reduce((s, c) => s + c.members.length, 0),
    nodesAfter: store.validNodes().length,
    refreshedRecords: outcome.refreshedRecords,
    mediaWrites: writes,
    llmCalls: { ...llm.calls },
  }
  await store.close()
  if (mediumObj !== null) {
    await mediumObj.close()
    await mediumObj.remove()
  }
  results.fullRuns.push(row)
  log(`SleepCycle.run [${medium}] N=${int(FULL_N)} 总 ${f1(totalMs)}ms（聚类 ${f1(clusterMs)}ms）→ 簇 ${row.clustersProcessed}（${JSON.stringify(row.routes)}）节点 ${row.nodesAfter} 刷新记录 ${row.refreshedRecords}${writes !== null ? ` 介质写 ${int(writes)}` : ''}`)
}

function tally(arr) {
  return arr.reduce((m, x) => { m[x] = (m[x] ?? 0) + 1; return m }, {})
}

await fullRun('memory')
await fullRun('sqlite')

// ------------------------------------------------ 汇总

console.log('\n== buildClusters 规模曲线（O(N²) pairwise 余弦连边） ==')
console.log(mdTable(
  ['N', '耗时(ms)', 'pair/s', '簇数', '覆盖记录', '均值簇大小'],
  results.clusterScaling.map((r) => [int(r.n), f1(r.ms), int(r.pairsPerSec), r.clusters, r.clusteredRecords, f1(r.meanClusterSize)]),
))

console.log('\n== SleepCycle.run 完整一轮（N=' + int(FULL_N) + '，LLM 替身全放行） ==')
console.log(mdTable(
  ['介质', '总耗时(ms)', '聚类(ms)', '簇(处理)', '路由分布', '压缩成员', '节点数', '第7步刷新记录', '介质写(memory)'],
  results.fullRuns.map((r) => [
    r.medium, f1(r.totalMs), f1(r.clusterMs), r.clustersProcessed, JSON.stringify(r.routes),
    r.compressedMembers, r.nodesAfter, int(r.refreshedRecords), r.mediaWrites === null ? '—' : int(r.mediaWrites),
  ]),
))

writeFileSync(join(import.meta.dirname, 'results', 'sleep.json'), JSON.stringify(results, null, 2))
log('结果已写入 benchmark/results/sleep.json')

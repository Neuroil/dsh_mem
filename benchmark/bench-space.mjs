/**
 * Benchmark E · 空间利用率与内存占用：
 *
 * sqlite（生产介质）三规模测：
 *   - 逻辑负载：记录紧凑 JSON 字节数（介质 value 列的真实内容口径）；
 *   - 物理体积：库文件 / 每表 payload 字节 / 页统计 / 文件·负载开销比；
 *   - 构成拆解：embedding（1024 维十进制文本）占记录字节比例 vs
 *     int8+base64 量化假设（TODO 既有候选项）的理论体积；
 *   - 增长行为：全量改写一遍（refreshNeighborhood）、10% 取代churn
 *     （新记录 + 旧记录软失效——记录永不删除）、clearAll 后 VACUUM 回收；
 *   - json 后端口径模拟（2 空格缩进 pretty JSON，近似；README 有真实后端数据）。
 * 另测内存镜像堆占用（memory 介质，宿主运行时常驻成本）。
 *
 * 用法：node bench-space.mjs [--scales 1000,5000,10000]
 */
import os from 'node:os'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import {
  benchConfig, memoryUnit, SqliteMedium, makeRecord, mdTable, f2, f3, f1, int, log,
} from './support/harness.mjs'
import { generateCorpus } from './support/corpus.mjs'

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))
const SCALES = (args.scales ?? '1000,5000,10000').split(',').map(Number)
const KB = 1024

const config = benchConfig()
const embedding = createEmbeddingProvider(config.embedding)

const results = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    dims: config.embedding.dimensions,
    scales: SCALES,
  },
  disk: [],
  growth: null,
  heap: null,
}

/** 逻辑负载统计（紧凑 JSON 口径，即 sqlite value 列内容）。 */
function logicalStats(records) {
  let totalBytes = 0
  let embeddingBytes = 0
  let textBytes = 0
  for (const r of records) {
    totalBytes += Buffer.byteLength(JSON.stringify(r), 'utf8')
    embeddingBytes += Buffer.byteLength(JSON.stringify(r.embedding), 'utf8')
    textBytes += Buffer.byteLength(r.text, 'utf8')
  }
  // int8 + base64 假设：1024 维 → 1024B Int8 → base64 ≈ 4/3 × dims 字符（含引号）
  const quantizedPerRecord = Math.ceil((4 / 3) * config.embedding.dimensions) + 2
  return {
    n: records.length,
    totalBytes, embeddingBytes, textBytes,
    perRecord: totalBytes / records.length,
    embeddingShare: embeddingBytes / totalBytes,
    quantizedHypothetical: (totalBytes - embeddingBytes) + records.length * quantizedPerRecord,
  }
}

/** 在一个 sqlite 文件上跑完一个规模的全部分析阶段。 */
async function diskScale(n) {
  const medium = new SqliteMedium(`space_${n}`)
  const { texts } = generateCorpus(n, 2024)
  const vectors = await embedding.embed(texts)

  const unit = await medium.openUnit('dsh_mem_bench', config)
  const store = await MemoryStore.open(unit, createVectorIndex(), config)
  const now = Date.now()
  const ids = []
  for (let i = 0; i < n; i++) {
    const record = makeRecord({ text: texts[i], embedding: vectors[i], now })
    ids.push(record.id)
    await store.putRecord(record)
  }
  const logical = logicalStats(store.allMemories())
  const prettyJsonBytes = Buffer.byteLength(JSON.stringify(store.allMemories(), null, 2), 'utf8')
  await store.close()
  await medium.close()

  const sizes1 = await medium.sizes()
  const insp1 = await medium.inspect('dsh_mem_bench')
  const row = {
    n,
    logical,
    prettyJsonBytes,
    fileBytes: sizes1.db,
    walBytes: sizes1.wal,
    payloadBytes: insp1.tables.episodic.payloadBytes,
    fileOverhead: insp1.fileBytes / Math.max(1, insp1.tables.episodic.payloadBytes),
    pageSize: insp1.pageSize, pageCount: insp1.pageCount, freelistPages: insp1.freelistPages,
  }
  results.disk.push(row)
  log(`sqlite N=${int(n)} 逻辑 ${int(row.logical.totalBytes / KB)}KB / 文件 ${int(row.fileBytes / KB)}KB / payload ${int(row.payloadBytes / KB)}KB / embedding 占比 ${f1(row.logical.embeddingShare * 100)}%`)

  // ---- 增长阶段（只在最大规模上做） ----
  if (n === SCALES.at(-1)) {
    // 阶段 1：全量改写一遍（r_m 刷新 pass；对 1000 条采样，改写语义同 N）
    const unit2 = await medium.openUnit('dsh_mem_bench', config, { fresh: false })
    const store2 = await MemoryStore.open(unit2, createVectorIndex(), config)
    const rewriteCount = Math.min(n, 1000)
    const rewriteIds = ids.slice(-rewriteCount)
    const tA = performance.now()
    for (const id of rewriteIds) await store2.refreshNeighborhood(id, config.retrieval.cslsK)
    const rewriteMs = performance.now() - tA
    await store2.close()
    await medium.close()
    const sizesRewrite = await medium.sizes()
    const inspRewrite = await medium.inspect('dsh_mem_bench')

    // 阶段 2：10% 取代 churn——新记录入库 + 旧记录软失效（记录永不删除 → 只增不减）
    const unit3 = await medium.openUnit('dsh_mem_bench', config, { fresh: false })
    const store3 = await MemoryStore.open(unit3, createVectorIndex(), config)
    const churn = Math.floor(n / 10)
    const { texts: newTexts } = generateCorpus(churn, 31337)
    const newVectors = await embedding.embed(newTexts)
    const tB = performance.now()
    let superseded = 0
    for (let i = 0; i < churn; i++) {
      const record = makeRecord({ text: newTexts[i], embedding: newVectors[i] })
      await store3.putRecord(record)
      await store3.softInvalidate(ids[i], record.id, now + i)
      superseded++
    }
    const churnMs = performance.now() - tB
    await store3.close()
    await medium.close()
    const sizesChurn = await medium.sizes()
    const inspChurn = await medium.inspect('dsh_mem_bench')

    // 阶段 3：clearAll 硬删除 → VACUUM 回收
    const unit4 = await medium.openUnit('dsh_mem_bench', config, { fresh: false })
    const store4 = await MemoryStore.open(unit4, createVectorIndex(), config)
    const tC = performance.now()
    const cleared = await store4.clearAll()
    const clearMs = performance.now() - tC
    await store4.close()
    await medium.close()
    const sizesCleared = await medium.sizes()
    const afterVacuum = await medium.vacuum()

    results.growth = {
      n,
      rewrite: { count: rewriteCount, ms: rewriteMs, fileBytes: sizesRewrite.db, walBytes: sizesRewrite.wal, freelistPages: inspRewrite.freelistPages },
      churn: { count: churn, superseded, ms: churnMs, fileBytes: sizesChurn.db, walBytes: sizesChurn.wal, freelistPages: inspChurn.freelistPages, rows: inspChurn.tables.episodic.rows },
      clear: { ms: clearMs, removed: cleared, fileBytesBeforeVacuum: sizesCleared.db, fileBytesAfterVacuum: afterVacuum },
    }
    log(`增长：改写 ${rewriteCount} 条 → ${int(sizesRewrite.db / KB)}KB；churn ${churn} 取代 → ${int(sizesChurn.db / KB)}KB（rows=${inspChurn.tables.episodic.rows}）；clearAll ${clearMs.toFixed(0)}ms → ${int(sizesCleared.db / KB)}KB → VACUUM 后 ${int(afterVacuum / KB)}KB`)
  }

  await medium.remove()
}

for (const n of SCALES) {
  await diskScale(n)
}

// ---- 内存镜像堆占用（memory 介质） ----
{
  const n = Math.max(...SCALES)
  const { texts } = generateCorpus(n, 2024)
  const vectors = await embedding.embed(texts)
  const beforeEmb = process.memoryUsage().heapUsed
  const store = await MemoryStore.open(memoryUnit(), createVectorIndex(), config)
  const now = Date.now()
  for (let i = 0; i < n; i++) {
    await store.putRecord(makeRecord({ text: texts[i], embedding: vectors[i], now }))
  }
  const afterBuild = process.memoryUsage().heapUsed
  // 重开一次（loadAll 结构化克隆 + 索引重建），测宿主重启后的常驻口径
  const unit2 = memoryUnit()
  const store2 = await MemoryStore.open(unit2, createVectorIndex(), config)
  await store2.close()
  results.heap = {
    n,
    buildDeltaPerRecord: (afterBuild - beforeEmb) / n,
    note: '含记录对象 + 索引归一化向量副本；不含嵌入原始数组（vectors 已在测前分配）',
  }
  log(`堆占用：每记录镜像 ≈ ${f1(results.heap.buildDeltaPerRecord)}B（N=${int(n)}）`)
  await store.close()
}

// ------------------------------------------------ 汇总

console.log('\n== 磁盘：逻辑负载 vs sqlite 物理 ==')
console.log(mdTable(
  ['N', '逻辑紧凑JSON', 'sqlite文件', 'payload列', '文件/payload', '每记录(文件B)', '每记录(逻辑B)', 'embedding占比'],
  results.disk.map((r) => [
    int(r.n), int(r.logical.totalBytes / KB) + 'KB', int(r.fileBytes / KB) + 'KB', int(r.payloadBytes / KB) + 'KB',
    f2(r.fileOverhead) + 'x', int(r.fileBytes / r.n), int(r.logical.perRecord), f1(r.logical.embeddingShare * 100) + '%',
  ]),
))

const maxDisk = results.disk.at(-1)
if (maxDisk !== undefined) {
  console.log('\n== 构成与假设（N=' + int(maxDisk.n) + '） ==')
  console.log(mdTable(
    ['口径', '体积', '相对当前逻辑负载'],
    [
      ['紧凑 JSON（当前 sqlite value 口径）', int(maxDisk.logical.totalBytes / KB) + 'KB', '1.00x'],
      ['int8+base64 量化嵌入（假设）', int(maxDisk.logical.quantizedHypothetical / KB) + 'KB', f2(maxDisk.logical.quantizedHypothetical / maxDisk.logical.totalBytes) + 'x'],
      ['json 后端模拟（2 空格 pretty）', int(maxDisk.prettyJsonBytes / KB) + 'KB', f2(maxDisk.prettyJsonBytes / maxDisk.logical.totalBytes) + 'x'],
    ],
  ))
}

if (results.growth !== null) {
  const g = results.growth
  console.log('\n== 增长行为（N=' + int(g.n) + '，同一库文件连续阶段） ==')
  const initialFile = results.disk.find((d) => d.n === g.n)?.fileBytes ?? 0
  console.log(mdTable(
    ['阶段', '写入量', '耗时(ms)', '库文件', 'WAL', 'freelist页'],
    [
      ['初始插入', int(g.n) + ' 条', '—', int(initialFile / KB) + 'KB', '—', '—'],
      ['全量改写(r_m刷新)', g.rewrite.count + ' 条', f1(g.rewrite.ms), int(g.rewrite.fileBytes / KB) + 'KB', int(g.rewrite.walBytes / KB) + 'KB', g.rewrite.freelistPages],
      ['10% 取代 churn', g.churn.count + ' 新 + ' + g.churn.superseded + ' 失效', f1(g.churn.ms), int(g.churn.fileBytes / KB) + 'KB', int(g.churn.walBytes / KB) + 'KB', g.churn.freelistPages],
      ['clearAll', int(g.clear.removed.episodic) + ' 条硬删', f1(g.clear.ms), int(g.clear.fileBytesBeforeVacuum / KB) + 'KB', '—', '—'],
      ['VACUUM 后', '—', '—', int(g.clear.fileBytesAfterVacuum / KB) + 'KB', '—', '—'],
    ],
  ))
}

if (results.heap !== null) {
  console.log(`\n内存镜像：每记录 ≈ ${f1(results.heap.buildDeltaPerRecord)}B 堆（${results.heap.note}）`)
}

writeFileSync(join(import.meta.dirname, 'results', 'space.json'), JSON.stringify(results, null, 2))
log('结果已写入 benchmark/results/space.json')

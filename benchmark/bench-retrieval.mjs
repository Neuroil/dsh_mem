/**
 * Benchmark B · 大量检索：
 *
 * 检索全部发生在内存镜像 + 内存索引上（介质只影响 open，见 bench-insert 的
 * reopen 列），故本基准用 memory 介质，按规模测两条检索通道：
 *   - MemoryStore.similar(k)   写入路径矛盾判定的近邻视图（同步精确 kNN）；
 *   - MemoryRetriever.retrieve 四级检索管线（L0→CSLS→精排→MMR，生产通道）。
 * 末尾对最大规模做阶段归因（retrievable 物化 / 查询嵌入 / L0 余弦全扫 + 排序）。
 *
 * 用法：node bench-retrieval.mjs [--scales 1000,...] [--queries 100]
 */
import os from 'node:os'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { MemoryStore } from '../lib/memory-store.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { createEmbeddingProvider, createRerankerProvider } from '../lib/embedding.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { cosine } from '../lib/metrics.js'
import {
  benchConfig, memoryUnit, makeRecord, summarize, fitPowerLaw,
  mdTable, f2, f3, f1, int, log,
} from './support/harness.mjs'
import { generateCorpus, generateQueries } from './support/corpus.mjs'

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))
const SCALES = (args.scales ?? '1000,2000,5000,10000,20000').split(',').map(Number)
const QCOUNT = Number(args.queries ?? 100)

const config = benchConfig()
const embedding = createEmbeddingProvider(config.embedding)
const reranker = createRerankerProvider(config.reranker, embedding)

// 查询集：全族覆盖（12 族 × 2 条），按 QCOUNT 循环扩展
const { queries, familyKeys } = generateQueries()
const allQueries = []
const allFamilies = []
for (let i = 0; i < QCOUNT; i++) {
  allQueries.push(queries[i % queries.length])
  allFamilies.push(familyKeys[i % queries.length])
}

const results = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    dims: config.embedding.dimensions,
    queries: QCOUNT,
    retrieval: config.retrieval,
    scales: SCALES,
  },
  runs: [],
  attribution: null,
}
log(`bench-retrieval 启动：scales=${SCALES.join(',')} queries=${QCOUNT}`)

for (const n of SCALES) {
  const { texts } = generateCorpus(n, 4242)
  const vectors = await embedding.embed(texts)
  const store = await MemoryStore.open(memoryUnit(), createVectorIndex(), config)
  const now = Date.now()
  for (let i = 0; i < n; i++) {
    await store.putRecord(makeRecord({ text: texts[i], embedding: vectors[i], now }))
  }
  const retriever = new MemoryRetriever({ embedding, reranker, lookup: store, config: config.retrieval })

  // 查询向量预嵌入（与检索计时分离）
  const queryVectors = await embedding.embed(allQueries)

  // A：similar（写入路径近邻视图）
  const similarSamples = new Float64Array(QCOUNT)
  for (let q = 0; q < QCOUNT; q++) {
    const t = performance.now()
    store.similar(queryVectors[q], 5)
    similarSamples[q] = performance.now() - t
  }

  // B：四级检索管线
  const retrieveSamples = new Float64Array(QCOUNT)
  for (let q = 0; q < QCOUNT; q++) {
    const t = performance.now()
    await retriever.retrieve(allQueries[q])
    retrieveSamples[q] = performance.now() - t
  }

  // 归因（仅最大规模）：retrievable 物化 / 查询嵌入 / L0 余弦全扫+排序副本
  if (n === SCALES.at(-1)) {
    const ATTR_REPEAT = 50
    const t1 = performance.now()
    for (let r = 0; r < ATTR_REPEAT; r++) store.retrievable()
    const retrievableMs = (performance.now() - t1) / ATTR_REPEAT
    const t2 = performance.now()
    for (let r = 0; r < ATTR_REPEAT; r++) await embedding.embed([allQueries[0]])
    const embedMs = (performance.now() - t2) / ATTR_REPEAT
    const items = store.retrievable()
    const qv = queryVectors[0]
    const t3 = performance.now()
    for (let r = 0; r < ATTR_REPEAT; r++) {
      const scored = items.map((item) => ({ id: item.id, cos: cosine(qv, item.embedding) }))
      scored.sort((a, b) => b.cos - a.cos)
      scored.slice(0, config.retrieval.l0TopK)
    }
    const l0ReplicaMs = (performance.now() - t3) / ATTR_REPEAT
    const t4 = performance.now()
    for (let r = 0; r < ATTR_REPEAT; r++) {
      const docs = items.slice(0, config.retrieval.cslsTopK).map((x) => x.text)
      await reranker.rerank(allQueries[0], docs)
    }
    const rerankMs = (performance.now() - t4) / ATTR_REPEAT
    results.attribution = {
      n, retrievableMs, embedMs, l0ReplicaMs, rerankMs,
      itemsCount: items.length,
    }
  }

  results.runs.push({
    n,
    similar: summarize(similarSamples),
    retrieve: summarize(retrieveSamples),
  })
  log(`N=${int(n)} similar ${f3(results.runs.at(-1).similar.mean)}ms retrieve ${f3(results.runs.at(-1).retrieve.mean)}ms QPS=${int(1000 / results.runs.at(-1).retrieve.mean)}`)
  await store.close()
}

// ---------------------------------------------------------------- 汇总

console.log('\n== 检索性能随库规模（memory 介质；生产检索全在内存镜像上） ==')
console.log(mdTable(
  ['N', 'similar mean(ms)', 'similar p95(ms)', 'retrieve mean(ms)', 'retrieve p95(ms)', 'retrieve QPS'],
  results.runs.map((r) => [
    int(r.n), f3(r.similar.mean), f2(r.similar.p95), f3(r.retrieve.mean), f2(r.retrieve.p95), int(1000 / r.retrieve.mean),
  ]),
))

console.log('\n== 经验复杂度指数 b（单次调用时延 t = a·N^b） ==')
const similarFit = fitPowerLaw(results.runs.map((r) => ({ x: r.n, y: r.similar.mean })))
const retrieveFit = fitPowerLaw(results.runs.map((r) => ({ x: r.n, y: r.retrieve.mean })))
console.log(mdTable(['通道', 'b', 'R²'], [['similar(k=5)', f2(similarFit.b), f3(similarFit.r2)], ['retrieve(full)', f2(retrieveFit.b), f3(retrieveFit.r2)]]))

if (results.attribution !== null) {
  const a = results.attribution
  console.log(`\n== 阶段归因 @N=${int(a.n)}（单次 ms，retrieve 实测均值见上表） ==`)
  console.log(mdTable(
    ['阶段', 'ms', '说明'],
    [
      ['retrievable() 物化', f3(a.retrievableMs), `产出 ${int(a.itemsCount)} 条 L0 候选（含 embedding 引用）`],
      ['查询嵌入(deterministic)', f3(a.embedMs), '生产为一次 HTTP 调用（数十 ms 量级，不在本基准）'],
      ['L0 余弦全扫+排序（复刻）', f3(a.l0ReplicaMs), `${int(a.itemsCount)}×1024 维余弦 + sort`],
      ['精排 rerank(deterministic)', f3(a.rerankMs), `${config.retrieval.cslsTopK} 文档；生产为 HTTP`],
    ],
  ))
}

writeFileSync(join(import.meta.dirname, 'results', 'retrieval.json'), JSON.stringify(results, null, 2))
log('结果已写入 benchmark/results/retrieval.json')

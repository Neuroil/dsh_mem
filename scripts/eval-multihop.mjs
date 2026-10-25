/**
 * PPR 确定性联想离线评测（P3-3 验证，HippoRAG 式多跳）：
 * 合成链式语料——每条链 锚点a — 桥b — 目标c 用共享词袋连接（a↔b 共享
 * anchorB、b↔c 共享 bridge），查询只含锚点词。直接余弦下目标 c 与查询无
 * 共享词（≈0），只能经图上两跳被 PPR 抬进 top。同一库分别以
 * pprWeight=0（纯语义基线）与配置值（联想开）跑 recall@k 对比。
 *
 *   node scripts/eval-multihop.mjs [--chains 30] [--distractors 60] [--topk 5] [--ppr-weight 0.2]
 *
 * 数值基于 deterministic 嵌入（哈希词袋），只度量排序行为，无语义意义；
 * 换真实端点跑同结构语料即得真实部署数字。
 */
import { Config } from '../lib/config.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { createEmbeddingProvider, createRerankerProvider } from '../lib/embedding.js'

// ---- 参数 ----
const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]] : []).filter(([, v]) => v !== undefined))
const chainCount = Number(args['chains'] ?? 30)
const distractorCount = Number(args['distractors'] ?? 60)
const topK = Number(args['topk'] ?? 5)
const pprWeight = Number(args['ppr-weight'] ?? 0.2)

// ---- 合成链式语料 ----
// 每链三个记录：a=锚点（查询共享 anchorA/anchorB 两词）、b=桥（共享 anchorB 与
// bridge）、c=目标（只共享 bridge——与查询零共享词）。links 手工铺 a-b / b-c。
const config = Config({
  storage: { backend: 'json', unit: 'eval_multihop', version: 1, workspaceScope: false },
  embedding: { mode: 'deterministic' },
  reranker: { mode: 'deterministic' },
  retrieval: { pprWeight },
})

const embedder = createEmbeddingProvider(config.embedding)

/** 离线检索池替身：与 eval-emotion 同款，外加 links → neighbors 投影（PPR 消费）。 */
class OfflineLookup {
  constructor(records) { this.records = records }
  retrievable() {
    return this.records.map(r => ({
      id: r.id, text: r.text, embedding: r.embedding, r_m: 0, tier: 'warm',
      createdAt: 0, layer: 'episodic',
      neighbors: [...new Set(r.links.map(l => l.target))],
    }))
  }
}

const records = []
for (let i = 0; i < chainCount; i++) {
  const ids = { a: `c${i}_a`, b: `c${i}_b`, c: `c${i}_c` }
  records.push({ id: ids.a, text: `anchorA${i} anchorB${i}`, links: [{ kind: 'related', target: ids.b }] })
  records.push({ id: ids.b, text: `anchorB${i} bridge${i}`, links: [{ kind: 'related', target: ids.a }, { kind: 'related', target: ids.c }] })
  records.push({ id: ids.c, text: `bridge${i} goal${i}`, links: [{ kind: 'related', target: ids.b }] })
}
for (let i = 0; i < distractorCount; i++) {
  records.push({ id: `d${String(i).padStart(3, '0')}`, text: `distract${i} filler${i}`, links: [] })
}
for (const record of records) {
  record.embedding = (await embedder.embed([record.text]))[0]
}

const queries = []
for (let i = 0; i < chainCount; i++) {
  queries.push({ text: `anchorA${i} anchorB${i}`, target: `c${i}_c` })
}

// ---- 双臂评测 ----
function makeRetriever(weight) {
  return new MemoryRetriever({
    embedding: embedder,
    reranker: createRerankerProvider(config.reranker, embedder),
    lookup: new OfflineLookup(records),
    config: { ...config.retrieval, pprWeight: weight },
  })
}

async function evaluate(weight) {
  const retriever = makeRetriever(weight)
  let hits = 0
  let rankSum = 0
  let missRank = 0
  for (const query of queries) {
    const result = await retriever.retrieve(query.text, { topK })
    const rank = result.top.findIndex(c => c.id === query.target)
    if (rank >= 0) { hits++; rankSum += rank + 1 } else missRank++
  }
  return {
    recall: hits / queries.length,
    meanHitRank: hits > 0 ? rankSum / hits : Number.NaN,
    missed: missRank,
  }
}

const baseline = await evaluate(0)
const withPpr = await evaluate(pprWeight)

const pct = (x) => `${(x * 100).toFixed(1)}%`
console.log(`chains=${chainCount} distractors=${distractorCount} top-${topK} pprWeight=${pprWeight} ` +
  `(seedK=${config.retrieval.pprSeedK} damping=${config.retrieval.pprDamping}；deterministic 嵌入)\n`)
console.log(`纯语义（pprWeight=0）: recall@${topK} = ${pct(baseline.recall)}，命中均值排名 = ${baseline.meanHitRank.toFixed(1)}，未命中 ${baseline.missed}`)
console.log(`PPR 联想（pprWeight=${pprWeight}）: recall@${topK} = ${pct(withPpr.recall)}，命中均值排名 = ${withPpr.meanHitRank.toFixed(1)}，未命中 ${withPpr.missed}`)
console.log('\n判读：目标记录与查询零共享词，纯语义臂只能靠哈希碰撞噪声偶中；PPR 臂应显著抬升 recall。' +
  '若两臂相同，检查语料 links 是否进入 neighbors 投影（PPR 图无边时精确跳过）。')

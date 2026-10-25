/**
 * 情绪重排离线消融（P3-1 验证）：合成带 valence/arousal 的记忆库，
 * 同一组查询分别在 无mood / 负面mood / 正面mood 下检索，对比 top-5 重叠率
 * 与情绪契合度（deterministic 嵌入，数值仅度量排序行为，无语义意义）。
 *
 *   node scripts/eval-emotion.mjs [--alpha 0.15] [--records 200] [--queries 40]
 */
import { Config } from '../lib/config.js'
import { MemoryStore, memoryUnitDescriptor } from '../lib/memory-store.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { createEmbeddingProvider } from '../lib/embedding.js'
import { createVectorIndex } from '../lib/vector-store.js'

// ---- 参数 ----
const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) =>
  a.startsWith('--') ? [a.slice(2), all[i + 1]] : []).filter(([, v]) => v !== undefined))
const alpha = Number(args['alpha'] ?? 0.15)
const recordCount = Number(args['records'] ?? 200)
const queryCount = Number(args['queries'] ?? 40)

// ---- 合成库：文本 = 主题词组合（哈希词袋决定余弦结构），情绪 = 均匀采样 ----
const TOPICS = ['database', 'deploy', 'frontend', 'testing', 'meeting', 'budget', 'hiring', 'design']
const MOODS = [
  'frustrating', 'tiring',            // 负面低唤醒
  'annoying', 'stressful',            // 负面高唤醒
  'fine', 'routine',                  // 中性
  'pleasant', 'satisfying',           // 正面低唤醒
  'exciting', 'delightful',           // 正面高唤醒
]
const VALENCE = { frustrating: -0.7, tiring: -0.4, annoying: -0.6, stressful: -0.5, fine: 0, routine: 0.05, pleasant: 0.5, satisfying: 0.4, exciting: 0.8, delightful: 0.7 }
const AROUSAL = { frustrating: 0.6, tiring: 0.2, annoying: 0.7, stressful: 0.8, fine: 0.3, routine: 0.2, pleasant: 0.4, satisfying: 0.3, exciting: 0.9, delightful: 0.7 }

const config = Config({
  storage: { backend: 'json', unit: 'eval_emotion', version: 1, workspaceScope: false },
  embedding: { mode: 'deterministic' },
  reranker: { mode: 'deterministic' },
  emotion: { alpha },
})

// MemoryStore 需要真实 KvUnit——离线脚本直接用 json 后端文件太重，
// 这里手写最小只写存储替身（putRecord 即收，retrievable 由记录投影）。
class OfflineLookup {
  constructor(records) { this.records = records }
  retrievable() {
    return this.records.map(r => ({
      id: r.id, text: r.text, embedding: r.embedding, r_m: r.r_m, tier: r.tier,
      createdAt: r.time_meta.created_at, layer: 'episodic',
      valence: r.valence, arousal: r.arousal,
    }))
  }
}

const embedder = createEmbeddingProvider(config.embedding)
const store = []
for (let i = 0; i < recordCount; i++) {
  const topic = TOPICS[i % TOPICS.length]
  const mood = MOODS[(i * 7) % MOODS.length]
  const text = `The ${topic} work on milestone ${Math.floor(i / TOPICS.length)} was ${mood}`
  store.push({
    id: `m${String(i).padStart(4, '0')}`,
    text,
    embedding: await (await embedder.embed([text]))[0],
    valence: VALENCE[mood], arousal: AROUSAL[mood],
    r_m: 0, tier: 'warm', time_meta: { created_at: 1000 },
  })
}

const retriever = new MemoryRetriever({
  embedding: embedder,
  reranker: createRerankerStub(embedder),
  lookup: new OfflineLookup(store),
  config: config.retrieval,
})

const queries = []
for (let q = 0; q < queryCount; q++) {
  queries.push(`How is the ${TOPICS[q % TOPICS.length]} work going?`)
}

const moods = {
  none: undefined,
  negative: { state: { valence: -0.6, arousal: 0.7, updated_at: Date.now() }, alpha },
  positive: { state: { valence: 0.6, arousal: 0.6, updated_at: Date.now() }, alpha },
}

const meanValence = (ids) => {
  if (ids.length === 0) return 0
  return ids.reduce((s, id) => s + (store.find(m => m.id === id)?.valence ?? 0), 0) / ids.length
}

console.log(`records=${recordCount} queries=${queryCount} alpha=${alpha} (deterministic 嵌入，余弦结构由词袋决定)\n`)
const baselineTop5 = {}
for (const [name, mood] of Object.entries(moods)) {
  let overlapSum = 0
  let valSum = 0
  for (const query of queries) {
    const result = await retriever.retrieve(query, { topK: 5, ...(mood !== undefined ? { mood } : {}) })
    const ids = result.top.map(c => c.id)
    if (name === 'none') baselineTop5[query] = ids
    else {
      const base = baselineTop5[query] ?? []
      overlapSum += ids.filter(id => base.includes(id)).length / 5
    }
    valSum += meanValence(ids)
  }
  const avgVal = (valSum / queries.length).toFixed(3)
  if (name === 'none') {
    console.log(`无 mood   : top-5 平均 valence = ${avgVal}`)
  } else {
    const overlap = (overlapSum / queries.length * 100).toFixed(1)
    console.log(`${name.padEnd(8)}: top-5 平均 valence = ${avgVal}，与无 mood 的 top-5 重叠率 = ${overlap}%`)
  }
}
console.log('\n判读：negative 应显著拉低 top-5 平均 valence、positive 反之；重叠率过高（>95%）说明 α 影响不足，过低（<40%）说明重排过强。')

function createRerankerStub(embedding) {
  return {
    async rerank(query, documents) {
      const vectors = await embedding.embed([query, ...documents])
      const [qv] = vectors
      return vectors.slice(1).map(dv => {
        let dot = 0
        for (let i = 0; i < qv.length; i++) dot += qv[i] * (dv[i] ?? 0)
        return Math.min(1, Math.max(0, (dot + 1) / 2))
      })
    },
  }
}

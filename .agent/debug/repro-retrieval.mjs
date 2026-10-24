// 离线复现：用真实 dsh_mem.json 跑插件的检索链路，定位"始终返回空"的位置。
import { readFileSync } from 'node:fs'
const lib = 'file:///C:/Users/Neuroil/Code/dsh_mem/lib/'
const { MemoryStore } = await import(lib + 'memory-store.js')
const { MemoryRetriever } = await import(lib + 'retrieval.js')
const { createEmbeddingProvider, createRerankerProvider } = await import(lib + 'embedding.js')
const { createVectorIndex } = await import(lib + 'vector-store.js')
const { Config } = await import(lib + 'config.js')

const cfg = Config({
  storage: { backend: 'json', unit: 'dsh_mem' },
  embedding: { mode: 'deterministic' },
  reranker: { mode: 'deterministic' },
  llm: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
})

const doc = JSON.parse(readFileSync('C:/Users/Neuroil/.dsh/storages/dsh_mem.json', 'utf8'))
console.log('file unit:', JSON.stringify(doc.unit), '| episodic:', Object.keys(doc.tables.episodic || {}).length)

const unit = {
  async loadAll() { return { tables: doc.tables, global: doc.global ?? null } },
  async putRecord() {}, async deleteRecord() {}, async close() {},
}
const store = await MemoryStore.open(unit, createVectorIndex(), cfg)
console.log('store.allMemories:', store.allMemories().length, '| valid:', store.validMemories().length)
const items = store.retrievable()
console.log('retrievable items:', items.length)
if (items.length > 0) {
  const it = items[0]
  console.log('first item: id=' + String(it.id).slice(0, 8), 'tier=' + it.tier, 'layer=' + it.layer, 'dim=' + it.embedding.length, 'text=' + String(it.text).slice(0, 40))
}

const embedding = createEmbeddingProvider(cfg.embedding)
const reranker = createRerankerProvider(cfg.reranker, embedding)
const retriever = new MemoryRetriever({ embedding, reranker, lookup: store, config: cfg.retrieval })

const q = "dsh 0.1.2 要求 storage unit 名匹配"
const r = await retriever.retrieve(q, { recentTurns: [], topK: 3 })
for (const [name, arr] of Object.entries(r.stages)) console.log('stage', name, '=', arr.length)
console.log('top:', r.top.length, '| suppressed:', r.suppressed.length)
for (const c of r.top) console.log('  hit', c.id.slice(0, 8), 'cos=' + c.cos.toFixed(3), 'cue=' + (c.cue ?? 0).toFixed(3), 'rel=' + (c.relevance ?? 0).toFixed(3), String(c.item.text).slice(0, 40))

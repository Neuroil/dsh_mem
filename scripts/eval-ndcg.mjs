/**
 * nDCG four-rung ablation scaffold (design §7.1).
 *
 *   corpus:  JSONL, one {"id","text"} per line
 *   queries: JSONL, one {"query","relevant":[id...]} per line (graded
 *            relevance optional: {"relevant":{"id":2,...}})
 *
 * Runs the retrieval chain at every ablation rung — base cosine → +CSLS →
 * +cross-encoder rerank → +MMR — and reports nDCG@10 / recall@50 per rung.
 * Embeddings default to the deterministic offline provider; point
 * --embedding-mode http at a real endpoint for meaningful numbers.
 *
 * Usage:
 *   npm run build
 *   node scripts/eval-ndcg.mjs --corpus data/corpus.jsonl --queries data/queries.jsonl
 */
import { readFileSync } from 'node:fs'
import { createEmbeddingProvider, createRerankerProvider } from '../lib/embedding.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { MemoryStore } from '../lib/memory-store.js'

// ---- minimal argv ----
const args = Object.fromEntries(process.argv.slice(2).map((arg, i, all) =>
  arg.startsWith('--') ? [arg.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))

const corpusPath = args.corpus ?? 'data/corpus.jsonl'
const queriesPath = args.queries ?? 'data/queries.jsonl'
const ndcgAt = Number(args['ndcg-at'] ?? 10)
const recallAt = Number(args['recall-at'] ?? 50)

// ---- config: mirrors src/config.ts defaults ----
const config = {
  storage: { backend: 'json', unit: 'dsh-mem', version: 1 },
  embedding: {
    mode: args['embedding-mode'] ?? 'deterministic',
    endpoint: args['embedding-endpoint'] ?? 'https://api.siliconflow.cn/v1/embeddings',
    model: args['embedding-model'] ?? 'BAAI/bge-m3',
    apiKey: args['embedding-api-key'] ?? process.env.EMBEDDING_API_KEY ?? '',
    timeoutMs: 30000,
    dimensions: Number(args['embedding-dims'] ?? 1024),
    batchSize: 32,
  },
  reranker: {
    mode: args['reranker-mode'] ?? 'deterministic',
    endpoint: args['reranker-endpoint'] ?? 'https://api.siliconflow.cn/v1/rerank',
    model: args['reranker-model'] ?? 'BAAI/bge-reranker-v2-m3',
    apiKey: args['reranker-api-key'] ?? process.env.RERANKER_API_KEY ?? '',
    timeoutMs: 30000,
  },
  retrieval: {
    l0TopK: Number(args['l0-topk'] ?? 100),
    cslsTopK: 20,
    rerankTopK: 8,
    mmrTopK: 5,
    mmrLambda: 0.6,
    cslsK: 10,
    contextTurns: 0,
    coldCueBoost: 0.1,
    cueThreshold: 0.3,
  },
  tier: { hotThreshold: 8, coldThreshold: 2 },
}

// ---- load dataset ----
function readJsonl(path) {
  return readFileSync(path, 'utf8').split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line))
}

const corpus = readJsonl(corpusPath)
const queries = readJsonl(queriesPath)

// ---- in-memory KvUnit stand-in (evaluation never touches a real medium) ----
function fakeUnit() {
  const tables = { episodic: {}, semantic: {} }
  return {
    async loadAll() { return { tables: structuredClone(tables), global: null } },
    async putRecord(table, key, value) { tables[table][key] = value },
    async deleteRecord(table, key) { delete tables[table][key] },
    async setGlobal() { throw new Error('no global slot') },
    async close() {},
  }
}

const embedding = createEmbeddingProvider(config.embedding)
const reranker = createRerankerProvider(config.reranker, embedding)

// Embed corpus in batches, then build records with neighborhood stats.
const vectors = await embedding.embed(corpus.map((entry) => entry.text))
const store = await MemoryStore.open(fakeUnit(), createVectorIndex(), config)
let seq = 0
for (const [i, entry] of corpus.entries()) {
  const id = entry.id ?? `m${i}`
  const now = Date.now()
  await store.putRecord({
    id,
    text: entry.text,
    embedding: vectors[i],
    imp: 5,
    valence: 0,
    arousal: 0,
    epi: 0,
    red: 0,
    r_m: 0,
    tier: 'warm',
    con_route: { kind: 'pending' },
    links: [],
    S: 5,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: now, valid_at: now },
    flags: {},
    origin: { session_id: 'eval', turn: 0, seq: seq++ },
  })
}
for (const entry of corpus) await store.refreshNeighborhood(entry.id, config.retrieval.cslsK)

const retriever = new MemoryRetriever({ embedding, reranker, lookup: store, config: config.retrieval })

// ---- metrics ----
function graded(q) {
  if (Array.isArray(q.relevant)) return Object.fromEntries(q.relevant.map((id) => [id, 1]))
  return q.relevant ?? {}
}

function ndcgAtK(ranking, rel, at) {
  const k = Math.min(at, ranking.length)
  const dcgValue = ranking.slice(0, k).reduce((sum, id, i) => sum + (rel[id] ?? 0) / Math.log2(i + 2), 0)
  const ideal = Object.values(rel).sort((a, b) => b - a).slice(0, k)
  const idcg = ideal.reduce((sum, v, i) => sum + v / Math.log2(i + 2), 0)
  return idcg === 0 ? 0 : dcgValue / idcg
}

const RUNGS = ['base', 'csls', 'rerank', 'full']
const sums = Object.fromEntries(RUNGS.map((rung) => [rung, { ndcg: 0, recall: 0 }]))

for (const q of queries) {
  const rel = graded(q)
  const relIds = new Set(Object.keys(rel))
  for (const rung of RUNGS) {
    const result = await retriever.retrieve(q.query, { ablation: rung, topK: ndcgAt })
    const ranking = result.top.map((c) => c.id)
    sums[rung].ndcg += ndcgAtK(ranking, rel, ndcgAt)
    const pool = result.stages.l0.map((c) => c.id)
    const hits = pool.filter((id) => relIds.has(id)).length
    sums[rung].recall += relIds.size === 0 ? 0 : Math.min(1, hits / Math.min(relIds.size, recallAt))
  }
}

console.log(`corpus=${corpus.length} queries=${queries.length} embedding=${config.embedding.mode} reranker=${config.reranker.mode}`)
console.log('rung\tnDCG@' + ndcgAt + '\trecall@' + recallAt + '(L0 pool)')
for (const rung of RUNGS) {
  console.log(`${rung}\t${(sums[rung].ndcg / queries.length).toFixed(4)}\t${(sums[rung].recall / queries.length).toFixed(4)}`)
}

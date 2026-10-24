/**
 * Sleep-cycle consolidation scaffold (design §6 P1 verification: QA retention,
 * long-range answer survival after compression).
 *
 *   corpus:  JSONL, one {"id","text"} per line (same format as eval-ndcg)
 *   queries: JSONL, one {"query","relevant":[id...]} per line
 *
 * Pipeline: build records → baseline nDCG@10 → run one SleepCycle → nDCG@10
 * again with relevance remapped through compression (a relevant member hit now
 * counts via the semantic node that absorbed it). The LLM is a scripted echo
 * stand-in and embeddings default to the deterministic offline provider — the
 * numbers only prove the pipeline executes end to end; point --embedding-mode
 * http at a real endpoint for meaningful semantics.
 *
 * Usage:
 *   npm run build
 *   node scripts/eval-sleep.mjs --corpus data/corpus.jsonl --queries data/queries.jsonl \
 *     --cluster-min-size 2
 */
import { readFileSync } from 'node:fs'
import { createEmbeddingProvider, createRerankerProvider } from '../lib/embedding.js'
import { epiLexical } from '../lib/judges.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { SleepCycle } from '../lib/sleep-cycle.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { MemoryStore } from '../lib/memory-store.js'

// ---- minimal argv ----
const args = Object.fromEntries(process.argv.slice(2).map((arg, i, all) =>
  arg.startsWith('--') ? [arg.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))

const corpusPath = args.corpus ?? 'data/corpus.jsonl'
const queriesPath = args.queries ?? 'data/queries.jsonl'
const ndcgAt = Number(args['ndcg-at'] ?? 10)

// ---- config: mirrors src/config.ts defaults, sleep section overridable ----
const config = {
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: false },
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
  llm: { provider: 'echo', model: 'echo', temperature: 0, maxTokens: 2048 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: {
    l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6,
    cslsK: 10, contextTurns: 0, coldCueBoost: 0.1, cueThreshold: 0.3,
  },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: {
    enabled: true,
    minNewMemories: 10,
    clusterCos: Number(args['cluster-cos'] ?? 0.75),
    clusterMinSize: Number(args['cluster-min-size'] ?? 3),
    qaPairCount: 3,
    qaF1Threshold: Number(args['qa-f1-threshold'] ?? 0.8),
    epiThreshold: Number(args['epi-threshold'] ?? 0.99),
    conWindow: 1000,
    conAttachSigma: 0.5,
    conNewNodeSigma: -1,
    conNewNodeMinImp: 7,
    attachGain: 0.5,
  },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
}

// ---- scripted echo LLM: gist/QA/answers derived from the prompt itself ----
class EchoLlm {
  async complete(system, user) {
    const firstBullet = () => user.split('\n').find((line) => line.startsWith('- '))?.slice(2) ?? 'content'
    if (system.includes('semantic core')) {
      return JSON.stringify({ gist: `GIST: ${firstBullet()}` })
    }
    if (system.includes('question-answer pairs')) {
      return JSON.stringify([{ question: 'What does this cluster say?', answer: firstBullet() }])
    }
    if (system.includes('ONLY the provided summary')) {
      const match = user.match(/Summary statement: (.*)\n\nQuestion:/s)
      return JSON.stringify({ answer: match === null ? 'unknown' : match[1] })
    }
    if (system.includes('folded into')) {
      return JSON.stringify({ compatible: true })
    }
    throw new Error(`echo llm: unexpected call: ${system.slice(0, 60)}`)
  }
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
    imp: Number(entry.imp ?? 5),
    valence: 0,
    arousal: 0,
    epi: epiLexical(entry.text),
    red: 0,
    r_m: 0,
    tier: 'warm',
    con_route: { kind: 'pending' },
    links: [],
    S: Number(entry.imp ?? 5),
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: now, valid_at: now },
    flags: {},
    origin: { session_id: 'eval', turn: 0, seq: seq++ },
  })
}
for (const [i, entry] of corpus.entries()) {
  await store.refreshNeighborhood(entry.id ?? `m${i}`, config.retrieval.cslsK)
}

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

async function meanNdcg() {
  let sum = 0
  for (const q of queries) {
    const result = await retriever.retrieve(q.query, { topK: ndcgAt })
    sum += ndcgAtK(result.top.map((c) => c.id), graded(q), ndcgAt)
  }
  return sum / queries.length
}

const before = await meanNdcg()

// ---- one sleep cycle ----
const cycle = new SleepCycle({
  llm: new EchoLlm(), embedding, store, config,
  log: (msg) => console.error(`[sleep] ${msg}`),
})
const outcome = await cycle.run()

// relevance remapping: a compressed member's relevance transfers to its node
const memberToNode = new Map()
for (const record of store.allMemories()) {
  if (record.flags.compressed_into !== undefined) memberToNode.set(record.id, record.flags.compressed_into)
}
function remap(rel) {
  const merged = {}
  for (const [id, grade] of Object.entries(rel)) {
    const target = memberToNode.get(id) ?? id
    merged[target] = Math.max(merged[target] ?? 0, grade)
  }
  return merged
}

let after = 0
for (const q of queries) {
  const result = await retriever.retrieve(q.query, { topK: ndcgAt })
  after += ndcgAtK(result.top.map((c) => c.id), remap(graded(q)), ndcgAt)
}
after /= queries.length

// ---- report ----
const migrated = outcome.clusters.filter((c) => c.route === 'new_node' || c.route === 'attached')
const gated = outcome.clusters.filter((c) => c.route === 'qa_failed' || c.route === 'epi_failed')
const qaRetained = migrated.length === 0 ? 0
  : migrated.reduce((sum, c) => sum + c.qaF1, 0) / migrated.length
const epiDrop = migrated.length === 0 ? 0
  : migrated.reduce((sum, c) => sum + (c.memberMeanEpi - c.gistEpi), 0) / migrated.length

console.log(`corpus=${corpus.length} queries=${queries.length} embedding=${config.embedding.mode} ` +
  `clusterCos=${config.sleep.clusterCos} clusterMinSize=${config.sleep.clusterMinSize}`)
console.log(`clusters=${outcome.clusters.length} migrated=${migrated.length} gated=${gated.length} ` +
  `compressed=${memberToNode.size} nodes=${store.allNodes().length}`)
for (const c of outcome.clusters) {
  console.log(`  ${c.route.padEnd(10)} members=${c.members.length} qaF1=${c.qaF1.toFixed(3)} ` +
    `gistEpi=${c.gistEpi.toFixed(3)} memberEpi=${c.memberMeanEpi.toFixed(3)} node=${c.nodeId ?? '-'} :: ${c.gist.slice(0, 60)}`)
}
console.log(`QA-F1 of migrated clusters (retention, echo stand-in): ${qaRetained.toFixed(4)}`)
console.log(`mean epi drop of migrated clusters (detail forgetting):  ${epiDrop.toFixed(4)}`)
console.log(`nDCG@${ndcgAt} before=${before.toFixed(4)} after(remapped)=${after.toFixed(4)}`)

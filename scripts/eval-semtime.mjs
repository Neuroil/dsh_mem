/**
 * Semantic vs time ablation scaffold (design §7.4 — the KEY ablation testing this
 * project's core hypothesis: forgetting driven by semantics beats Generative-Agents
 * -style time decay).
 *
 *   corpus:  JSONL, one {"id","text","imp"?,"time"?,"validAt"?,"invalidAt"?} per line
 *            (time = ISO 8601 or epoch ms; missing times are spread linearly over
 *            the last 30 days by line order so the time baseline stays executable)
 *   queries: JSONL, one {"query","relevant":[id...],"at"?} per line; "at" makes the
 *            query time-anchored (epoch ms / ISO) — the semantic arm then searches
 *            the memory state reconstructed at that point (pillar-5 time travel)
 *
 * Arms:
 *   semantic   — this plugin's full retrieval chain (validity-aware pool; time-
 *                anchored queries use retrievableAt(at))
 *   time       — Generative-Agents recency alone: score = 0.995^hours_old, i.e. the
 *                pure time-decay ranking (no semantics, no supersession model)
 *   ga-product — the full Generative-Agents scoring product:
 *                imp × 0.995^hours_old × cos(query, memory)
 *
 * The two baselines rank the RAW corpus (Generative Agents has no supersession or
 * validity model — that gap is part of what the ablation measures). Embeddings
 * default to the deterministic offline provider: numbers only prove the arms run;
 * point --embedding-mode http at a real endpoint for meaningful semantics.
 *
 * Usage:
 *   npm run build
 *   npm run eval:semtime -- --corpus data/corpus.jsonl --queries data/queries.jsonl
 */
import { readFileSync } from 'node:fs'
import { createEmbeddingProvider, createRerankerProvider } from '../lib/embedding.js'
import { epiLexical } from '../lib/judges.js'
import { cosine } from '../lib/metrics.js'
import { MemoryRetriever } from '../lib/retrieval.js'
import { createVectorIndex } from '../lib/vector-store.js'
import { MemoryStore, tierOf } from '../lib/memory-store.js'

// ---- minimal argv ----
const args = Object.fromEntries(process.argv.slice(2).map((arg, i, all) =>
  arg.startsWith('--') ? [arg.slice(2), all[i + 1]?.startsWith('--') ? true : all[i + 1]] : []))

const corpusPath = args.corpus ?? 'data/corpus.jsonl'
const queriesPath = args.queries ?? 'data/queries.jsonl'
const ndcgAt = Number(args['ndcg-at'] ?? 10)

/** Generative Agents recency factor（arXiv:2304.03442：recency = 0.995^h）。 */
const RECENCY_HOURLY = 0.995
const MS_PER_HOUR = 3600 * 1000
const DAY = 24 * MS_PER_HOUR

// ---- config: mirrors src/config.ts defaults ----
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
    pprSeedK: 8, pprDamping: 0.5, pprWeight: 0.2,
  },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: {
    enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3,
    qaF1Threshold: 0.8, epiThreshold: 0.99, conWindow: 1000, conAttachSigma: 0.5,
    conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5,
  },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
  emotion: { workspaces: [], alpha: 0.15, tauHours: 4, disclose: false },
  graph: { synonymCos: 0.8, maxDegree: 8 },
  inspire: { counterProbability: 0.25 },
}

// ---- load dataset ----
function readJsonl(path) {
  return readFileSync(path, 'utf8').split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line))
}

/** ISO 8601 字符串或 epoch ms → epoch ms；缺省 undefined。 */
function parseTime(raw) {
  if (raw === undefined || raw === null) return undefined
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  if (typeof raw === 'string') {
    if (/^\d+$/.test(raw)) return Number(raw)
    const parsed = Date.parse(raw)
    if (!Number.isNaN(parsed)) return parsed
  }
  throw new Error(`unparseable time: ${JSON.stringify(raw)}`)
}

const corpus = readJsonl(corpusPath)
const queries = readJsonl(queriesPath)

// 缺省时间：按行序在过去 30 天线性铺开（仅让时间 baseline 可执行，无语义意义）
const syntheticBase = Date.now() - 30 * DAY
const corpusTimes = corpus.map((entry, i) => parseTime(entry.time) ?? syntheticBase + (i * DAY) / Math.max(1, corpus.length))
/** 数据集的"现在"：最近一条的时间（时间 baseline 的参照时点）。 */
const evalNow = Math.max(...corpusTimes)

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

// Embed corpus + queries once; build records with dataset-driven time_meta.
const vectors = await embedding.embed(corpus.map((entry) => entry.text))
const store = await MemoryStore.open(fakeUnit(), createVectorIndex(), config)
let seq = 0
for (const [i, entry] of corpus.entries()) {
  const id = entry.id ?? `m${i}`
  const time = corpusTimes[i]
  const imp = Number(entry.imp ?? 5)
  await store.putRecord({
    id,
    text: entry.text,
    embedding: vectors[i],
    imp,
    valence: 0,
    arousal: 0,
    epi: epiLexical(entry.text),
    red: 0,
    r_m: 0,
    tier: tierOf(imp, config.tier),
    con_route: { kind: 'pending' },
    links: [],
    S: imp,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: {
      created_at: time,
      valid_at: parseTime(entry.validAt) ?? time,
      ...(parseTime(entry.invalidAt) !== undefined ? { invalid_at: parseTime(entry.invalidAt) } : {}),
    },
    flags: {},
    origin: { session_id: 'eval', turn: 0, seq: seq++ },
  })
}
for (const [i, entry] of corpus.entries()) await store.refreshNeighborhood(entry.id ?? `m${i}`, config.retrieval.cslsK)

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

/** 时间 baseline 排名对象：全语料按分排序（无有效性模型——差异本身是被测对象）。 */
function baselineRanking(queryVector, q, arm) {
  const at = parseTime(q.at)
  const now = at !== undefined ? at : evalNow
  return corpus
    .map((entry, i) => {
      const id = entry.id ?? `m${i}`
      const hours = (now - corpusTimes[i]) / MS_PER_HOUR
      const recency = RECENCY_HOURLY ** hours
      const imp = Number(entry.imp ?? 5)
      const score = arm === 'time' ? recency : imp * recency * (queryVector ? cosine(queryVector, vectors[i]) : 0)
      return { id, score }
    })
    .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1))
    .map((hit) => hit.id)
}

// ---- run the arms ----
const ARMS = ['semantic', 'time', 'ga-product']
const sums = Object.fromEntries(ARMS.map((arm) => [arm, 0]))
let anchoredQueries = 0
let resurfaced = 0

for (const q of queries) {
  const rel = graded(q)
  const at = parseTime(q.at)
  if (at !== undefined) anchoredQueries++

  // semantic arm: full chain, time-travel pool when anchored
  const result = await retriever.retrieve(q.query, { topK: ndcgAt, ...(at !== undefined ? { at } : {}) })
  sums.semantic += ndcgAtK(result.top.map((c) => c.id), rel, ndcgAt)

  // time-anchored bookkeeping: relevant ids visible at `at` but not in the current pool
  if (at !== undefined) {
    const currentIds = new Set(store.retrievable().map((item) => item.id))
    const atIds = new Set(store.retrievableAt(at).map((item) => item.id))
    for (const id of Object.keys(rel)) {
      if (atIds.has(id) && !currentIds.has(id)) resurfaced++
    }
  }

  // baselines rank the raw corpus (no validity model)
  const embedded = await embedding.embed([q.query])
  const queryVector = embedded[0]
  sums.time += ndcgAtK(baselineRanking(queryVector, q, 'time').slice(0, ndcgAt), rel, ndcgAt)
  sums['ga-product'] += ndcgAtK(baselineRanking(queryVector, q, 'ga-product').slice(0, ndcgAt), rel, ndcgAt)
}

// ---- report ----
console.log(`corpus=${corpus.length} queries=${queries.length} embedding=${config.embedding.mode} ` +
  `evalNow=${new Date(evalNow).toISOString()} recency=0.995^h`)
for (const arm of ARMS) {
  console.log(`${arm.padEnd(10)}\tnDCG@${ndcgAt}=${(sums[arm] / queries.length).toFixed(4)}`)
}
console.log(`time-anchored queries: ${anchoredQueries}` +
  (anchoredQueries > 0 ? ` (resurfaced relevant ids via time travel: ${resurfaced})` : ''))

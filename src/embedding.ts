/**
 * 嵌入与重排客户端（设计 §3.1）：dsh 无对应 seam，二者是插件自有依赖，
 * 端点/模型/维度经 Config 配置。http 模式走 OpenAI 兼容 /embeddings 与
 * Cohere/Jina 式 /rerank 协议；deterministic 模式是离线测试替身。
 */
import type { EmbeddingConfig, HttpModelConfig } from './config.js'
import { fnv1a } from './scope.js'
import type { EmbeddingProvider, RerankerProvider } from './types.js'

/** 本地归一化（服务端向量不保证单位长度，统一在此归一）。 */
function normalize(v: readonly number[]): number[] {
  let sum = 0
  for (const x of v) sum += x * x
  const norm = Math.sqrt(sum)
  if (norm === 0 || !Number.isFinite(norm)) return [...v]
  return v.map((x) => x / norm)
}

/** 合并调用方 signal 与超时。 */
function deadlineSignal(timeoutMs: number, signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs)
  return signal ? AbortSignal.any([signal, timeout]) : timeout
}

async function postJson(
  endpoint: string,
  apiKey: string,
  body: unknown,
  signal: AbortSignal,
): Promise<unknown> {
  const headers: Record<string, string> = { 'content-type': 'application/json' }
  if (apiKey) headers.authorization = `Bearer ${apiKey}`
  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`${endpoint} responded ${response.status}: ${text.slice(0, 300)}`)
  }
  return response.json()
}

/**
 * 确定性词袋嵌入（离线替身）：token 落两个桶（hash 与 hash>>>16），符号一正一负，
 * L2 归一化。同文本同向量；仅供测试/评估管线连通，语义质量无意义。
 */
function deterministicEmbed(text: string, dimensions: number): number[] {
  const vector = new Array<number>(dimensions).fill(0)
  const tokens = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []
  for (const token of tokens) {
    const hash = fnv1a(token)
    const first = hash % dimensions
    const second = (hash >>> 16) % dimensions
    vector[first] = (vector[first] ?? 0) + 1
    vector[second] = (vector[second] ?? 0) - 1
  }
  return normalize(vector)
}

export function createEmbeddingProvider(config: EmbeddingConfig): EmbeddingProvider {
  if (config.mode === 'deterministic') {
    return {
      dimensions: config.dimensions,
      async embed(texts) {
        return texts.map((text) => deterministicEmbed(text, config.dimensions))
      },
    }
  }
  if (!config.endpoint) {
    throw new Error('embedding.mode=http 需要 embedding.endpoint 配置')
  }
  return {
    dimensions: config.dimensions,
    async embed(texts, signal) {
      if (texts.length === 0) return []
      const results: number[][] = []
      for (let start = 0; start < texts.length; start += config.batchSize) {
        const batch = texts.slice(start, start + config.batchSize)
        const payload = await postJson(
          config.endpoint,
          config.apiKey,
          { model: config.model, input: batch },
          deadlineSignal(config.timeoutMs, signal),
        ) as { data?: Array<{ index?: number; embedding?: number[] }> }
        const rows = payload.data
        if (!Array.isArray(rows) || rows.length !== batch.length) {
          throw new Error(`${config.endpoint} 返回 ${rows?.length ?? 0} 条嵌入，预期 ${batch.length}`)
        }
        // 按 index 对齐回输入顺序，缺失 index 按数组序兜底
        const ordered: Array<number[] | undefined> = new Array(batch.length)
        rows.forEach((row, fallback) => {
          const embedding = row.embedding
          if (!Array.isArray(embedding)) throw new Error(`${config.endpoint} 返回缺失 embedding 字段`)
          ordered[row.index ?? fallback] = normalize(embedding)
        })
        for (const [i, value] of ordered.entries()) {
          if (!value) throw new Error(`${config.endpoint} 返回嵌入缺口（第 ${i} 条）`)
          results.push(value)
        }
      }
      return results
    },
  }
}

interface RerankResponse {
  results?: Array<{ index?: number; relevance_score?: number }>
}

export function createRerankerProvider(
  config: HttpModelConfig,
  embedding: EmbeddingProvider,
): RerankerProvider {
  if (config.mode === 'deterministic') {
    return {
      async rerank(query, documents) {
        if (documents.length === 0) return []
        const vectors = await embedding.embed([query, ...documents])
        const [queryVector] = vectors
        if (!queryVector) return []
        return vectors.slice(1).map((docVector) => {
          let dot = 0
          for (let i = 0; i < queryVector.length; i++) dot += queryVector[i]! * (docVector[i] ?? 0)
          // 余弦经 (cos+1)/2 映射到 [0,1]（嵌入已归一，dot 即余弦）
          return Math.min(1, Math.max(0, (dot + 1) / 2))
        })
      },
    }
  }
  if (!config.endpoint) {
    throw new Error('reranker.mode=http 需要 reranker.endpoint 配置')
  }
  return {
    async rerank(query, documents, signal) {
      if (documents.length === 0) return []
      const payload = await postJson(
        config.endpoint,
        config.apiKey,
        { model: config.model, query, documents, top_n: documents.length },
        deadlineSignal(config.timeoutMs, signal),
      ) as RerankResponse
      const rows = payload.results
      if (!Array.isArray(rows)) throw new Error(`${config.endpoint} 返回缺失 results 字段`)
      const scores = new Array<number>(documents.length).fill(0)
      rows.forEach((row, fallback) => {
        const score = row.relevance_score
        if (typeof score !== 'number' || !Number.isFinite(score)) return
        const index = row.index ?? fallback
        if (index < 0 || index >= documents.length) return
        scores[index] = Math.min(1, Math.max(0, score))
      })
      return scores
    },
  }
}

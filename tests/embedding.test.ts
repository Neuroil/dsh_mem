import { afterEach, describe, expect, it, vi } from 'vitest'
import { createEmbeddingProvider, createRerankerProvider } from '../src/embedding.js'
import type { EmbeddingConfig, HttpModelConfig } from '../src/config.js'

const embedConfig = (over: Partial<EmbeddingConfig> = {}): EmbeddingConfig => ({
  mode: 'deterministic',
  endpoint: '',
  model: '',
  apiKey: '',
  timeoutMs: 5_000,
  dimensions: 64,
  batchSize: 2,
  ...over,
})

const rerankConfig = (over: Partial<HttpModelConfig> = {}): HttpModelConfig => ({
  mode: 'deterministic',
  endpoint: '',
  model: '',
  apiKey: '',
  timeoutMs: 5_000,
  ...over,
})

describe('deterministic 嵌入', () => {
  it('同文本同向量（确定性）', async () => {
    const provider = createEmbeddingProvider(embedConfig())
    const [a] = await provider.embed(['部署延迟与配置复杂性相关'])
    const [b] = await provider.embed(['部署延迟与配置复杂性相关'])
    expect(a).toEqual(b)
  })

  it('不同文本不同向量；输出 L2 归一', async () => {
    const provider = createEmbeddingProvider(embedConfig())
    const [a, b] = await provider.embed(['apple pie', 'kubernetes rollback'])
    expect(a).not.toEqual(b)
    for (const v of [a!, b!]) {
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
      expect(norm).toBeCloseTo(1, 10)
    }
  })

  it('空输入不发请求', async () => {
    const provider = createEmbeddingProvider(embedConfig())
    expect(await provider.embed([])).toEqual([])
  })
})

describe('deterministic 重排', () => {
  it('值域 [0,1]；同文本=1；空文档返回空', async () => {
    const embedding = createEmbeddingProvider(embedConfig())
    const reranker = createRerankerProvider(rerankConfig(), embedding)
    const scores = await reranker.rerank('apple pie', ['apple pie', 'kubernetes rollback'])
    expect(scores[0]).toBeCloseTo(1, 10)
    expect(scores[1]).toBeCloseTo(0.5, 6) // 正交 → (0+1)/2
    expect(await reranker.rerank('q', [])).toEqual([])
  })
})

describe('http 嵌入', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('请求体形状、鉴权头、按 index 对齐、归一化', async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe('https://emb.test/v1/embeddings')
      expect(init.method).toBe('POST')
      const body = JSON.parse(String(init.body))
      expect(body).toEqual({ model: 'bge-m3', input: ['a', 'b', 'c', 'd'] })
      expect((init.headers as Record<string, string>).authorization).toBe('Bearer sk-test')
      return new Response(JSON.stringify({
        data: [
          { index: 1, embedding: [0, 3, 4] },
          { index: 0, embedding: [3, 0, 4] },
          { index: 3, embedding: [0, 0, 5] },
          { index: 2, embedding: [5, 0, 0] },
        ],
      }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)

    const provider = createEmbeddingProvider(embedConfig({
      mode: 'http',
      endpoint: 'https://emb.test/v1/embeddings',
      model: 'bge-m3',
      apiKey: 'sk-test',
      dimensions: 3,
      batchSize: 4,
    }))
    const vectors = await provider.embed(['a', 'b', 'c', 'd'])
    // index 对齐 + 归一化
    expect(vectors[0]).toEqual([0.6, 0, 0.8])
    expect(vectors[1]).toEqual([0, 0.6, 0.8])
    expect(vectors[2]).toEqual([1, 0, 0])
    expect(vectors[3]).toEqual([0, 0, 1])
  })

  it('batchSize 分批多次请求', async () => {
    const calls: string[][] = []
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      calls.push(JSON.parse(String(init.body)).input)
      return new Response(JSON.stringify({
        data: JSON.parse(String(init.body)).input.map((_: string, i: number) => ({ index: i, embedding: [1, 0] })),
      }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2, batchSize: 2 }))
    await provider.embed(['a', 'b', 'c', 'd', 'e'])
    expect(calls).toEqual([['a', 'b'], ['c', 'd'], ['e']])
  })

  it('非 2xx 抛错；返回条数不符抛错', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 503 })))
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2 }))
    await expect(provider.embed(['x'])).rejects.toThrow(/503/)

    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ data: [] }), { status: 200 })))
    await expect(provider.embed(['x'])).rejects.toThrow(/预期 1/)
  })

  it('fetch 网络异常（非 HTTP 状态错误）原样上抛', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('fetch failed') }))
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2 }))
    await expect(provider.embed(['x'])).rejects.toThrow(/fetch failed/)
  })

  it('调用方 signal 中止向上传播（AbortError 上抛）', async () => {
    const controller = new AbortController()
    vi.stubGlobal('fetch', vi.fn((_url: string, init: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        if (init.signal?.aborted) {
          reject(new DOMException('aborted', 'AbortError'))
          return
        }
        init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
      })))
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2 }))
    const pending = provider.embed(['x'], controller.signal)
    controller.abort()
    await expect(pending).rejects.toThrow(/abort/i)
  })

  it('index 对齐缺口（重复 index 留洞）抛错', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({
      data: [
        { index: 0, embedding: [1, 0] },
        { index: 0, embedding: [0, 1] },
      ],
    }), { status: 200 })))
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2, batchSize: 2 }))
    await expect(provider.embed(['a', 'b'])).rejects.toThrow(/嵌入缺口/)
  })

  it('响应缺 data 字段抛错', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })))
    const provider = createEmbeddingProvider(embedConfig({ mode: 'http', endpoint: 'https://emb.test', dimensions: 2 }))
    await expect(provider.embed(['x'])).rejects.toThrow(/预期 1/)
  })
})

describe('http 重排', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('请求体形状、按 index 回填并 clamp 到 [0,1]', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body))
      expect(body.query).toBe('q')
      expect(body.documents).toEqual(['d0', 'd1', 'd2'])
      expect(body.top_n).toBe(3)
      return new Response(JSON.stringify({
        results: [
          { index: 2, relevance_score: 1.7 },
          { index: 0, relevance_score: 0.42 },
          { index: 1, relevance_score: -0.3 },
        ],
      }), { status: 200 })
    })
    vi.stubGlobal('fetch', fetchMock)
    const embedding = createEmbeddingProvider(embedConfig())
    const reranker = createRerankerProvider(rerankConfig({ mode: 'http', endpoint: 'https://rr.test/v1/rerank' }), embedding)
    expect(await reranker.rerank('q', ['d0', 'd1', 'd2'])).toEqual([0.42, 0, 1])
  })

  it('响应缺 results 字段抛错；fetch 网络异常上抛', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })))
    const embedding = createEmbeddingProvider(embedConfig())
    const reranker = createRerankerProvider(rerankConfig({ mode: 'http', endpoint: 'https://rr.test/v1/rerank' }), embedding)
    await expect(reranker.rerank('q', ['d'])).rejects.toThrow(/results/)

    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('fetch failed') }))
    await expect(reranker.rerank('q', ['d'])).rejects.toThrow(/fetch failed/)
  })
})

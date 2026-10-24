/**
 * 检索算法模块单测：metrics 纯函数、VectorIndex、buildExpandedQuery、
 * 四级检索管线（含 CSLS 压 hub、cold 门控、四档消融、空库与降级路径）。
 * stub 全部内联自造：embedding 用手工向量映射，reranker 用注入的固定分数函数。
 */
import { describe, expect, it, vi } from 'vitest'
import type { RetrievalConfig } from '../src/config.js'
import { cosine, cslsCue, mean, mmrSelect, normalize } from '../src/metrics.js'
import { buildExpandedQuery, MemoryRetriever } from '../src/retrieval.js'
import type { MemoryLookup } from '../src/retrieval.js'
import type {
  EmbeddingProvider,
  MemoryTier,
  RerankerProvider,
  RetrievableItem,
  RetrievalAblation,
  ScoredCandidate,
} from '../src/types.js'
import { createVectorIndex } from '../src/vector-store.js'

const ids = (candidates: readonly ScoredCandidate[]): string[] => candidates.map(c => c.id)

// ---------- stub 工具 ----------

function makeConfig(overrides: Partial<RetrievalConfig> = {}): RetrievalConfig {
  return {
    l0TopK: 10,
    cslsTopK: 5,
    rerankTopK: 3,
    mmrTopK: 2,
    mmrLambda: 0.6,
    cslsK: 10,
    contextTurns: 3,
    coldCueBoost: 0.1,
    cueThreshold: 0.3,
    ...overrides,
  }
}

function makeItem(init: {
  id: string
  text: string
  embedding: readonly number[]
  r_m?: number
  tier?: MemoryTier
  createdAt?: number
  layer?: 'episodic' | 'semantic'
}): RetrievableItem {
  return {
    id: init.id,
    text: init.text,
    embedding: init.embedding,
    r_m: init.r_m ?? 0.1,
    tier: init.tier ?? 'warm',
    createdAt: init.createdAt ?? 0,
    layer: init.layer ?? 'episodic',
  }
}

function makeEmbedding(map: Record<string, readonly number[]>) {
  const embed = vi.fn(async (texts: readonly string[]): Promise<readonly number[][]> =>
    texts.map(text => {
      const v = map[text]
      if (v === undefined) throw new Error(`embedding stub 缺少向量: ${JSON.stringify(text)}`)
      return [...v]
    }),
  )
  const provider: EmbeddingProvider = { dimensions: 2, embed }
  return { provider, embed }
}

type RerankImpl = (query: string, documents: readonly string[]) => Promise<readonly number[]>

function scoreReranker(scoresByDoc: Record<string, number>): RerankImpl {
  return async (_query, documents) => documents.map(doc => scoresByDoc[doc] ?? 0)
}

function makeReranker(impl: RerankImpl) {
  const rerank = vi.fn(impl)
  const provider: RerankerProvider = { rerank }
  return { provider, rerank }
}

function makeRetriever(opts: {
  items: readonly RetrievableItem[]
  vectors: Record<string, readonly number[]>
  scores?: Record<string, number>
  config?: Partial<RetrievalConfig>
  rerankImpl?: RerankImpl
}) {
  const { provider: embedding, embed } = makeEmbedding(opts.vectors)
  const { provider: reranker, rerank } = makeReranker(opts.rerankImpl ?? scoreReranker(opts.scores ?? {}))
  const lookup: MemoryLookup = { retrievable: () => opts.items }
  const retriever = new MemoryRetriever({ embedding, reranker, lookup, config: makeConfig(opts.config) })
  return { retriever, embed, rerank }
}

/** 首分量为 x 的 2 维单位向量：与 q=[1,0] 的余弦恰为 x。 */
function unitByX(x: number): number[] {
  return [x, Math.sqrt(Math.max(0, 1 - x * x))]
}

const QUERY = 'query'
const QVEC: readonly number[] = [1, 0]

/**
 * 全管线夹具（q=[1,0]）：
 *   cos: alpha 0.9 / beta 0.8 / gamma 0.7 / delta 0，r_m 均 0.1
 *   r(q) = 0.6 → cue: 1.1 / 0.9 / 0.7 / -0.7（序不变）
 *   精排注入分数: delta .99 > beta .9 > gamma .7 > alpha .5 → 精排序颠覆余弦序
 */
function fullPipelineFixture() {
  const items: RetrievableItem[] = [
    makeItem({ id: 'alpha', text: 'alpha', embedding: unitByX(0.9) }),
    makeItem({ id: 'beta', text: 'beta', embedding: unitByX(0.8) }),
    makeItem({ id: 'gamma', text: 'gamma', embedding: unitByX(0.7) }),
    makeItem({ id: 'delta', text: 'delta', embedding: [0, 1] }),
  ]
  const vectors: Record<string, readonly number[]> = {
    [QUERY]: QVEC,
    alpha: unitByX(0.9),
    beta: unitByX(0.8),
    gamma: unitByX(0.7),
    delta: [0, 1],
  }
  const scores: Record<string, number> = { alpha: 0.5, beta: 0.9, gamma: 0.7, delta: 0.99 }
  return { items, vectors, scores }
}

// ---------- metrics ----------

describe('metrics', () => {
  it('cosine：正交为 0、同向为 1、零向量返回 0', () => {
    expect(cosine([1, 0], [0, 1])).toBe(0)
    expect(cosine([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 12)
    expect(cosine([1, 0], [1, 1])).toBeCloseTo(Math.SQRT1_2, 12)
    expect(cosine([1, -1], [1, 1])).toBeCloseTo(0, 12)
    expect(cosine([0, 0], [1, 2])).toBe(0)
    expect(cosine([1, 0], [0, 0])).toBe(0)
  })

  it('normalize：L2 归一化；零向量原样返回', () => {
    const normalized = normalize([3, 4])
    expect(normalized[0]).toBeCloseTo(0.6, 12)
    expect(normalized[1]).toBeCloseTo(0.8, 12)
    expect(normalize([0, 0])).toEqual([0, 0])
    expect(normalize([0, 1])).toEqual([0, 1])
    const unit = normalize([1.5, -2.5, 3.5])
    const norm = Math.sqrt(unit.reduce((s, x) => s + x * x, 0))
    expect(norm).toBeCloseTo(1, 12)
  })

  it('cslsCue：2·cos − r(q) − r(m)', () => {
    expect(cslsCue(0.8, 0.3, 0.2)).toBeCloseTo(1.1, 12)
  })

  it('mean：算术均值；空数组返回 0', () => {
    expect(mean([1, 2, 3])).toBeCloseTo(2, 12)
    expect(mean([])).toBe(0)
  })

  it('mmrSelect：高相似冗余项让位于低相关的独特项，按选择顺序返回', () => {
    type Item = { id: string; rel: number }
    const items: Item[] = [
      { id: 'A', rel: 0.9 },
      { id: 'B', rel: 0.85 },
      { id: 'C', rel: 0.5 },
    ]
    // B 与 A 近乎重复（sim 0.99），C 独立（sim 0）
    const simTable: Record<string, number> = { 'A|B': 0.99, 'B|A': 0.99 }
    const sim = (a: Item, b: Item): number => simTable[`${a.id}|${b.id}`] ?? 0
    const relevance = (x: Item): number => x.rel
    // λ=0.6：先选 A；B 得 0.6·0.85−0.4·0.99=0.114，输给 C 的 0.6·0.5=0.3
    expect(mmrSelect(items, 2, 0.6, relevance, sim).map(x => x.id)).toEqual(['A', 'C'])
    // k 放开到全量：B 最后入选，选择顺序保持
    expect(mmrSelect(items, 3, 0.6, relevance, sim).map(x => x.id)).toEqual(['A', 'C', 'B'])
    // λ=1 退化为纯相关序
    expect(mmrSelect(items, 3, 1, relevance, sim).map(x => x.id)).toEqual(['A', 'B', 'C'])
    // 边界：k<=0 或空集
    expect(mmrSelect(items, 0, 0.6, relevance, sim)).toEqual([])
    expect(mmrSelect(items, 1, 0.6, relevance, sim).map(x => x.id)).toEqual(['A'])
    expect(mmrSelect([] as Item[], 3, 0.6, relevance, sim)).toEqual([])
  })
})

// ---------- VectorIndex ----------

describe('createVectorIndex', () => {
  it('增删查与 size', () => {
    const index = createVectorIndex()
    expect(index.size()).toBe(0)
    index.add('a', [1, 0])
    index.add('b', [0, 1])
    index.add('c', [1, 1])
    expect(index.size()).toBe(3)
    expect(index.remove('b')).toBe(true)
    expect(index.remove('b')).toBe(false)
    expect(index.size()).toBe(2)
    const hits = index.search([1, 0], 10)
    expect(hits.map(h => h.id)).toEqual(['a', 'c'])
    expect(hits[0]!.cos).toBeCloseTo(1, 12)
    expect(hits[1]!.cos).toBeCloseTo(Math.SQRT1_2, 12)
  })

  it('k 截断与 k<=0 返回空', () => {
    const index = createVectorIndex()
    index.add('a', [1, 0])
    index.add('b', [0.9, Math.sqrt(1 - 0.81)])
    index.add('c', [0, 1])
    expect(index.search([1, 0], 2).map(h => h.id)).toEqual(['a', 'b'])
    expect(index.search([1, 0], 0)).toEqual([])
    expect(index.search([1, 0], -1)).toEqual([])
  })

  it('filter 在排序前过滤', () => {
    const index = createVectorIndex()
    index.add('a', [1, 0])
    index.add('b', [0.95, Math.sqrt(1 - 0.9025)])
    index.add('c', [0.9, Math.sqrt(1 - 0.81)])
    const hits = index.search([1, 0], 10, id => id !== 'b')
    expect(hits.map(h => h.id)).toEqual(['a', 'c'])
  })

  it('余弦平局按 id 字典序（确定性）', () => {
    const index = createVectorIndex()
    index.add('y', [0.8, 0.6])
    index.add('x', [0.8, 0.6])
    index.add('m', [0.8, 0.6])
    expect(index.search([0.8, 0.6], 3).map(h => h.id)).toEqual(['m', 'x', 'y'])
  })

  it('写入时归一化：不同尺度的同向向量余弦为 1', () => {
    const index = createVectorIndex()
    index.add('scaled', [3, 4])
    const hits = index.search([0.6, 0.8], 1)
    expect(hits[0]!.id).toBe('scaled')
    expect(hits[0]!.cos).toBeCloseTo(1, 12)
  })
})

// ---------- buildExpandedQuery ----------

describe('buildExpandedQuery', () => {
  it('无近期轮次、全空轮次或 contextTurns<=0 时仅返回 query', () => {
    expect(buildExpandedQuery('q', [], 3)).toBe('q')
    expect(buildExpandedQuery('q', ['', '', ''], 3)).toBe('q')
    expect(buildExpandedQuery('q', ['t1', 't2'], 0)).toBe('q')
  })

  it('非空轮次以 \'\\n---\\n\' 拼接在 query 之后', () => {
    expect(buildExpandedQuery('q', ['a', 'b'], 3)).toBe('q\n---\na\n---\nb')
    expect(buildExpandedQuery('q', ['t1', ''], 3)).toBe('q\n---\nt1')
  })

  it('只取末尾 contextTurns 条', () => {
    expect(buildExpandedQuery('q', ['t1', 't2', 't3', 't4'], 2)).toBe('q\n---\nt3\n---\nt4')
  })
})

// ---------- MemoryRetriever ----------

describe('MemoryRetriever', () => {
  it('full：四级管线各级排序与最终 top/suppressed', async () => {
    const fx = fullPipelineFixture()
    const { retriever, embed, rerank } = makeRetriever({ items: fx.items, vectors: fx.vectors, scores: fx.scores })
    const result = await retriever.retrieve(QUERY)
    expect(result.expandedQuery).toBe(QUERY)
    // L0：余弦序 0.9 > 0.8 > 0.7 > 0
    expect(ids(result.stages.l0)).toEqual(['alpha', 'beta', 'gamma', 'delta'])
    expect(result.stages.l0[0]!.cos).toBeCloseTo(0.9, 8)
    // CSLS：r(q)=0.6，cue = 2cos − 0.6 − 0.1，序不变
    expect(ids(result.stages.csls)).toEqual(['alpha', 'beta', 'gamma', 'delta'])
    expect(result.stages.csls[0]!.cue).toBeCloseTo(1.1, 8)
    // 精排：注入分数颠覆余弦序，截断 rerankTopK=3
    expect(ids(result.stages.rerank)).toEqual(['delta', 'beta', 'gamma'])
    // MMR：先 delta；其次 beta（与 delta 相似度 0.6，胜过 gamma 的 ~0.714）
    expect(ids(result.stages.mmr)).toEqual(['delta', 'beta'])
    expect(ids(result.top)).toEqual(['delta', 'beta'])
    // 逐级字段贯穿：top[0] 同时保留 cos / cue / relevance
    expect(result.top[0]!.cos).toBeCloseTo(0, 8)
    expect(result.top[0]!.cue).toBeCloseTo(-0.7, 8)
    expect(result.top[0]!.relevance).toBeCloseTo(0.99, 8)
    // suppressed：被打过分但未进 top
    expect(new Set(ids(result.suppressed))).toEqual(new Set(['alpha', 'gamma']))
    // 调用契约：q' 只嵌入一次；精排以 q' 与 CSLS 序文本调用
    expect(embed).toHaveBeenCalledTimes(1)
    expect(embed.mock.calls[0]![0]).toEqual([QUERY])
    expect(rerank).toHaveBeenCalledTimes(1)
    expect(rerank.mock.calls[0]![0]).toBe(QUERY)
    expect(rerank.mock.calls[0]![1]).toEqual(['alpha', 'beta', 'gamma', 'delta'])
  })

  it('CSLS 压 hub：泛化记忆余弦更高，但独特匹配反超（§3.2 关键创新点）', async () => {
    const items: RetrievableItem[] = [
      // "和谁都沾边"：与查询余弦中等（0.5）但 r_m 大
      makeItem({ id: 'hub', text: 'hub', embedding: unitByX(0.5), r_m: 0.4 }),
      // 独特匹配：余弦略低（0.45）但 r_m 小
      makeItem({ id: 'unique', text: 'unique', embedding: unitByX(0.45), r_m: 0.05 }),
    ]
    const vectors: Record<string, readonly number[]> = {
      [QUERY]: QVEC,
      hub: unitByX(0.5),
      unique: unitByX(0.45),
    }
    const { retriever, rerank } = makeRetriever({ items, vectors })
    const result = await retriever.retrieve(QUERY, { ablation: 'csls' })
    // L0 余弦序：hub 在前
    expect(ids(result.stages.l0)).toEqual(['hub', 'unique'])
    // r(q)=0.475 → cue: hub = 0.125 < unique = 0.375，CSLS 后反超
    expect(ids(result.stages.csls)).toEqual(['unique', 'hub'])
    expect(result.stages.csls[0]!.cue).toBeCloseTo(0.375, 8)
    expect(result.stages.csls[1]!.cue).toBeCloseTo(0.125, 8)
    expect(ids(result.top)).toEqual(['unique', 'hub'])
    expect(rerank).not.toHaveBeenCalled()
  })

  it('cold 门控：cue 低于 阈值+boost 的 cold 记忆被剔除，warm 不受限', async () => {
    const items: RetrievableItem[] = [
      makeItem({ id: 'coldLow', text: 'coldLow', embedding: unitByX(0.8), r_m: 0.9, tier: 'cold' }),
      makeItem({ id: 'coldHigh', text: 'coldHigh', embedding: unitByX(0.6), r_m: 0.05, tier: 'cold' }),
      makeItem({ id: 'warmLow', text: 'warmLow', embedding: unitByX(0.5), r_m: 0.1, tier: 'warm' }),
    ]
    const vectors: Record<string, readonly number[]> = {
      [QUERY]: QVEC,
      coldLow: unitByX(0.8),
      coldHigh: unitByX(0.6),
      warmLow: unitByX(0.5),
    }
    const { retriever, rerank } = makeRetriever({ items, vectors })
    // 门槛 0.3+0.1=0.4：r(q)=1.9/3≈0.633 → coldLow cue≈0.067 剔除、coldHigh cue≈0.517 保留、warmLow cue≈0.267 保留
    const result = await retriever.retrieve(QUERY, { ablation: 'csls' })
    expect(ids(result.stages.l0)).toEqual(['coldLow', 'coldHigh', 'warmLow'])
    expect(ids(result.stages.csls)).toEqual(['coldHigh', 'warmLow'])
    expect(rerank).not.toHaveBeenCalled()
  })

  it('四档消融各自返回预期排序（§7.1）', async () => {
    const run = async (ablation: RetrievalAblation) => {
      const fx = fullPipelineFixture()
      const { retriever, rerank } = makeRetriever({ items: fx.items, vectors: fx.vectors, scores: fx.scores })
      return { result: await retriever.retrieve(QUERY, { ablation }), rerank }
    }
    // base：L0 余弦序直接取前 topK=2，后续 stage 空占位，不精排
    const base = await run('base')
    expect(ids(base.result.top)).toEqual(['alpha', 'beta'])
    expect(base.result.stages.csls).toEqual([])
    expect(base.result.stages.rerank).toEqual([])
    expect(base.result.stages.mmr).toEqual([])
    expect(base.result.suppressed).toEqual([])
    expect(base.rerank).not.toHaveBeenCalled()
    // csls：CSLS 截断后取前 2（本夹具 CSLS 序与余弦序一致）
    const csls = await run('csls')
    expect(ids(csls.result.top)).toEqual(['alpha', 'beta'])
    expect(csls.result.stages.rerank).toEqual([])
    expect(csls.result.stages.mmr).toEqual([])
    expect(csls.rerank).not.toHaveBeenCalled()
    // rerank：精排截断后取前 2；mmr 空；suppressed = 打分池 − top
    const rk = await run('rerank')
    expect(ids(rk.result.top)).toEqual(['delta', 'beta'])
    expect(rk.result.stages.mmr).toEqual([])
    expect(new Set(ids(rk.result.suppressed))).toEqual(new Set(['alpha', 'gamma']))
    expect(rk.rerank).toHaveBeenCalledTimes(1)
    // full：+MMR
    const full = await run('full')
    expect(ids(full.result.top)).toEqual(['delta', 'beta'])
    expect(full.rerank).toHaveBeenCalledTimes(1)
  })

  it('opts.topK 覆盖最终输出条数', async () => {
    const fx = fullPipelineFixture()
    const { retriever } = makeRetriever({ items: fx.items, vectors: fx.vectors, scores: fx.scores })
    const result = await retriever.retrieve(QUERY, { topK: 1 })
    expect(ids(result.top)).toEqual(['delta'])
    expect(new Set(ids(result.suppressed))).toEqual(new Set(['alpha', 'beta', 'gamma']))
  })

  it('空库：不调用 embedding/reranker，直接返回空结果', async () => {
    const { retriever, embed, rerank } = makeRetriever({ items: [], vectors: { [QUERY]: QVEC } })
    const result = await retriever.retrieve(QUERY)
    expect(result.expandedQuery).toBe(QUERY)
    expect(result.stages.l0).toEqual([])
    expect(result.stages.csls).toEqual([])
    expect(result.stages.rerank).toEqual([])
    expect(result.stages.mmr).toEqual([])
    expect(result.top).toEqual([])
    expect(result.suppressed).toEqual([])
    expect(embed).not.toHaveBeenCalled()
    expect(rerank).not.toHaveBeenCalled()
  })

  it('CSLS 门控后候选清空：不调用 reranker', async () => {
    // 唯一候选是 cue 低于门槛的 cold 记忆
    const items: RetrievableItem[] = [
      makeItem({ id: 'coldOnly', text: 'coldOnly', embedding: unitByX(0.8), r_m: 0.9, tier: 'cold' }),
    ]
    const vectors: Record<string, readonly number[]> = { [QUERY]: QVEC, coldOnly: unitByX(0.8) }
    const { retriever, rerank } = makeRetriever({ items, vectors })
    const result = await retriever.retrieve(QUERY)
    expect(result.stages.l0).toHaveLength(1)
    expect(result.stages.csls).toEqual([])
    expect(result.top).toEqual([])
    expect(rerank).not.toHaveBeenCalled()
  })

  it('reranker 抛错时降级沿用 cue 序，检索不崩溃', async () => {
    const fx = fullPipelineFixture()
    const { retriever, rerank } = makeRetriever({
      items: fx.items,
      vectors: fx.vectors,
      rerankImpl: async () => {
        throw new Error('reranker down')
      },
    })
    const result = await retriever.retrieve(QUERY)
    expect(rerank).toHaveBeenCalledTimes(1)
    // 精排阶段降级为 cue 序：取 CSLS 序前 rerankTopK=3，relevance 缺失
    expect(ids(result.stages.rerank)).toEqual(['alpha', 'beta', 'gamma'])
    expect(result.stages.rerank[0]!.relevance).toBeUndefined()
    // MMR 以 cue 代理 relevance：alpha(1.1) 先选，其次 beta
    expect(ids(result.top)).toEqual(['alpha', 'beta'])
  })

  it('q\' 同时传给 embedding 与 reranker；contextTurns 截断与 signal 透传', async () => {
    const expanded = 'q\n---\nt3'
    const items: RetrievableItem[] = [makeItem({ id: 'm1', text: 'm1', embedding: unitByX(1) })]
    const vectors: Record<string, readonly number[]> = { [expanded]: QVEC, m1: unitByX(1) }
    const { retriever, embed, rerank } = makeRetriever({
      items,
      vectors,
      scores: { m1: 1 },
      config: { contextTurns: 2 },
    })
    const signal = new AbortController().signal
    const result = await retriever.retrieve('q', { recentTurns: ['t1', '', 't3'], signal })
    // 末尾 2 条中空串被丢弃，仅保留 t3
    expect(result.query).toBe('q')
    expect(result.expandedQuery).toBe(expanded)
    expect(embed.mock.calls[0]![0]).toEqual([expanded])
    expect(embed.mock.calls[0]![1]).toBe(signal)
    expect(rerank.mock.calls[0]![0]).toBe(expanded)
    expect(rerank.mock.calls[0]![1]).toEqual(['m1'])
    expect(rerank.mock.calls[0]![2]).toBe(signal)
    expect(ids(result.top)).toEqual(['m1'])
  })
})

/**
 * 四级检索管线（设计 §5.2）：
 * L0 余弦召回 → CSLS 重排（§3.2）→ cross-encoder 精排（§3.1）→ MMR 去冗余（§3.3）。
 * 纯算法模块：模型经 EmbeddingProvider/RerankerProvider 注入，存储经窄接口
 * MemoryLookup 注入，不依赖宿主框架，可独立测试。
 */
import type { RetrievalConfig } from './config.js'
import { moodSim } from './emotion.js'
import { buildAdjacency, personalizedPagerank } from './graph.js'
import { cosine, cslsCue, mean, mmrSelect } from './metrics.js'
import type {
  EmbeddingProvider,
  MoodState,
  RerankerProvider,
  RetrievableItem,
  RetrievalAblation,
  RetrievalResult,
  ScoredCandidate,
} from './types.js'

/** 存储层只读视图（由 MemoryStore 实现；此处只依赖该窄接口）。 */
export interface MemoryLookup {
  /** L0 检索池（§5.2）：有效未压缩情景记录 + 有效语义节点。 */
  retrievable(): readonly RetrievableItem[]
  /**
   * 时点重建的 L0 检索池（§2.5 完整版时间旅行，P2）：valid_at ≤ at <
   * invalid_at 且当时未压缩的记录 + 当时有效的语义节点。MemoryStore 实现；
   * 测试替身可省略（省略时时间旅行查询退化为当前池）。
   */
  retrievableAt?(at: number): readonly RetrievableItem[]
}

/**
 * q' = query ⊕ 最近 contextTurns 轮摘要（§3.1）：
 * 取 recentTurns 末尾 contextTurns 条，非空项与 query 以 '\n---\n' 连接；全空则仅 query。
 */
export function buildExpandedQuery(query: string, recentTurns: readonly string[], contextTurns: number): string {
  // 注意 slice(-0) 会取整个数组，contextTurns <= 0 时必须显式置空
  const tail = contextTurns > 0
    ? recentTurns.slice(-contextTurns).filter(turn => turn !== '')
    : []
  return [query, ...tail].join('\n---\n')
}

export interface RetrieveOptions {
  readonly recentTurns?: readonly string[]
  /** 消融档（设计 §7.1），默认 'full'。 */
  readonly ablation?: RetrievalAblation
  readonly signal?: AbortSignal
  /** 覆盖最终输出条数（默认 config.mmrTopK）。 */
  readonly topK?: number
  /** 时间旅行时点（epoch ms，§2.5 完整版）：检索池换为该时点的记忆状态。 */
  readonly at?: number
  /**
   * 情绪重排上下文（P3-1，仅情绪白名单工作区由 runtime 传入）：L0 召回池
   * 纯语义不变，情绪项只进 CSLS 排序键 cue + α·moodSim；cold 门控用纯 cue。
   * 缺省时全链路与无情绪版本逐字节一致。
   */
  readonly mood?: { readonly state: MoodState; readonly alpha: number }
}

/** id 字典序，平局时的确定性决胜（保证测试可复现）。 */
function compareId(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

export class MemoryRetriever {
  private readonly embedding: EmbeddingProvider
  private readonly reranker: RerankerProvider
  private readonly lookup: MemoryLookup
  private readonly config: RetrievalConfig

  constructor(deps: {
    embedding: EmbeddingProvider
    reranker: RerankerProvider
    lookup: MemoryLookup
    config: RetrievalConfig
  }) {
    this.embedding = deps.embedding
    this.reranker = deps.reranker
    this.lookup = deps.lookup
    this.config = deps.config
  }

  async retrieve(query: string, opts: RetrieveOptions = {}): Promise<RetrievalResult> {
    const cfg = this.config
    const ablation: RetrievalAblation = opts.ablation ?? 'full'
    const topK = opts.topK ?? cfg.mmrTopK
    const expandedQuery = buildExpandedQuery(query, opts.recentTurns ?? [], cfg.contextTurns)
    // 每级幸存者依次填入；未执行到的 stage 保持空数组占位
    const stages: {
      l0: ScoredCandidate[]
      csls: ScoredCandidate[]
      rerank: ScoredCandidate[]
      mmr: ScoredCandidate[]
    } = { l0: [], csls: [], rerank: [], mmr: [] }
    const result = (top: readonly ScoredCandidate[], suppressed: readonly ScoredCandidate[]): RetrievalResult =>
      ({ query, expandedQuery, stages, top, suppressed })

    // ---- L0 召回（§3.1）：q' 嵌入一次，对检索宇宙（记录 + 语义节点；时间旅行
    // 时换为 at 时点重建视图，§2.5）算余弦，取 top l0TopK
    const items = opts.at !== undefined
      ? (this.lookup.retrievableAt?.(opts.at) ?? this.lookup.retrievable())
      : this.lookup.retrievable()
    if (items.length === 0) return result([], []) // 空库：跳过嵌入与精排，直接空结果
    const embedded = await this.embedding.embed([expandedQuery], opts.signal)
    const queryVector = embedded[0]
    if (queryVector === undefined) return result([], [])
    const scored = items.map(item => ({ id: item.id, cos: cosine(queryVector, item.embedding), item }))
    scored.sort((a, b) => b.cos - a.cos || compareId(a.id, b.id))
    stages.l0 = scored.slice(0, cfg.l0TopK)

    // ---- P3-3 确定性联想：L0 种子 → 图上 PPR → 分数作为排序加成项
    // 图无边 / 无配置时精确跳过（pprScore = undefined → 全链路与无 PPR 版本恒等）；
    // PPR 只重排不召回——非 L0 池内项不可被联想拉入（与情绪加成同一保守边界）。
    let pprScore: ReadonlyMap<string, number> | undefined
    if (cfg.pprWeight > 0 && cfg.pprSeedK > 0) {
      const seeds = new Map<string, number>()
      for (const c of stages.l0.slice(0, cfg.pprSeedK)) seeds.set(c.id, Math.max(c.cos, 0))
      const adjacency = buildAdjacency(items)
      const hasEdge = [...adjacency.values()].some((neighbors) => neighbors.size > 0)
      if (hasEdge) {
        const rank = personalizedPagerank(adjacency, seeds, cfg.pprDamping)
        if (rank.size > 0) pprScore = rank
      }
    }

    if (ablation === 'base') {
      return result(stages.l0.slice(0, topK), [])
    }

    // ---- CSLS 重排（§3.2）：r(q) 取 L0 候选池余弦均值；cue = 2·cos − r(q) − r(m)
    // P3-1/P3-3 排序加成（只动排序不动召回）：α·moodSim（情绪白名单）+ pprWeight·PPR
    // （图联想）——作用于语义筛选后的三处排序键（CSLS/精排截断/MMR），效果须传到
    // 最终 top；**cold 门控与 L0 召回用纯 cue/纯余弦**（冷层不得凭情绪/联想契合
    // 混过门槛——侧门封死）。候选项无相应特征或上下文未传时加成恒 0（与旧版本恒等）。
    const moodCtx = opts.mood
    const moodBonus = (c: ScoredCandidate): number =>
      moodCtx !== undefined && c.moodSim !== undefined ? moodCtx.alpha * c.moodSim : 0
    const pprBonus = (c: ScoredCandidate): number =>
      pprScore !== undefined ? (pprScore.get(c.id) ?? 0) * cfg.pprWeight : 0
    const bonus = (c: ScoredCandidate): number => moodBonus(c) + pprBonus(c)
    const rQuery = mean(stages.l0.map(c => c.cos))
    const withCue = stages.l0.map(c => {
      const cue = cslsCue(c.cos, rQuery, c.item.r_m)
      const sim =
        moodCtx !== undefined && c.item.valence !== undefined && c.item.arousal !== undefined
          ? moodSim(moodCtx.state, c.item.valence, c.item.arousal)
          : undefined
      return sim === undefined ? { ...c, cue } : { ...c, cue, moodSim: sim }
    })
    const cslsSortKey = (c: ScoredCandidate): number => (c.cue ?? 0) + bonus(c)
    // cold 层门控（§5.2）：cold 项 cue 需达到 cueThreshold + coldCueBoost 才保留
    const coldFloor = cfg.cueThreshold + cfg.coldCueBoost
    stages.csls = withCue
      .filter(c => c.item.tier !== 'cold' || c.cue === undefined || c.cue >= coldFloor)
      .sort((a, b) => cslsSortKey(b) - cslsSortKey(a) || compareId(a.id, b.id))
      .slice(0, cfg.cslsTopK)

    if (ablation === 'csls') {
      return result(stages.csls.slice(0, topK), [])
    }

    // ---- 精排（§3.1）：cross-encoder 对 CSLS 幸存文本打分，分值只做同查询内排序
    if (stages.csls.length === 0) return result([], []) // 无候选不调 reranker
    let rerankPool: ScoredCandidate[]
    try {
      const documents = stages.csls.map(c => c.item.text)
      const scores = await this.reranker.rerank(expandedQuery, documents, opts.signal)
      rerankPool = stages.csls.map((c, i) => ({ ...c, relevance: clamp01(scores[i] ?? 0) }))
      rerankPool.sort((a, b) =>
        ((b.relevance ?? 0) + bonus(b)) - ((a.relevance ?? 0) + bonus(a)) || compareId(a.id, b.id))
    } catch {
      // 精排失败不致命：降级沿用 cue 序（relevance 缺失，MMR 阶段以 cue 代理）
      rerankPool = [...stages.csls]
    }
    stages.rerank = rerankPool.slice(0, cfg.rerankTopK)

    // suppressed：进入过精排阶段（被 cross-encoder 竞争过）但未进 top 的候选（§2.2 弱形式抑制）
    const finalize = (top: readonly ScoredCandidate[]): RetrievalResult => {
      const topIds = new Set(top.map(c => c.id))
      return result(top, rerankPool.filter(c => !topIds.has(c.id)))
    }

    if (ablation === 'rerank') {
      return finalize(stages.rerank.slice(0, topK))
    }

    // ---- MMR（§3.3）：relevance 用精排分（降级时以 cue 代理），相似度用候选间余弦；
    // 相关性输入加情绪/PPR 加成（P3-1/P3-3，与 CSLS/精排截断同一 bonus，效果传到最终 top）
    const mmrStage = mmrSelect(
      stages.rerank,
      topK,
      cfg.mmrLambda,
      c => (c.relevance ?? c.cue ?? c.cos) + bonus(c),
      (a, b) => cosine(a.item.embedding, b.item.embedding),
    )
    stages.mmr = mmrStage
    return finalize(mmrStage)
  }
}

/** 将提供方分数夹到 [0,1]（接口约定区间，防御性处理）。 */
function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x))
}

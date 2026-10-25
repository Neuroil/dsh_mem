/**
 * Data model for the semantic memory system (design doc §4).
 *
 * Core stance (§1.2): forgetting and consolidation are driven by semantics,
 * never by time. Timestamps are metadata only — `time_meta` never drives
 * decay or deletion; the only "expiry" is semantic supersession
 * (`invalid_at` / `superseded_by`).
 */

/** Accessibility tier derived from the consolidation score S (§2.1). */
export type MemoryTier = 'hot' | 'warm' | 'cold'

/** Directional edge on the memory graph. Population grows over phases. */
export interface MemoryLink {
  readonly kind: 'supersedes' | 'complements' | 'related'
  readonly target: string
}

/** L0 候选统一视图（§4.1/§4.2）：有效未压缩的情景记录，或有效语义节点的 gist。 */
export interface RetrievableItem {
  readonly id: string
  /** 记录原文或节点 gist。 */
  readonly text: string
  readonly embedding: readonly number[]
  /** CSLS r(m) 项（§3.2）；语义节点在睡眠周期第 7 步刷新。 */
  readonly r_m: number
  readonly tier: MemoryTier
  readonly createdAt: number
  readonly layer: 'episodic' | 'semantic'
  /** M5 情绪判分（仅情景记录；语义节点已去情境化，恒 undefined）。P3-1 情绪重排消费。 */
  readonly valence?: number
  readonly arousal?: number
  /** 图邻居 id（去重：links 目标 + 节点 member_refs；池外 id 由 buildAdjacency 过滤）。P3-3 PPR 消费。 */
  readonly neighbors?: readonly string[]
}

/** M5 structured rubric verdict for one memory (§3.5). */
export interface ImportanceVerdict {
  importance: number // 1–10
  valence: number // −1..1
  arousal: number // 0..1
  reasons: ReadonlyArray<'task_impact' | 'user_identity' | 'decision' | 'emotion' | 'reversible'>
}

/** Simplified contradiction verdict against one similar memory (§2.5, Mem0-style). */
export type SupersessionVerdict = 'contradicts' | 'complements' | 'duplicate' | 'unrelated'

/**
 * 矛盾的时间重叠判定（§2.5 完整版，P2）：'overlapping' = 两条陈述针对同一
 * （或现在时）时间框架，只有此时才软失效旧记忆；'disjoint' = 描述不同时期
 * （如"过去用 MySQL"vs"现在用 PostgreSQL"），两条都真、均保留。判分缺省
 * 视为 overlapping（保持 P0 语义的安全网）。
 */
export type TemporalOverlap = 'overlapping' | 'disjoint'

/** 使用/抑制/衰减记账（§2.1 S 动态的记录侧）。记录侧 P0 落位；节点侧 P2 起可选。 */
export interface ScoreCounters {
  used_count: number
  suppressed_count: number
  last_used_at?: number
  last_cue_quality?: number
  /** 上次兜底衰减锚点（§2.1）；缺省以 time_meta.created_at 为锚（旧记录容忍）。 */
  last_decay_at?: number
  /**
   * 经灵感通道（memory_inspire，P3-4）产生的使用次数——wild-card 使用率 =
   * inspired_count / used_count，灵感效用的量化指标（旧记录缺省按 0 容忍）。
   */
  inspired_count?: number
}

/**
 * 作用域级 mood 状态（P3 情绪编码）：从写入路径的 turn 级情绪聚合（EMA），
 * 读取时按 τ 惰性衰减（INV-7 同架构），持久化于 unit 的 global 槽（单值，
 * 不是记忆——不参与检索宇宙，`memory_clear` 也不清除它）。
 */
export interface MoodState {
  valence: number
  arousal: number
  updated_at: number
}

/** Where this memory came from (session log provenance, for audit only). */
export interface MemoryOrigin {
  readonly session_id: string
  readonly turn: number
  readonly seq: number
}

/**
 * Episodic-layer record (§4.1). `time_meta` is pure metadata; the sole
 * invalidation path is semantic supersession via `flags.superseded_by`.
 */
export interface MemoryRecord {
  readonly id: string
  readonly text: string
  /** Unit-normalized embedding vector (dims from Config). */
  readonly embedding: readonly number[]
  // M5
  readonly imp: number
  readonly valence: number
  readonly arousal: number
  // M6 / M3 / tiering
  epi: number
  red: number
  /** M2: mean cosine to the K=10 library neighbours, precomputed at write time (§3.2). */
  r_m: number
  tier: MemoryTier
  /** M4 routing outcome — `pending` until the P1 sleep cycle attaches nodes. */
  con_route: { kind: 'attached'; nodeIds: readonly string[] } | { kind: 'new_node'; nodeId: string } | { kind: 'pending' }
  links: MemoryLink[]
  /** Consolidation score (§2.1): initialized to `imp` at write time; use/suppress events and fallback decay drive it afterwards (scoring.ts, P2). */
  S: number
  counters: ScoreCounters
  time_meta: {
    created_at: number
    valid_at: number
    invalid_at?: number
    /** 归档时点（P2，§2.3 第 6 步写入）：时间旅行视图据此判定"当时是否还在检索宇宙"。 */
    compressed_at?: number
  }
  flags: {
    superseded_by?: string
    compressed_into?: string
    qa_passed?: boolean
    /** §2.3 第 3 步：QA/epi 闸门拒收的"不可压缩"标记；全簇标记后需新成员加入才重试。 */
    incompressible?: boolean
  }
  origin?: MemoryOrigin
}

/** Semantic-layer node (§4.2). A graph from day one; populated by the P1 sleep cycle. */
export interface SemanticNode {
  readonly id: string
  readonly gist: string
  readonly embedding: readonly number[]
  readonly epi: number
  links: MemoryLink[]
  /** Ids of the compressed episodic records this node summarizes. */
  member_refs: readonly string[]
  time_meta: {
    created_at: number
    valid_at: number
    invalid_at?: number
  }
  S: number
  /** CSLS r(m) 项（§3.2）：与检索宇宙 top-k 近邻的平均余弦，睡眠周期第 7 步刷新。 */
  r_m: number
  /** 使用/抑制/衰减记账（P2，§2.1 节点侧 S 动态）；旧节点缺省按全 0 容忍。 */
  counters?: ScoreCounters
  /** 被新记忆推翻时指向取代者（支柱五完整版，P2；与记录 flags.superseded_by 同义）。 */
  superseded_by?: string
}

/** One QA pair for the sleep-cycle compression gate (§2.3 step 3). */
export interface QaPair {
  readonly question: string
  readonly answer: string
}

/** M4 routing outcome for one cluster inside a sleep cycle (§2.3 step 4). */
export type SleepRoute = 'new_node' | 'attached' | 'stayed' | 'qa_failed' | 'epi_failed'

/** Per-cluster record of one sleep-cycle run, for logging/audit/evaluation. */
export interface SleepClusterOutcome {
  readonly members: readonly string[]
  readonly gist: string
  /** Mean token-level F1 across the cluster's QA pairs (0 when no pairs were produced). */
  readonly qaF1: number
  /** M6 epi of the distilled gist (§3.6 full formula). */
  readonly gistEpi: number
  /** Mean epi of the member records before compression — the drop gistEpi ← memberMeanEpi quantifies detail forgetting. */
  readonly memberMeanEpi: number
  readonly route: SleepRoute
  /** Node created or attached to, when the cluster migrated. */
  readonly nodeId?: string
}

/** Result of one full sleep cycle (§2.3), keeping every cluster for audit. */
export interface SleepOutcome {
  readonly clusters: readonly SleepClusterOutcome[]
  /** How many remaining episodic records had r_m/red refreshed (step 7). */
  readonly refreshedRecords: number
  /** Node ids whose r_m was refreshed (step 7). */
  readonly refreshedNodes: readonly string[]
  /** 第 7 步兜底衰减（§2.1，P2）：本周期应用衰减的记录数 / 节点数。 */
  readonly decayedRecords: number
  readonly decayedNodes: number
}

/** A candidate at one stage of the retrieval pipeline (§5.2). */
export interface ScoredCandidate {
  readonly id: string
  /** Cosine similarity from the L0 stage (kept for all later stages). */
  readonly cos: number
  /** CSLS-corrected cue distinctiveness, present from the CSLS stage on. */
  readonly cue?: number
  /**
   * 情绪相似度（P3-1，仅情绪白名单工作区且候选项带情绪判分时出现）：
   * 排序键为 cue + α·moodSim；**cold 门控仍用纯 cue**（侧门封死）。
   */
  readonly moodSim?: number
  /** Cross-encoder relevance in [0,1], present from the rerank stage on. */
  readonly relevance?: number
  /** Episodic record or semantic node this candidate resolves to. */
  readonly item: RetrievableItem
}

/** The four ablation rungs for nDCG evaluation (design §7.1). */
export type RetrievalAblation = 'base' | 'csls' | 'rerank' | 'full'

/** Result of one full retrieval run, keeping every stage for ablation/audit. */
export interface RetrievalResult {
  readonly query: string
  readonly expandedQuery: string
  /** Survivors after each stage; index 0 = L0 recall, then CSLS, rerank, MMR. */
  readonly stages: {
    readonly l0: readonly ScoredCandidate[]
    readonly csls: readonly ScoredCandidate[]
    readonly rerank: readonly ScoredCandidate[]
    readonly mmr: readonly ScoredCandidate[]
  }
  /** Final output order (top-K after MMR). */
  readonly top: readonly ScoredCandidate[]
  /** Candidates that competed and lost in the final selection (weak-form suppression, §2.2). */
  readonly suppressed: readonly ScoredCandidate[]
  /**
   * 检索来源标记（P3-4）：缺省视为常规检索（memory_search / 自动注入）；
   * `'inspire'` = 灵感通道采样结果——使用事件照常增益 S（效用闭环），但
   * counters.inspired_count 单独累加（wild-card 使用率可量化）。
   */
  readonly source?: 'inspire'
}

/**
 * 灵感采样池项（P3-4，设计 §10.1）：检索宇宙（RetrievableItem 全字段）+
 * 采样偏置所需的记账/判分侧字段。MemoryStore.inspirable() 的投影形状。
 */
export interface InspireItem extends RetrievableItem {
  /** 使用次数（低 used_count = 冷门，采样正向偏置）。 */
  readonly usedCount: number
  /** 冗余度（高 red = 语义网络中心，采样正向偏置）；语义节点无 red，恒 0。 */
  readonly red: number
  /** 巩固分数（counter 对冲臂偏向高 S）。 */
  readonly S: number
  /** 上次使用时点（缺省 = 从未使用）；输出侧"久未激活"框架消费。 */
  readonly lastUsedAt?: number
}

/**
 * Provider-agnostic embedding + rerank interface (§3.1): these models are
 * plugin-owned dependencies — dsh ships no embedding/reranker seam — with
 * endpoints and model ids chosen through Config.
 */
export interface EmbeddingProvider {
  readonly dimensions: number
  embed(texts: readonly string[], signal?: AbortSignal): Promise<readonly number[][]>
}

export interface RerankerProvider {
  /** Cross-encoder relevance scores in [0,1], aligned with `documents`. */
  rerank(query: string, documents: readonly string[], signal?: AbortSignal): Promise<readonly number[]>
}

/** Internal LLM call surface (M5 scoring / contradiction / extraction) over `ctx.llm`. */
export interface InternalLlm {
  complete(system: string, user: string, signal?: AbortSignal): Promise<string>
}

/** One nearest-neighbour hit from the vector index. */
export interface Neighbor {
  readonly id: string
  readonly cos: number
}

/**
 * Exact-kNN vector index over unit-normalized embeddings (P0 scale; the
 * interface leaves room for a real ANN implementation later).
 */
export interface VectorIndex {
  add(id: string, vector: readonly number[]): void
  remove(id: string): boolean
  size(): number
  /** Top-`k` by cosine, optionally filtered (e.g. valid memories only). */
  search(query: readonly number[], k: number, filter?: (id: string) => boolean): readonly Neighbor[]
}

/** Contradiction judgment against one existing valid memory or semantic node (§2.5). */
export interface ContradictionJudgment {
  readonly verdict: SupersessionVerdict
  readonly againstId: string
  /** 仅 verdict === 'contradicts' 时有意义：时间重叠才软失效（§2.5 完整版）。 */
  readonly temporal?: TemporalOverlap
  /**
   * P3-2 扩产出：同一次调用顺带标注的「主题相关」近邻 id（不含主判定目标），
   * 写入路径据此建 related 边（度数上限内）；宽容解析，缺失/非法为空。
   */
  readonly relatedIds?: readonly string[]
}

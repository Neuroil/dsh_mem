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
  /** Consolidation score. P0: initialized to `imp`; full dynamics land in P2. */
  S: number
  counters: {
    used_count: number
    suppressed_count: number
    last_used_at?: number
    last_cue_quality?: number
  }
  time_meta: {
    created_at: number
    valid_at: number
    invalid_at?: number
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
}

/** A candidate at one stage of the retrieval pipeline (§5.2). */
export interface ScoredCandidate {
  readonly id: string
  /** Cosine similarity from the L0 stage (kept for all later stages). */
  readonly cos: number
  /** CSLS-corrected cue distinctiveness, present from the CSLS stage on. */
  readonly cue?: number
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

/** Simplified-contradiction judgment against one existing valid memory. */
export interface ContradictionJudgment {
  readonly verdict: SupersessionVerdict
  readonly againstId: string
}

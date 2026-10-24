/**
 * Plugin configuration (design §5.0): every hyperparameter and default named
 * in the design doc lives in this schema — no hardcoded tunables in code.
 * Invalid configurations fail loud at load; changes take effect via HMR.
 */
import Schema from '@deepseek-ai/schemastery'

export interface HttpModelConfig {
  /** `http` calls the configured endpoint; `deterministic` is an offline test stand-in. */
  mode: 'http' | 'deterministic'
  endpoint: string
  model: string
  apiKey: string
  timeoutMs: number
}

export interface EmbeddingConfig extends HttpModelConfig {
  /** Vector dimensionality, must match the deployed model (bge-m3: 1024). */
  dimensions: number
  batchSize: number
}

export interface JudgeLlmConfig {
  /** Provider route registered on ctx.llm (e.g. `deepseek`). */
  provider: string
  model: string
  temperature: number
  maxTokens: number
}

export interface WriteConfig {
  /** Message window for candidate extraction, Mem0 uses m=10. */
  messageWindow: number
  /** Upper bound of extracted candidates per turn. */
  maxCandidates: number
  /** Drop candidates scoring below this importance (M5 is a 1–10 scale). */
  minImportance: number
  /** How many top-similar valid memories face the contradiction judge. */
  contradictionTopK: number
  /** Cosine floor below which a neighbor is not even shown to the judge. */
  contradictionMinCos: number
}

export interface RetrievalConfig {
  /** L0 recall width (bi-encoder cosine, exact kNN at P0 scale). */
  l0TopK: number
  /** CSLS rerank cut before the cross-encoder stage. */
  cslsTopK: number
  /** Cross-encoder rerank cut. */
  rerankTopK: number
  /** Final MMR output size. */
  mmrTopK: number
  /** MMR relevance/diversity trade-off λ (§3.3). */
  mmrLambda: number
  /** K nearest neighbours for the CSLS r(m) term (§3.2). */
  cslsK: number
  /** Recent turns folded into the expanded query q' (§3.1). */
  contextTurns: number
  /** Extra cue threshold cold-tier memories must clear (§2.1). */
  coldCueBoost: number
  /** Base CSLS cue floor; only cold-tier gating consumes it in P0. */
  cueThreshold: number
}

export interface TierConfig {
  hotThreshold: number
  coldThreshold: number
}

/** 支柱一 S 完整动态（§2.1，P2）。 */
export interface ScoreConfig {
  /** 成功使用增益 α（§2.1：S ← S + α·(1 + cue_quality)，经 arousal 乘法门）。 */
  useGain: number
  /** 抑制事件惩罚 ε（§2.1：S ← S − ε，每查询至多计一次）。 */
  suppressPenalty: number
  /** 兜底衰减时间常数 τ（天；§2.1 S ← S·e^(−1/τ) 每日，只影响分层、永不删除）。 */
  decayTauDays: number
  /** 层级跃迁重判 M5 时新判分的 EMA 权重（§3.5「重判时取 EMA」）。 */
  rejudgeEma: number
}

/** 离线巩固周期（"睡眠"，设计 §2.3/§5.3）。 */
export interface SleepConfig {
  enabled: boolean
  /** 自上次睡眠后新写入记忆数达到该值时触发一次周期（turn/end 信号门控，§5.3）。 */
  minNewMemories: number
  /** 聚类边阈值兼簇内平均 pairwise 余弦下限（§3.3 压缩触发 0.75）。 */
  clusterCos: number
  /** 最小簇大小（§3.3 ≥ 3）。 */
  clusterMinSize: number
  /** 每簇生成的 QA 对数量（§2.3 第 3 步）。 */
  qaPairCount: number
  /** QA 闸门的平均 F1 下限（§2.3 ≥ 0.8 才通过）。 */
  qaF1Threshold: number
  /** gist 的 M6 epi 需低于该值才迁语义层（§2.3 第 5 步"epi 足够低"）。 */
  epiThreshold: number
  /** con z 基线窗口：最近 N 条有效未压缩情景记忆（§3.4 窗口 = 1000）。 */
  conWindow: number
  /** 挂入已有语义节点的 con 下限（§3.4 con > +0.5σ）。 */
  conAttachSigma: number
  /** 建新语义节点的 con 上限（§3.4 con < −1σ）。 */
  conNewNodeSigma: number
  /** 建新语义节点的最低簇均 imp（§3.4 imp ≥ 7）。 */
  conNewNodeMinImp: number
  /** 挂入已有节点时的节点 S 增益（§2.1 α = 0.5 的沿用；使用/抑制/衰减动态见 score 段）。 */
  attachGain: number
}

export interface SectionConfig {
  enabled: boolean
  maxMemories: number
  header: string
}

export interface ToolConfig {
  enabled: boolean
  maxResults: number
  /**
   * `memory_clear` 工具开关（默认 false）：按作用域硬删除全部长期记忆的
   * 开发/测试工具——唯一违反「记录永不删除」（§2.5）的操作，时间旅行也
   * 不可回溯，仅建议测试环境开启。
   */
  clear: boolean
}

/** 会话历史入口（M3）：对话历史的来源与"什么算人类输入"由配置决定。 */
export interface IntakeConfig {
  session: {
    /** dsh 会话事件入口总开关：false = 完全停写（检索通道不受影响）。 */
    enabled: boolean
    /**
     * 哪些 `source.kind` 的 user 消息算人类对话（默认仅 'user'）。IM 适配器
     * 以自定义 kind 注入消息时把该 kind 加进此列表即可；带 `form` 的消息
     * （插件上下文）恒不算对话（intake.ts 守卫）。
     */
    acceptSourceKinds: string[]
  }
}

export interface StorageConfig {
  /** Backend name registered on ctx.storage (`json` / `sqlite`). */
  backend: string
  /**
   * Unit 名前缀 + 无 cwd 会话的回退 unit（M2）：工作区 unit 名由插件派生为
   * `<unit>_ws_<scopeHash>`；必须满足 UNIT_NAME_RE——不允许连字符（宿主规则，
   * 安全作文件/SQL 名）。
   */
  unit: string
  /** Unit format version, bumped on incompatible record-shape changes. */
  version: number
  /** true = 按会话工作目录（session.header.cwd）一区一库；false = 全局单库（旧行为）。 */
  workspaceScope: boolean
}

export interface Config {
  storage: StorageConfig
  embedding: EmbeddingConfig
  reranker: HttpModelConfig
  llm: JudgeLlmConfig
  write: WriteConfig
  retrieval: RetrievalConfig
  tier: TierConfig
  score: ScoreConfig
  sleep: SleepConfig
  section: SectionConfig
  tool: ToolConfig
  intake: IntakeConfig
}

const httpModel = (defaults: { endpoint: string; model: string; timeoutMs: number }) =>
  Schema.object({
    mode: Schema.union(['http', 'deterministic'] as const).default('http'),
    endpoint: Schema.string().default(defaults.endpoint),
    model: Schema.string().default(defaults.model),
    apiKey: Schema.string().default(''),
    timeoutMs: Schema.number().min(100).default(defaults.timeoutMs),
  })

/**
 * 宿主 dsh-storage 的 unit 名规则（UNIT_NAME_RE）：小写字母开头，仅小写字母/
 * 数字/下划线——**不允许连字符**（连字符会被宿主 validateDescriptor 以
 * malformed-medium 拒绝）。schema 与 memoryUnitDescriptor 的本地校验共用。
 */
export const UNIT_NAME_RE = /^[a-z][a-z0-9_]*$/

export const Config: Schema<Config> = Schema.object({
  storage: Schema.object({
    backend: Schema.string().default('json'),
    unit: Schema.string().pattern(UNIT_NAME_RE).default('dsh_mem'),
    version: Schema.natural().default(1),
    workspaceScope: Schema.boolean().default(true),
  }).default({ backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: true }),

  // Embeddings and reranking are plugin-owned dependencies: dsh ships no
  // seam for them, so endpoints/models/dims are configured here (§3.1).
  embedding: Schema.object({
    mode: Schema.union(['http', 'deterministic'] as const).default('http'),
    endpoint: Schema.string().default('https://api.siliconflow.cn/v1/embeddings'),
    model: Schema.string().default('BAAI/bge-m3'),
    apiKey: Schema.string().default(''),
    timeoutMs: Schema.number().min(100).default(30_000),
    dimensions: Schema.natural().min(1).default(1024),
    batchSize: Schema.natural().min(1).default(32),
  }).default({
    mode: 'http',
    endpoint: 'https://api.siliconflow.cn/v1/embeddings',
    model: 'BAAI/bge-m3',
    apiKey: '',
    timeoutMs: 30_000,
    dimensions: 1024,
    batchSize: 32,
  }),

  reranker: httpModel({
    endpoint: 'https://api.siliconflow.cn/v1/rerank',
    model: 'BAAI/bge-reranker-v2-m3',
    timeoutMs: 30_000,
  }).default({
    mode: 'http',
    endpoint: 'https://api.siliconflow.cn/v1/rerank',
    model: 'BAAI/bge-reranker-v2-m3',
    apiKey: '',
    timeoutMs: 30_000,
  }),

  // Internal LLM calls (M5 scoring, contradiction, extraction) always go
  // through ctx.llm's provider-agnostic streaming service (§5.0).
  llm: Schema.object({
    provider: Schema.string().default('deepseek'),
    model: Schema.string().default('deepseek-chat'),
    temperature: Schema.number().min(0).max(2).default(0),
    maxTokens: Schema.natural().default(2048),
  }).default({ provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 2048 }),

  write: Schema.object({
    messageWindow: Schema.natural().min(1).default(10),
    maxCandidates: Schema.natural().min(1).default(8),
    minImportance: Schema.number().min(1).max(10).default(3),
    contradictionTopK: Schema.natural().min(1).default(5),
    contradictionMinCos: Schema.percent().default(0.8),
  }).default({
    messageWindow: 10,
    maxCandidates: 8,
    minImportance: 3,
    contradictionTopK: 5,
    contradictionMinCos: 0.8,
  }),

  retrieval: Schema.object({
    l0TopK: Schema.natural().min(1).default(100),
    cslsTopK: Schema.natural().min(1).default(20),
    rerankTopK: Schema.natural().min(1).default(8),
    mmrTopK: Schema.natural().min(1).default(5),
    mmrLambda: Schema.percent().default(0.6),
    cslsK: Schema.natural().min(1).default(10),
    contextTurns: Schema.natural().min(0).default(3),
    coldCueBoost: Schema.number().min(0).max(1).default(0.1),
    cueThreshold: Schema.number().min(-2).max(2).default(0.3),
  }).default({
    l0TopK: 100,
    cslsTopK: 20,
    rerankTopK: 8,
    mmrTopK: 5,
    mmrLambda: 0.6,
    cslsK: 10,
    contextTurns: 3,
    coldCueBoost: 0.1,
    cueThreshold: 0.3,
  }),

  tier: Schema.object({
    hotThreshold: Schema.number().default(8),
    coldThreshold: Schema.number().default(2),
  }).default({ hotThreshold: 8, coldThreshold: 2 }),

  // 支柱一 S 完整动态（§2.1，P2）
  score: Schema.object({
    useGain: Schema.number().min(0).default(0.5),
    suppressPenalty: Schema.number().min(0).default(0.05),
    decayTauDays: Schema.number().min(1).default(180),
    rejudgeEma: Schema.percent().default(0.5),
  }).default({ useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 }),

  // 离线巩固周期（§2.3/§5.3）：turn/end 信号触发，批处理在空闲期执行
  sleep: Schema.object({
    enabled: Schema.boolean().default(true),
    minNewMemories: Schema.natural().min(1).default(10),
    clusterCos: Schema.percent().default(0.75),
    clusterMinSize: Schema.natural().min(2).default(3),
    qaPairCount: Schema.natural().min(1).default(3),
    qaF1Threshold: Schema.percent().default(0.8),
    epiThreshold: Schema.percent().default(0.5),
    conWindow: Schema.natural().min(1).default(1000),
    conAttachSigma: Schema.number().default(0.5),
    conNewNodeSigma: Schema.number().default(-1),
    conNewNodeMinImp: Schema.number().min(1).max(10).default(7),
    attachGain: Schema.number().min(0).default(0.5),
  }).default({
    enabled: true,
    minNewMemories: 10,
    clusterCos: 0.75,
    clusterMinSize: 3,
    qaPairCount: 3,
    qaF1Threshold: 0.8,
    epiThreshold: 0.5,
    conWindow: 1000,
    conAttachSigma: 0.5,
    conNewNodeSigma: -1,
    conNewNodeMinImp: 7,
    attachGain: 0.5,
  }),

  section: Schema.object({
    enabled: Schema.boolean().default(true),
    maxMemories: Schema.natural().min(1).default(5),
    header: Schema.string().default('Relevant long-term memories (semantic retrieval; recency was not a factor):'),
  }).default({ enabled: true, maxMemories: 5, header: 'Relevant long-term memories (semantic retrieval; recency was not a factor):' }),

  tool: Schema.object({
    enabled: Schema.boolean().default(true),
    maxResults: Schema.natural().min(1).default(5),
    clear: Schema.boolean().default(false),
  }).default({ enabled: true, maxResults: 5, clear: false }),

  // 会话历史入口（M3）：见 IntakeConfig 注释
  intake: Schema.object({
    session: Schema.object({
      enabled: Schema.boolean().default(true),
      acceptSourceKinds: Schema.array(Schema.string()).default(['user']),
    }).default({ enabled: true, acceptSourceKinds: ['user'] }),
  }).default({ session: { enabled: true, acceptSourceKinds: ['user'] } }),
})

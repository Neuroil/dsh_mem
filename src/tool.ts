/**
 * The `memory_search` tool — the model-driven recall channel (§5.2). The
 * tool returns a single canonical JSON value; `render` is a pure projection
 * of that value (tool contract, §5.0).
 */
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Config } from './config.js'
import { cosine } from './metrics.js'
import { inspireSeed, localDateKey, mulberry32, sampleInspirations } from './inspire.js'
import type { RetrieveOptions } from './retrieval.js'
import type { MemoryRetriever } from './retrieval.js'
import type {
  EmbeddingProvider,
  InspireItem,
  MemoryTier,
  MoodState,
  RetrievalResult,
  ScoredCandidate,
} from './types.js'

/** 按 agent 解析出的作用域内检索依赖（M2：检索与计分必须落在同一 scope）。 */
export interface MemoryScopeDeps {
  readonly retriever: MemoryRetriever
  /** 使用记账钩子（winners/suppressed counters），fire-and-forget。 */
  readonly recordRetrieval: (result: RetrievalResult) => void
  /** 情绪重排上下文（P3-1）：仅情绪白名单工作区且该作用域已有 mood 时出现。 */
  readonly mood?: { readonly state: MoodState; readonly alpha: number }
}

export interface MemorySearchDeps {
  /** 由 agent 会话的 cwd 解析 scope 并返回该作用域的检索/计分依赖。 */
  readonly resolve: (agent: Agent | undefined) => Promise<MemoryScopeDeps>
  readonly recentTurns: (agent: Agent | undefined) => readonly string[]
  readonly config: Config
}

interface MemoryHit {
  id: string
  text: string
  /** Best available score: rerank relevance, else CSLS cue, else cosine. */
  score: number
  cue: number | undefined
  tier: MemoryTier
  /** 'semantic' hits are distilled gists from the sleep cycle; 'episodic' are raw memories. */
  layer: 'episodic' | 'semantic'
  createdAt: number
}

interface SearchOutput {
  query: string
  expandedQuery: string
  memories: MemoryHit[]
  suppressedCount: number
  /** 时间旅行时点（epoch ms；仅在调用方传了合法 `at` 时出现）。 */
  at?: number
  /** 领域性不良输入（如无法解析的 `at`）写入规范值而非抛异常（工具契约）。 */
  error?: string
}

function toHit(candidate: ScoredCandidate): MemoryHit {
  return {
    id: candidate.id,
    text: candidate.item.text,
    score: candidate.relevance ?? candidate.cue ?? candidate.cos,
    cue: candidate.cue,
    tier: candidate.item.tier,
    layer: candidate.item.layer,
    createdAt: candidate.item.createdAt,
  }
}

/**
 * 解析时间旅行参数（§2.5 完整版）：纯数字串按 epoch ms、其余按 ISO 8601
 * （须以 4 位年份开头，Date.parse 对杂串过于宽容——如 '-5' 也能解析出正
 * 时间戳）；非法或非正时点（记忆库不存在 1970 前数据）返回 undefined
 * （调用方转为规范错误值）。
 */
export function parseAt(raw: string): number | undefined {
  const trimmed = raw.trim()
  if (trimmed === '') return undefined
  if (/^\d+$/.test(trimmed)) {
    const ms = Number(trimmed)
    return Number.isFinite(ms) && ms > 0 ? ms : undefined
  }
  if (!/^\d{4}/.test(trimmed)) return undefined
  const parsed = Date.parse(trimmed)
  return Number.isNaN(parsed) || parsed <= 0 ? undefined : parsed
}

/** Render retrieval winners as a prompt-section block (design §5.2). */
export function formatMemories(result: RetrievalResult, max: number, header: string): string {
  if (result.top.length === 0) return ''
  const lines = result.top.slice(0, max).map((candidate, i) => {
    const score = candidate.relevance ?? candidate.cue ?? candidate.cos
    const gistTag = candidate.item.layer === 'semantic' ? ' type=gist' : ''
    return `${i + 1}. ${candidate.item.text}  [id=${candidate.id} score=${score.toFixed(3)} tier=${candidate.item.tier}${gistTag}]`
  })
  return `${header}\n${lines.join('\n')}`
}

export function createMemorySearchTool(deps: MemorySearchDeps) {  const maxResults = deps.config.tool.maxResults
  return defineTool({
    name: 'memory_search',
    description:
      'Search the user\'s long-term memory. Ranking is semantic (embedding recall, ' +
      'hubness-corrected distinctiveness, cross-encoder relevance, diversity): a ' +
      'decade-old but relevant memory outranks a recent irrelevant one. Superseded ' +
      'memories are excluded unless you pass `at` to query a past memory state. Use ' +
      'this whenever prior preferences, decisions or facts may matter.',
    parameters: {
      query: { type: 'string', required: true, description: 'What to recall, in natural language.' },
      // 注：参数 DSL 不支持数值边界（仅 type/enum/const/注解键），limit ≥ 1 在
      // execute 内做领域级校验（非法值写规范 error 值，不抛异常）。
      limit: { type: 'integer', description: 'Maximum memories to return (default from config).' },
      at: {
        type: 'string',
        description:
          'Optional time-travel point, ISO 8601 or epoch milliseconds: search the memory ' +
          'state as of that time instead of now (superseded memories from before that point ' +
          'resurface). Use for questions about what was true at some past moment.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          query: { type: 'string', required: true },
          expandedQuery: { type: 'string', required: true },
          memories: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                id: { type: 'string', required: true },
                text: { type: 'string', required: true },
                score: { type: 'number', required: true },
                cue: { type: 'number' },
                tier: { type: 'string', enum: ['hot', 'warm', 'cold'], required: true },
                layer: { type: 'string', enum: ['episodic', 'semantic'], required: true },
                createdAt: { type: 'integer', required: true },
              },
            },
          },
          suppressedCount: { type: 'integer', required: true },
          at: { type: 'integer' },
          error: { type: 'string' },
        },
      },
      render: (_args, value: SearchOutput) => [
        {
          type: 'text',
          text: value.error !== undefined
            ? `memory_search failed: ${value.error}`
            : value.memories.length === 0
              ? 'No relevant memories found.'
              : value.memories.map((m, i) => `${i + 1}. ${m.text} [id=${m.id} score=${m.score.toFixed(3)}${m.layer === 'semantic' ? ' gist' : ''}]`).join('\n'),
        },
      ],
    },
    presentCall: (args) => ({
      card: 'generic',
      title: `memory_search: ${args.query}`,
      kind: 'search',
    }),
    presentResult: (_args) => ({
      card: 'generic',
      title: 'memory_search',
    }),
    async execute(args, exec) {
      let at: number | undefined
      if (args.at !== undefined) {
        at = parseAt(args.at)
      }
      if (args.at !== undefined && at === undefined) {
        const badAt: SearchOutput = {
          query: args.query,
          expandedQuery: args.query,
          memories: [],
          suppressedCount: 0,
          error: `invalid 'at' timestamp: ${args.at} (expected ISO 8601 or epoch milliseconds)`,
        }
        return badAt
      }
      if (args.limit !== undefined && args.limit < 1) {
        const badLimit: SearchOutput = {
          query: args.query,
          expandedQuery: args.query,
          memories: [],
          suppressedCount: 0,
          error: `invalid 'limit': ${args.limit} (must be >= 1)`,
        }
        return badLimit
      }
      const scope = await deps.resolve(exec.agent)
      const result = await scope.retriever.retrieve(args.query, {
        recentTurns: deps.recentTurns(exec.agent),
        signal: exec.signal,
        topK: args.limit ?? maxResults,
        at,
        ...(scope.mood !== undefined ? { mood: scope.mood } : {}),
      } satisfies RetrieveOptions)
      scope.recordRetrieval(result)
      const output: SearchOutput = {
        query: args.query,
        expandedQuery: result.expandedQuery,
        memories: result.top.map(toHit),
        suppressedCount: result.suppressed.length,
        ...(at !== undefined ? { at } : {}),
      }
      return output
    },
  })
}

// ---- memory_inspire（P3-4 灵感通道） ----

/** 按 agent 解析出的灵感通道作用域依赖。 */
export interface MemoryInspireScope {
  /** 作用域键（seed 分量：不同工作区灵感流不同）。 */
  readonly scopeKey: string | undefined
  /** 灵感采样池（store.inspirable()：检索宇宙 + usedCount/red/S）。 */
  readonly inspirable: readonly InspireItem[]
  /** 工作区是否在情绪白名单内（false = agent 工作区，工具返回规范拒绝值）。 */
  readonly emotionEnabled: boolean
  /** 惰性衰减后的当前 mood（白名单工作区且已有时出现；counter 对冲消费）。 */
  readonly mood: MoodState | null
  /** 灵感使用记账（S 增益照常 + inspired_count，效用闭环的入账点）。 */
  readonly recordInspiration: (result: RetrievalResult) => void
}

export interface MemoryInspireDeps {
  readonly resolve: (agent: Agent | undefined) => Promise<MemoryInspireScope>
  /** topic 弱锚定的嵌入提供方（仅 topic 非空时调用一次）。 */
  readonly embedding: EmbeddingProvider
  readonly config: Config
  /** 日期分量注入点（测试可固定；缺省取本地当下）。 */
  readonly now?: () => Date
}

interface InspireHit {
  id: string
  text: string
  layer: 'episodic' | 'semantic'
  tier: MemoryTier
  /** 历史使用次数（0 = 从未激活）。 */
  usedCount: number
  lastUsedAt?: number
  /** topic 余弦（无 topic 时 0）。 */
  score: number
  /** 来源标记恒 true：灵感通道输出（区别于 memory_search 的语义检索）。 */
  inspired: true
}

interface InspireOutput {
  memories: InspireHit[]
  /** 领域性拒绝（agent 工作区 / 空库）写入规范值而非抛异常（工具契约）。 */
  error?: string
}

/**
 * The `memory_inspire` tool — the serendipity channel (P3-4, design §10.1):
 * a seeded random walk over the memory graph surfaces long-dormant but
 * semantically connected memories. Deterministic per (workspace, day, topic):
 * the same topic on the same day returns the same memories; drift across days
 * provides natural spacing. **Companion workspaces only** (emotion.workspaces
 * whitelist) — agent workspaces get a canonical error value, never samples.
 * Not part of automatic injection until wild-card usage data justifies it.
 */
export function createMemoryInspireTool(deps: MemoryInspireDeps) {
  return defineTool({
    name: 'memory_inspire',
    description:
      'Serendipitous recall for companion conversations: surfaces long-unused but ' +
      'semantically connected memories via a seeded random walk over the memory graph ' +
      '(biased toward rarely-used, well-connected ones). Deterministic per day and topic — ' +
      'the same topic within a day returns the same memories. Only available in companion ' +
      'workspaces; not a substitute for memory_search (no relevance ranking).',
    parameters: {
      topic: {
        type: 'string',
        description:
          'Optional weak anchor: biases sampling toward memories related to this topic ' +
          '(omit for pure serendipity).',
      },
      mood: {
        type: 'string',
        enum: ['counter'],
        description:
          "Pass 'counter' to hedge the current mood: with some probability, sampling shifts " +
          'toward high-consolidation memories of opposite emotional valence (e.g. uplifting ' +
          'memories when the user is down).',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          memories: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                id: { type: 'string', required: true },
                text: { type: 'string', required: true },
                layer: { type: 'string', enum: ['episodic', 'semantic'], required: true },
                tier: { type: 'string', enum: ['hot', 'warm', 'cold'], required: true },
                usedCount: { type: 'integer', required: true },
                lastUsedAt: { type: 'integer' },
                score: { type: 'number', required: true },
                // 恒 true（灵感通道输出的来源标记）；不用 const 键——输出 schema
                // 未验证过 const 支持（F-11 教训：注册期校验失败会杀掉整个插件）
                inspired: { type: 'boolean', required: true },
              },
            },
          },
          error: { type: 'string' },
        },
      },
      render: (_args, value: InspireOutput) => [
        {
          type: 'text',
          text: value.error !== undefined
            ? `memory_inspire refused: ${value.error}`
            : value.memories.length === 0
              ? 'No memories to draw inspiration from yet.'
              : 'Long-dormant memories that might be worth revisiting:\n' +
                value.memories
                  .map((m, i) => `${i + 1}. ${m.text} [id=${m.id} used ${m.usedCount}×${m.lastUsedAt !== undefined ? `, last used ${new Date(m.lastUsedAt).toISOString().slice(0, 10)}` : ', never used'}${m.layer === 'semantic' ? ' gist' : ''}]`)
                  .join('\n'),
        },
      ],
    },
    presentCall: (args) => ({
      card: 'generic',
      title: `memory_inspire${args.topic !== undefined ? `: ${args.topic}` : ''}`,
      kind: 'search',
    }),
    presentResult: () => ({
      card: 'generic',
      title: 'memory_inspire',
    }),
    async execute(args, exec) {
      const scope = await deps.resolve(exec.agent)
      // agent（非白名单）工作区拒绝：灵感通道是陪伴场景特性，agent 助手零影响
      if (!scope.emotionEnabled) {
        const refused: InspireOutput = {
          memories: [],
          error: 'inspiration channel is only available in companion workspaces (emotion.workspaces); use memory_search instead',
        }
        return refused
      }
      if (scope.inspirable.length === 0) {
        const empty: InspireOutput = { memories: [], error: 'no memories in this workspace yet' }
        return empty
      }
      const topic = typeof args.topic === 'string' && args.topic.trim() !== '' ? args.topic.trim() : undefined
      let topicVector: readonly number[] | undefined
      if (topic !== undefined) {
        const embedded = await deps.embedding.embed([topic], exec.signal)
        topicVector = embedded[0]
      }
      // seed = fnv1a(scope ⊕ 本地日期 ⊕ topic)：同日同话题可复现、跨日漂移
      const now = deps.now?.() ?? new Date()
      const seed = inspireSeed(scope.scopeKey, localDateKey(now), topic ?? '')
      const counterMood =
        args.mood === 'counter' && scope.mood !== null && scope.mood.valence !== 0
          ? { valence: scope.mood.valence, probability: deps.config.inspire.counterProbability }
          : undefined
      const picked = sampleInspirations(scope.inspirable, {
        count: deps.config.tool.maxResults,
        rng: mulberry32(seed),
        ...(topicVector !== undefined ? { topicVector } : {}),
        ...(counterMood !== undefined ? { counterMood } : {}),
      })
      // 效用闭环：灵感使用照常入账（S 增益 + inspired_count），被使用的记忆
      // 沉回正常 top-K、下一个冷门者顶上——S 动态自动充当灵感质量裁判
      scope.recordInspiration({
        query: topic ?? '',
        expandedQuery: topic ?? '',
        stages: { l0: [], csls: [], rerank: [], mmr: [] },
        top: picked.map((item) => ({
          id: item.id,
          cos: topicVector !== undefined ? cosine(topicVector, item.embedding) : 0,
          item,
        })),
        suppressed: [],
        source: 'inspire',
      })
      const output: InspireOutput = {
        memories: picked.map((item) => ({
          id: item.id,
          text: item.text,
          layer: item.layer,
          tier: item.tier,
          usedCount: item.usedCount,
          ...(item.lastUsedAt !== undefined ? { lastUsedAt: item.lastUsedAt } : {}),
          score: topicVector !== undefined ? cosine(topicVector, item.embedding) : 0,
          inspired: true,
        })),
      }
      return output
    },
  })
}

interface ClearOutput {
  episodic: number
  semantic: number
  total: number
  /** 领域性拒绝（confirm 闸）写入规范值而非抛异常（工具契约）。 */
  error?: string
}

export interface MemoryClearDeps {
  /** 由 agent 会话的 cwd 解析 scope 并返回该作用域的清空操作（runtime 复位睡眠计数）。 */
  readonly resolve: (agent: Agent | undefined) => Promise<{
    clear: () => Promise<{ episodic: number; semantic: number }>
  }>
}

/**
 * The `memory_clear` tool — a destructive dev/test utility: hard-deletes every
 * long-term memory in the calling session's workspace store, the ONLY operation
 * in the system that violates the "records are never deleted" invariant (§2.5).
 * Gated behind `config.tool.clear` (default false) and an explicit `confirm`
 * argument; superseded/archived memories are gone too and time travel cannot
 * bring them back.
 */
export function createMemoryClearTool(deps: MemoryClearDeps) {
  return defineTool({
    name: 'memory_clear',
    description:
      'Permanently delete ALL long-term memories in this workspace\'s memory store: ' +
      'episodic memories and distilled semantic gists alike, including superseded and ' +
      'archived ones. Destructive and irreversible — time-travel search will NOT bring ' +
      'them back. Only call this when the user explicitly asks to wipe or reset ' +
      'long-term memory (e.g. for test iteration), and always pass confirm=true.',
    parameters: {
      confirm: {
        type: 'boolean',
        required: true,
        description: 'Must be exactly true to acknowledge the destructive wipe.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          episodic: { type: 'integer', required: true },
          semantic: { type: 'integer', required: true },
          total: { type: 'integer', required: true },
          error: { type: 'string' },
        },
      },
      render: (_args, value: ClearOutput) => [
        {
          type: 'text',
          text: value.error !== undefined
            ? `memory_clear refused: ${value.error}`
            : `Cleared ${value.total} long-term memories (${value.episodic} episodic, ${value.semantic} semantic nodes).`,
        },
      ],
    },
    presentCall: () => ({
      card: 'generic',
      title: 'memory_clear: wipe all long-term memories (irreversible)',
      kind: 'delete',
    }),
    presentResult: () => ({
      card: 'generic',
      title: 'memory_clear',
    }),
    async execute(args, exec) {
      // 领域级 confirm 闸：宿主参数校验只保证 confirm 是 boolean，false 在此
      // 拒绝（规范 error 值，不抛异常，不触碰存储）。
      if (args.confirm !== true) {
        const refused: ClearOutput = {
          episodic: 0,
          semantic: 0,
          total: 0,
          error: 'confirm must be exactly true (destructive, irreversible wipe of all long-term memories)',
        }
        return refused
      }
      const scope = await deps.resolve(exec.agent)
      const counts = await scope.clear()
      const output: ClearOutput = {
        episodic: counts.episodic,
        semantic: counts.semantic,
        total: counts.episodic + counts.semantic,
      }
      return output
    },
  })
}

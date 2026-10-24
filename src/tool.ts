/**
 * The `memory_search` tool — the model-driven recall channel (§5.2). The
 * tool returns a single canonical JSON value; `render` is a pure projection
 * of that value (tool contract, §5.0).
 */
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { MemoryRetriever } from './retrieval.js'
import type { Config } from './config.js'
import type { MemoryTier, RetrievalResult, ScoredCandidate } from './types.js'

export interface MemorySearchDeps {
  readonly retriever: MemoryRetriever
  readonly recentTurns: (agent: Agent | undefined) => readonly string[]
  /** Usage accounting hook (winners/suppressed counters), fire-and-forget. */
  readonly onRetrieved: (result: RetrievalResult) => void
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

export function createMemorySearchTool(deps: MemorySearchDeps) {
  const maxResults = deps.config.tool.maxResults
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
      const result = await deps.retriever.retrieve(args.query, {
        recentTurns: deps.recentTurns(exec.agent),
        signal: exec.signal,
        topK: args.limit ?? maxResults,
        at,
      })
      deps.onRetrieved(result)
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

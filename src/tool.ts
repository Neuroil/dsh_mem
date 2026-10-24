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
      'decade-old but relevant memory outranks a recent irrelevant one. Use this ' +
      'whenever prior preferences, decisions or facts may matter.',
    parameters: {
      query: { type: 'string', required: true, description: 'What to recall, in natural language.' },
      limit: { type: 'integer', min: 1, description: 'Maximum memories to return (default from config).' },
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
        },
      },
      render: (_args, value: SearchOutput) => [
        {
          type: 'text',
          text: value.memories.length === 0
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
      const result = await deps.retriever.retrieve(args.query, {
        recentTurns: deps.recentTurns(exec.agent),
        signal: exec.signal,
        topK: args.limit ?? maxResults,
      })
      deps.onRetrieved(result)
      const output: SearchOutput = {
        query: args.query,
        expandedQuery: result.expandedQuery,
        memories: result.top.map(toHit),
        suppressedCount: result.suppressed.length,
      }
      return output
    },
  })
}

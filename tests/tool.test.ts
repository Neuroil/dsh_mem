/**
 * memory_search 工具（src/tool.ts）的行为测试：规范 JSON 输出、`at` 时间旅行
 * 参数（合法 ISO/epoch、非法→规范 error 值）、render 纯投影、onRetrieved 记账钩子。
 * defineTool 返回的 ToolDefinition.execute 可直接调用（内部先做参数校验）。
 */
import { describe, expect, it, vi } from 'vitest'
import type { ToolExecution } from '@deepseek-ai/dsh-tools'
import { createMemorySearchTool, formatMemories, parseAt } from '../src/tool.js'
import type { MemoryRetriever } from '../src/retrieval.js'
import type { Config } from '../src/config.js'
import type { RetrievalResult } from '../src/types.js'

const config: Config = {
  storage: { backend: 'json', unit: 'dsh-mem', version: 1 },
  embedding: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000, dimensions: 8, batchSize: 8 },
  reranker: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000 },
  llm: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 2048 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: { l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6, cslsK: 10, contextTurns: 3, coldCueBoost: 0.1, cueThreshold: 0.3 },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: { enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3, qaF1Threshold: 0.8, epiThreshold: 0.5, conWindow: 1000, conAttachSigma: 0.5, conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5 },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
}

/** 检索器替身：记录调用参数并回放固定结果。 */
function makeRetriever(result: RetrievalResult) {
  const retrieve = vi.fn(async () => result)
  return { retriever: { retrieve } as unknown as MemoryRetriever, retrieve }
}

function emptyResult(): RetrievalResult {
  return { query: '', expandedQuery: '', stages: { l0: [], csls: [], rerank: [], mmr: [] }, top: [], suppressed: [] }
}

/** 最小执行上下文（execute 只读 signal 与 agent）。 */
const exec = { callId: 'c1', name: 'memory_search', arguments: {} } as unknown as ToolExecution

describe('parseAt（时间旅行参数解析）', () => {
  it('纯数字串按 epoch ms', () => {
    expect(parseAt('1760000000000')).toBe(1_760_000_000_000)
    expect(parseAt(' 1760000000000 ')).toBe(1_760_000_000_000)
  })
  it('ISO 8601 字符串按 Date.parse', () => {
    expect(parseAt('2026-10-24T00:00:00Z')).toBe(Date.parse('2026-10-24T00:00:00Z'))
    expect(parseAt('2026-10-24')).toBe(Date.parse('2026-10-24'))
  })
  it('非法输入返回 undefined', () => {
    expect(parseAt('not a date')).toBeUndefined()
    expect(parseAt('')).toBeUndefined()
    expect(parseAt('0')).toBeUndefined()
    expect(parseAt('-5')).toBeUndefined()
  })
})

describe('memory_search.execute', () => {
  it('无 at：正常检索、规范输出不带 at、onRetrieved 触发', async () => {
    const { retriever, retrieve } = makeRetriever(emptyResult())
    const onRetrieved = vi.fn()
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved, config })

    const output = await tool.execute({ query: 'which database?' }, exec) as Record<string, unknown>

    expect(retrieve).toHaveBeenCalledTimes(1)
    expect(retrieve.mock.calls[0]![1]).toMatchObject({ topK: 5 })
    expect((retrieve.mock.calls[0]![1] as Record<string, unknown>)['at']).toBeUndefined()
    expect(onRetrieved).toHaveBeenCalledTimes(1)
    expect(output).not.toHaveProperty('at')
    expect(output).not.toHaveProperty('error')
    expect(output['query']).toBe('which database?')
  })

  it('合法 at（ISO）：at 透传检索器并回显到规范输出', async () => {
    const at = Date.parse('2026-01-01T00:00:00Z')
    const { retriever, retrieve } = makeRetriever(emptyResult())
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved: () => {}, config })

    const output = await tool.execute({ query: '当时用什么数据库？', at: '2026-01-01T00:00:00Z' }, exec) as Record<string, unknown>

    expect(retrieve.mock.calls[0]![1]).toMatchObject({ at })
    expect(output['at']).toBe(at)
  })

  it('非法 at：不调检索器、规范 error 值（不抛异常）', async () => {
    const { retriever, retrieve } = makeRetriever(emptyResult())
    const onRetrieved = vi.fn()
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved, config })

    const output = await tool.execute({ query: 'q', at: 'yesterday-ish' }, exec) as Record<string, unknown>

    expect(retrieve).not.toHaveBeenCalled()
    expect(onRetrieved).not.toHaveBeenCalled()
    expect(output['error']).toContain("invalid 'at'")
    expect(output['memories']).toEqual([])
  })

  it('limit 覆盖默认条数（topK 透传）', async () => {
    const { retriever, retrieve } = makeRetriever(emptyResult())
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved: () => {}, config })

    await tool.execute({ query: 'q', limit: 3 }, exec)

    expect(retrieve.mock.calls[0]![1]).toMatchObject({ topK: 3 })
  })

  it('limit < 1：不调检索器、规范 error 值（参数 DSL 无数值边界，领域级校验兜底）', async () => {
    const { retriever, retrieve } = makeRetriever(emptyResult())
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved: () => {}, config })

    const output = await tool.execute({ query: 'q', limit: 0 }, exec) as Record<string, unknown>

    expect(retrieve).not.toHaveBeenCalled()
    expect(output['error']).toContain("invalid 'limit'")
  })
})

describe('render / formatMemories（规范值纯投影）', () => {
  it('error 值渲染为失败文本', async () => {
    const { retriever } = makeRetriever(emptyResult())
    const tool = createMemorySearchTool({ retriever, recentTurns: () => [], onRetrieved: () => {}, config })
    const output = await tool.execute({ query: 'q', at: 'nope' }, exec) as never
    const blocks = tool.output.render({ query: 'q', at: 'nope' }, output)
    expect(blocks[0]!.type).toBe('text')
    expect(String((blocks[0] as { text: string }).text)).toContain('memory_search failed')
  })
})

describe('formatMemories（section 渲染）', () => {
  it('空结果返回空串（不注入 section）', () => {
    expect(formatMemories(emptyResult(), 5, 'memories:')).toBe('')
  })
})

import { describe, expect, it } from 'vitest'
import { WritePipeline, type MemoryStoreLike } from '../src/write-path.js'
import type { Config } from '../src/config.js'
import type { EmbeddingProvider, InternalLlm, MemoryRecord, SemanticNode } from '../src/types.js'

const config: Config = {
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: false },
  embedding: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000, dimensions: 8, batchSize: 8 },
  reranker: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000 },
  llm: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 2048 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: { l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6, cslsK: 10, contextTurns: 3, coldCueBoost: 0.1, cueThreshold: 0.3, pprSeedK: 8, pprDamping: 0.5, pprWeight: 0.2 },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: { enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3, qaF1Threshold: 0.8, epiThreshold: 0.5, conWindow: 1000, conAttachSigma: 0.5, conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5 },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
  emotion: { workspaces: [], alpha: 0.15, tauHours: 4, disclose: false },
  graph: { synonymCos: 0.8, maxDegree: 8 },
  inspire: { counterProbability: 0.25 },
}

function makeRecord(over: Partial<MemoryRecord> = {}): MemoryRecord {
  const now = 1_000
  return {
    id: over.id ?? 'existing-1',
    text: over.text ?? 'existing text',
    embedding: over.embedding ?? [1, 0, 0, 0, 0, 0, 0, 0],
    imp: 5, valence: 0, arousal: 0,
    epi: 0, red: 0, r_m: 0, tier: 'warm',
    con_route: { kind: 'pending' }, links: [], S: 5,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: now, valid_at: now },
    flags: {},
    ...over,
  }
}

function makeNode(over: Partial<SemanticNode> = {}): SemanticNode {
  return {
    id: over.id ?? 'node-1',
    gist: over.gist ?? 'node gist',
    embedding: over.embedding ?? [1, 0, 0, 0, 0, 0, 0, 0],
    epi: 0, links: [], member_refs: [],
    time_meta: { created_at: 1_000, valid_at: 1_000 },
    S: 5, r_m: 0,
    ...over,
  }
}

/** 记录调用序的假存储。 */
class FakeStore implements MemoryStoreLike {
  readonly records = new Map<string, MemoryRecord>()
  readonly nodes = new Map<string, SemanticNode>()
  readonly softInvalidations: Array<{ id: string; by: string }> = []
  readonly nodeInvalidations: Array<{ id: string; by: string }> = []
  readonly refreshed: string[] = []
  similarCalls: number = 0
  similarNodesCalls: number = 0
  /** 控制 similar() 返回什么；为空时对已入库记录算真实余弦（与 MemoryStore 语义一致）。 */
  nextNeighbors: Array<{ id: string; cos: number; record: MemoryRecord }> = []
  /** 控制 similarNodes() 返回什么。 */
  nextNodeNeighbors: Array<{ id: string; cos: number; node: SemanticNode }> = []

  validMemories(): readonly MemoryRecord[] { return [...this.records.values()] }
  similar(vector: readonly number[], _k: number): readonly { id: string; cos: number; record: MemoryRecord }[] {
    this.similarCalls++
    if (this.nextNeighbors.length > 0) return this.nextNeighbors
    return [...this.records.values()]
      .map((record) => ({ id: record.id, cos: cosine(vector, record.embedding), record }))
      .sort((a, b) => b.cos - a.cos)
  }
  similarNodes(vector: readonly number[], _k: number): readonly { id: string; cos: number; node: SemanticNode }[] {
    this.similarNodesCalls++
    if (this.nextNodeNeighbors.length > 0) return this.nextNodeNeighbors
    return [...this.nodes.values()]
      .map((node) => ({ id: node.id, cos: cosine(vector, node.embedding), node }))
      .sort((a, b) => b.cos - a.cos)
  }
  async putRecord(record: MemoryRecord): Promise<void> { this.records.set(record.id, record) }
  async softInvalidate(id: string, by: string, _at: number): Promise<void> { this.softInvalidations.push({ id, by }) }
  async softInvalidateNode(id: string, by: string, _at: number): Promise<void> {
    this.nodeInvalidations.push({ id, by })
    const node = this.nodes.get(id)
    if (node) this.nodes.set(id, { ...node, time_meta: { ...node.time_meta, invalid_at: 1 }, superseded_by: by })
  }
  async refreshNeighborhood(id: string): Promise<{ r_m: number; red: number }> {
    this.refreshed.push(id)
    const record = this.records.get(id)
    if (record) this.records.set(id, { ...record, r_m: 0.42, red: 0.3 })
    return { r_m: 0.42, red: 0.3 }
  }
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] ?? 0
    const y = b[i] ?? 0
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

class FakeLlm implements InternalLlm {
  private queue: string[]
  readonly completions: Array<{ system: string; user: string; signal?: AbortSignal }> = []
  constructor(replies: string[]) { this.queue = [...replies] }
  async complete(system: string, user: string, signal?: AbortSignal): Promise<string> {
    this.completions.push({ system, user, signal })
    const reply = this.queue.shift()
    if (reply === undefined) throw new Error('FakeLlm exhausted')
    return reply
  }
}

const embedding: EmbeddingProvider = {
  dimensions: 8,
  async embed(texts: readonly string[]) {
    return texts.map((text, i) => {
      const v = new Array(8).fill(0)
      v[i % 8] = 1
      void text
      return v
    })
  },
}

const MESSAGES = [{ role: 'user' as const, text: '我把默认数据库从 MySQL 换成了 Postgres' }]

function importances(score: number): string[] {
  // 每个候选一次 M5 调用
  return [`{"importance":${score},"valence":0,"arousal":0,"reasons":["decision"]}`]
}

describe('WritePipeline.processTurn', () => {
  it('全流程：抽取→M5→（无矛盾）→入库→刷新邻域；S=imp、tier 正确', async () => {
    const store = new FakeStore()
    const llm = new FakeLlm([
      '{"memories": ["用户把默认数据库从 MySQL 迁移到 Postgres"], "turn_mood": {"valence": 0, "arousal": 0}}', // 抽取
      ...importances(9), // M5 → hot
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config, now: () => 1234 })
    const outcome = await pipeline.processTurn('s1', 3, MESSAGES)

    expect(outcome.written.length).toBe(1)
    expect(outcome.dropped).toBe(0)
    const record = outcome.written[0]!
    expect(record.S).toBe(9)
    expect(record.tier).toBe('hot')
    expect(record.imp).toBe(9)
    expect(record.time_meta.valid_at).toBe(1234)
    expect(record.origin).toEqual({ session_id: 's1', turn: 3, seq: 0 })
    expect(store.refreshed).toEqual([record.id])
    expect(store.records.get(record.id)!.r_m).toBeCloseTo(0.42)
  })

  it('M5 低分候选被丢弃不入库', async () => {
    const store = new FakeStore()
    const llm = new FakeLlm(['{"memories": ["闲聊一句话"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(2)])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written).toEqual([])
    expect(outcome.dropped).toBe(1)
    expect(store.records.size).toBe(0)
    // 低分在矛盾判定之前丢弃：similar 不应被调用
    expect(store.similarCalls).toBe(0)
  })

  it('contradicts：softInvalidate 先于 putRecord，新记录带 supersedes 链接', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'old-1', text: '用户默认数据库是 MySQL' })
    store.records.set('old-1', existing)
    store.nextNeighbors = [{ id: 'old-1', cos: 0.93, record: existing }]
    const llm = new FakeLlm([
      '{"memories": ["用户把默认数据库从 MySQL 换成了 Postgres"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(7),
      '{"index":0,"verdict":"contradicts"}',
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config, now: () => 2000 })
    const outcome = await pipeline.processTurn('s1', 2, MESSAGES)

    expect(outcome.superseded).toEqual([{ id: 'old-1', by: outcome.written[0]!.id }])
    expect(store.softInvalidations).toEqual([{ id: 'old-1', by: outcome.written[0]!.id }])
    expect(outcome.written[0]!.links).toEqual([{ kind: 'supersedes', target: 'old-1' }])
  })

  it('complements：新记录带 complements 链接，旧记录不动', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'old-2', text: '用户在维护数据库迁移工具' })
    store.nextNeighbors = [{ id: 'old-2', cos: 0.85, record: existing }]
    const llm = new FakeLlm(['{"memories": ["用户为迁移工具写了回滚脚本"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(6), '{"index":0,"verdict":"complements"}'])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written[0]!.links).toEqual([{ kind: 'complements', target: 'old-2' }])
    expect(store.softInvalidations).toEqual([])
  })

  it('duplicate：NOOP 不写库', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'old-3', text: '用户用 Postgres' })
    store.nextNeighbors = [{ id: 'old-3', cos: 0.99, record: existing }]
    const llm = new FakeLlm(['{"memories": ["用户用 Postgres"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(6), '{"index":0,"verdict":"duplicate"}'])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written).toEqual([])
    expect(outcome.dropped).toBe(1)
    expect(store.records.size).toBe(0)
  })

  it('与库中已有文本完全相同的候选走 NOOP（不消耗 M5 调用）', async () => {
    const store = new FakeStore()
    store.records.set('old-4', makeRecord({ id: 'old-4', text: '已存在的记忆文本' }))
    const llm = new FakeLlm(['{"memories": ["已存在的记忆文本"], "turn_mood": {"valence": 0, "arousal": 0}}']) // 只有抽取一次调用
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, [{ role: 'user', text: '随便' }])
    // P3-1：turnMood 透传与记忆写入无关——无候选幸存时也携带本轮情绪观测
    expect(outcome).toEqual({
      turn: 1, written: [], superseded: [], dropped: 0,
      turnMood: { valence: 0, arousal: 0 },
    })
    expect(llm.completions.length).toBe(1)
  })

  it('近邻余弦低于 contradictionMinCos 时不送判定（不发生 LLM 矛盾调用）', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'far-1', text: '无关记忆' })
    store.nextNeighbors = [{ id: 'far-1', cos: 0.5, record: existing }]
    const llm = new FakeLlm(['{"memories": ["候选"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(6)]) // 无第三次调用
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written.length).toBe(1)
    expect(llm.completions.length).toBe(2)
  })

  it('单候选失败不中断整轮：其余候选入库', async () => {
    const store = new FakeStore()
    const llm = new FakeLlm([
      '{"memories": ["候选甲", "候选乙"], "turn_mood": {"valence": 0, "arousal": 0}}',
      '坏掉的回复', // 甲的 M5 第一次失败
      '还是坏的', // 甲的 M5 重试失败 → 甲 dropped
      ...importances(7), // 乙的 M5
    ])
    const logs: string[] = []
    const pipeline = new WritePipeline({ llm, embedding, store, config, log: (m) => logs.push(m) })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written.length).toBe(1)
    expect(outcome.dropped).toBe(1)
    expect(logs.length).toBe(1)
  })

  it('signal 传播到 LLM 调用', async () => {
    const store = new FakeStore()
    const controller = new AbortController()
    const llm = new FakeLlm(['{"memories": ["候选"], "turn_mood": {"valence": 0, "arousal": 0}}'])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    await pipeline.processTurn('s1', 1, MESSAGES, controller.signal)
    expect(llm.completions[0]!.signal).toBe(controller.signal)
  })

  it('messageWindow 截断：抽取只看到窗口内消息', async () => {
    const store = new FakeStore()
    const llm = new FakeLlm(['{"memories": ["候选"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(5)])
    const pipeline = new WritePipeline({ llm, embedding, store, config: { ...config, write: { ...config.write, messageWindow: 2 } } })
    const longHistory = [
      { role: 'user' as const, text: 'msg1' }, { role: 'assistant' as const, text: 'msg2' },
      { role: 'user' as const, text: 'msg3' }, { role: 'assistant' as const, text: 'msg4' },
      { role: 'user' as const, text: 'msg5' },
    ]
    await pipeline.processTurn('s1', 1, longHistory)
    expect(llm.completions[0]!.user).toContain('msg4')
    expect(llm.completions[0]!.user).toContain('msg5')
    expect(llm.completions[0]!.user).not.toContain('msg3')
  })

  it('嵌入向量缺口：缺向量的候选丢弃，其余照常入库', async () => {
    const store = new FakeStore()
    const gappy: EmbeddingProvider = {
      dimensions: 8,
      // 只回填第一个文本的向量（模拟服务端缺口）
      async embed(texts: readonly string[]) {
        return texts.slice(0, 1).map((_, i) => {
          const v = new Array(8).fill(0)
          v[i] = 1
          return v
        })
      },
    }
    const llm = new FakeLlm(['{"memories": ["候选甲", "候选乙"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(7)])
    const pipeline = new WritePipeline({ llm, embedding: gappy, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written.length).toBe(1)
    expect(outcome.dropped).toBe(1)
    expect(store.records.size).toBe(1)
  })

  it('signal 中止后单候选错误原样上抛（不被吞成 dropped）', async () => {
    const store = new FakeStore()
    const controller = new AbortController()
    const abortingEmbed: EmbeddingProvider = {
      dimensions: 8,
      async embed(texts: readonly string[]) {
        controller.abort() // 嵌入阶段触发中止
        return texts.map(() => new Array<number>(8).fill(0))
      },
    }
    // 抽取后队列耗尽：甲的 M5 两次调用均抛 → completeJson 抛 → catch 分支按 aborted 上抛
    const llm = new FakeLlm(['{"memories": ["候选甲", "候选乙"], "turn_mood": {"valence": 0, "arousal": 0}}'])
    const pipeline = new WritePipeline({ llm, embedding: abortingEmbed, store, config })
    await expect(pipeline.processTurn('s1', 1, MESSAGES, controller.signal)).rejects.toThrow()
    expect(store.records.size).toBe(0)
  })

  it('同轮先写入的候选对后续候选的矛盾判定可见（经 store.similar 查已入库记录）', async () => {
    const store = new FakeStore()
    // 按关键词聚类的嵌入：两条候选含同一关键词 → 向量相同 → cos=1
    const clustered: EmbeddingProvider = {
      dimensions: 8,
      async embed(texts: readonly string[]) {
        return texts.map((text) => {
          const v = new Array<number>(8).fill(0)
          v[text.includes('Postgres') ? 0 : 1] = 1
          return v
        })
      },
    }
    const llm = new FakeLlm([
      '{"memories": ["用户使用 Postgres", "用户使用 Postgres 作为主数据库"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(7), // 甲 M5（此时库空，无矛盾调用）
      ...importances(7), // 乙 M5
      '{"index":0,"verdict":"duplicate"}', // 乙对甲（同轮先入库）判定为重复
    ])
    const pipeline = new WritePipeline({ llm, embedding: clustered, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written.length).toBe(1)
    expect(outcome.dropped).toBe(1)
    expect(store.records.size).toBe(1)
  })
})

describe('WritePipeline.processTurn 矛盾判定完整版（§2.5，P2）', () => {
  it('contradicts + temporal=disjoint：两条都真（不同时期），不软失效、不建链、照常入库', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'old-1', text: '用户 2024 年默认数据库是 MySQL' })
    store.nextNeighbors = [{ id: 'old-1', cos: 0.93, record: existing }]
    const llm = new FakeLlm([
      '{"memories": ["用户现在默认数据库是 PostgreSQL"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(7),
      '{"index":0,"verdict":"contradicts","temporal":"disjoint"}',
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config, now: () => 2000 })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)

    expect(store.softInvalidations).toEqual([])
    expect(store.nodeInvalidations).toEqual([])
    expect(outcome.superseded).toEqual([])
    expect(outcome.written.length).toBe(1)
    expect(outcome.written[0]!.links).toEqual([]) // 不建链
  })

  it('contradicts + temporal 缺省：按 overlapping 处理（P0 语义保持），软失效旧记录', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'old-1', text: '用户默认数据库是 MySQL' })
    store.nextNeighbors = [{ id: 'old-1', cos: 0.93, record: existing }]
    const llm = new FakeLlm([
      '{"memories": ["用户把默认数据库换成了 PostgreSQL"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(7),
      '{"index":0,"verdict":"contradicts"}', // 无 temporal 字段
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(store.softInvalidations).toEqual([{ id: 'old-1', by: outcome.written[0]!.id }])
  })

  it('contradicts 指向语义节点 gist：节点软失效、新记录建 supersedes 链', async () => {
    const store = new FakeStore()
    const node = makeNode({ id: 'node-1', gist: '用户偏好用 PostgreSQL 作为生产数据库' })
    store.nodes.set('node-1', node)
    store.nextNodeNeighbors = [{ id: 'node-1', cos: 0.95, node }]
    const llm = new FakeLlm([
      '{"memories": ["用户已把生产数据库迁移到 MySQL"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(8),
      '{"index":0,"verdict":"contradicts","temporal":"overlapping"}',
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config, now: () => 3000 })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)

    expect(store.nodeInvalidations).toEqual([{ id: 'node-1', by: outcome.written[0]!.id }])
    expect(store.softInvalidations).toEqual([]) // 记录侧不动
    expect(outcome.superseded).toEqual([{ id: 'node-1', by: outcome.written[0]!.id }])
    expect(outcome.written[0]!.links).toEqual([{ kind: 'supersedes', target: 'node-1' }])
    // 节点退出检索宇宙（FakeStore.softInvalidateNode 已置 invalid_at）
    expect(store.nodes.get('node-1')!.time_meta.invalid_at).toBe(1)
  })

  it('近邻送判顺序：记录在前、节点在后（index 跨表寻址确定性）', async () => {
    const store = new FakeStore()
    const existing = makeRecord({ id: 'rec-1', text: '用户在维护数据库迁移工具' })
    const node = makeNode({ id: 'node-1', gist: '用户重视数据完整性' })
    store.nextNeighbors = [{ id: 'rec-1', cos: 0.9, record: existing }]
    store.nextNodeNeighbors = [{ id: 'node-1', cos: 0.95, node }]
    const llm = new FakeLlm([
      '{"memories": ["用户为迁移工具补了数据校验"], "turn_mood": {"valence": 0, "arousal": 0}}',
      ...importances(6),
      '{"index":1,"verdict":"complements"}', // index=1 → 第二项 = 节点
    ])
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    const listing = llm.completions[2]!.user // 第三次调用 = 矛盾判定
    expect(listing.split('EXISTING memories:')[1]!.startsWith('\n0. 用户在维护数据库迁移工具')).toBe(true)
    expect(listing).toContain('1. 用户重视数据完整性')
    // P3-2：主判定目标建 complements 边；另一近邻 rec-1（cos 0.9 ≥ synonymCos 0.8）顺带建同义 related 边
    expect(outcome.written[0]!.links).toEqual([
      { kind: 'complements', target: 'node-1' },
      { kind: 'related', target: 'rec-1' },
    ])
  })

  it('节点 gist 低于 contradictionMinCos 不送判（安全网阈值对节点同样生效）', async () => {
    const store = new FakeStore()
    const node = makeNode({ id: 'node-1', gist: '无关节点' })
    store.nextNodeNeighbors = [{ id: 'node-1', cos: 0.5, node }]
    store.nextNeighbors = []
    const llm = new FakeLlm(['{"memories": ["候选"], "turn_mood": {"valence": 0, "arousal": 0}}', ...importances(6)]) // 无第三次调用
    const pipeline = new WritePipeline({ llm, embedding, store, config })
    const outcome = await pipeline.processTurn('s1', 1, MESSAGES)
    expect(outcome.written.length).toBe(1)
    expect(llm.completions.length).toBe(2)
  })
})

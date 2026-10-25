/**
 * 支柱一 S 完整动态（P2，src/scoring.ts）的行为测试。
 *
 * 两层覆盖：
 *  1. 纯函数：decayFactor（e^(−Δt/τ)、幂等）、arousalGate / useGain / nodeUseGain
 *     （§3.5 乘法门与 cue 下夹）、applyUseEvent / applySuppressEvent（惰性衰减
 *     追补 + 事件增量 + tier 跃迁标记 + counters 全套）、applyNodeUseEvent /
 *     applyNodeSuppressEvent / applyDecay / applyNodeDecay；
 *  2. ScoreKeeper.recordRetrieval 端到端（FakeStore + FakeLlm）：情景/语义两层
 *     的使用与抑制事件、层级跃迁触发 M5 重判（EMA）、无跃迁不重判、软失效
 *     记录跳过、重判失败保留旧值不抛。
 */
import { describe, expect, it, vi } from 'vitest'
import {
  applyDecay,
  applyNodeDecay,
  applyNodeSuppressEvent,
  applyNodeUseEvent,
  applySuppressEvent,
  applyUseEvent,
  arousalGate,
  decayFactor,
  nodeUseGain,
  ScoreKeeper,
  useGain,
  zeroCounters,
  type ScoreStoreLike,
} from '../src/scoring.js'
import type { Config } from '../src/config.js'
import type { InternalLlm, MemoryRecord, RetrievableItem, RetrievalResult, ScoredCandidate, SemanticNode } from '../src/types.js'

// ---- 测试基建 ----

const DAY = 24 * 3600 * 1000

const BASE_CONFIG: Config = {
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: false },
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
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
  emotion: { workspaces: [], alpha: 0.15, tauHours: 4, disclose: false },
  graph: { synonymCos: 0.8, maxDegree: 8 },
}

function makeRecord(over: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id: over.id ?? 'm1',
    text: over.text ?? '用户偏好简洁的回复',
    embedding: over.embedding ?? [1, 0, 0, 0, 0, 0, 0, 0],
    imp: 5, valence: 0, arousal: 0,
    epi: 0, red: 0, r_m: 0, tier: 'warm',
    con_route: { kind: 'pending' }, links: [], S: 5,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: 1_000, valid_at: 1_000 },
    flags: {},
    ...over,
  }
}

function makeNode(over: Partial<SemanticNode> = {}): SemanticNode {
  return {
    id: over.id ?? 'n1',
    gist: over.gist ?? '用户重视已上线服务的质量',
    embedding: over.embedding ?? [1, 0, 0, 0, 0, 0, 0, 0],
    epi: 0.4, links: [], member_refs: [],
    time_meta: { created_at: 1_000, valid_at: 1_000 },
    S: 5, r_m: 0.3,
    ...over,
  }
}

/** 满足 ScoreStoreLike 的假存储。 */
class FakeStore implements ScoreStoreLike {
  readonly records = new Map<string, MemoryRecord>()
  readonly nodes = new Map<string, SemanticNode>()
  get(id: string): MemoryRecord | undefined { return this.records.get(id) }
  getNode(id: string): SemanticNode | undefined { return this.nodes.get(id) }
  async updateRecord(id: string, fn: (r: MemoryRecord) => MemoryRecord): Promise<MemoryRecord> {
    const record = this.records.get(id)
    if (record === undefined) throw new Error(`memory not found: ${id}`)
    const next = fn(record)
    this.records.set(id, next)
    return next
  }
  async putNode(node: SemanticNode): Promise<void> { this.nodes.set(node.id, node) }
}

/** 按调用序回放预置回复的假 LLM；耗尽即抛错（兼作"不应再被调用"的哨兵）。 */
class FakeLlm implements InternalLlm {
  readonly calls: Array<{ system: string; user: string }> = []
  private readonly queue: string[]
  constructor(replies: readonly string[] = []) { this.queue = [...replies] }
  async complete(system: string, user: string, _signal?: AbortSignal): Promise<string> {
    this.calls.push({ system, user })
    const reply = this.queue.shift()
    if (reply === undefined) throw new Error('FakeLlm 无更多预置回复')
    return reply
  }
}

/** 手工构造一个带单命中/单落选的检索结果。 */
function makeRetrieval(top: ScoredCandidate[], suppressed: ScoredCandidate[] = []): RetrievalResult {
  return {
    query: 'q',
    expandedQuery: 'q',
    stages: { l0: [...top, ...suppressed], csls: [], rerank: [], mmr: [] },
    top,
    suppressed,
  }
}

function makeCandidate(id: string, layer: 'episodic' | 'semantic', cue?: number): ScoredCandidate {
  const item: RetrievableItem = {
    id,
    text: `text-${id}`,
    embedding: [1, 0, 0, 0, 0, 0, 0, 0],
    r_m: 0.2,
    tier: 'warm',
    createdAt: 1_000,
    layer,
  }
  return { id, cos: 0.9, cue, item }
}

// ---- 纯函数 ----

describe('decayFactor（§2.1 兜底衰减）', () => {
  it('τ=180 天：elapsed=τ → e^−1；elapsed=τ·ln2 → 0.5（半衰期约 125 天）', () => {
    expect(decayFactor(180 * DAY, 180)).toBeCloseTo(Math.exp(-1), 12)
    expect(decayFactor(180 * Math.LN2 * DAY, 180)).toBeCloseTo(0.5, 12)
  })
  it('elapsed ≤ 0（同刻/时钟回拨）→ 1（幂等）', () => {
    expect(decayFactor(0, 180)).toBe(1)
    expect(decayFactor(-1000, 180)).toBe(1)
  })
})

describe('arousalGate / useGain / nodeUseGain（§3.5 乘法门）', () => {
  it('arousalGate：0 → 0.5；1 → 1；0.5 → 0.75；越界夹取', () => {
    expect(arousalGate(0)).toBe(0.5)
    expect(arousalGate(1)).toBe(1)
    expect(arousalGate(0.5)).toBe(0.75)
    expect(arousalGate(2)).toBe(1)
    expect(arousalGate(-1)).toBe(0.5)
  })
  it('useGain = α·(1+cue)·g(arousal)：α=0.5、cue=0.6、arousal=0 → 0.4；arousal=1 → 0.8', () => {
    expect(useGain(0.6, 0, BASE_CONFIG.score)).toBeCloseTo(0.4, 12)
    expect(useGain(0.6, 1, BASE_CONFIG.score)).toBeCloseTo(0.8, 12)
  })
  it('cue < −1 时 (1+cue) 下夹 0：成功使用不产生负巩固', () => {
    expect(useGain(-2, 1, BASE_CONFIG.score)).toBe(0)
  })
  it('nodeUseGain 无 arousal 门：α·(1+cue)', () => {
    expect(nodeUseGain(0.6, BASE_CONFIG.score)).toBeCloseTo(0.8, 12)
  })
})

describe('applyUseEvent（使用事件，情景记录）', () => {
  it('无 elapsed：S += 增益；counters 全套更新；tier 不变', () => {
    const record = makeRecord({ S: 5, arousal: 0, tier: 'warm' })
    const { record: next, tierChanged } = applyUseEvent(record, 0.6, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(5 + 0.4, 12)
    expect(next.tier).toBe('warm')
    expect(tierChanged).toBe(false)
    expect(next.counters).toEqual({
      used_count: 1, suppressed_count: 0, last_used_at: 1_000, last_cue_quality: 0.6, last_decay_at: 1_000,
    })
  })
  it('惰性衰减追补：锚点 180 天前 → S·e^−1 再加增益', () => {
    const record = makeRecord({ S: 5 })
    const now = 1_000 + 180 * DAY
    const { record: next } = applyUseEvent(record, 0, now, BASE_CONFIG)
    expect(next.S).toBeCloseTo(5 * Math.exp(-1) + 0.25, 12) // cue=0、arousal=0 → 增益 0.5·1·0.5
    expect(next.counters.last_decay_at).toBe(now)
  })
  it('cue 缺省回退 last_cue_quality，再缺省回退 0', () => {
    const withHistory = makeRecord({ counters: { used_count: 2, suppressed_count: 0, last_cue_quality: 0.2 } })
    const { record: next } = applyUseEvent(withHistory, undefined, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(5 + 0.5 * 1.2 * 0.5, 12)
    const fresh = applyUseEvent(makeRecord(), undefined, 1_000, BASE_CONFIG)
    expect(fresh.record.S).toBeCloseTo(5 + 0.25, 12)
  })
  it('跃迁：S=7.9 + 0.4 → 8.3 hot，tierChanged=true', () => {
    const record = makeRecord({ S: 7.9, tier: 'warm' })
    const { record: next, tierChanged } = applyUseEvent(record, 0.6, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(8.3, 12)
    expect(next.tier).toBe('hot')
    expect(tierChanged).toBe(true)
  })
})

describe('applySuppressEvent（抑制事件，情景记录）', () => {
  it('无 elapsed：S −= ε；suppressed_count+1；锚点前移', () => {
    const record = makeRecord({ S: 5 })
    const { record: next, tierChanged } = applySuppressEvent(record, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(4.95, 12)
    expect(tierChanged).toBe(false)
    expect(next.counters.suppressed_count).toBe(1)
    expect(next.counters.last_decay_at).toBe(1_000)
    expect(next.counters.used_count).toBe(0)
  })
  it('跃迁：S=2.02 − 0.05 → 1.97 cold，tierChanged=true', () => {
    const record = makeRecord({ S: 2.02, tier: 'warm' })
    const { record: next, tierChanged } = applySuppressEvent(record, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(1.97, 12)
    expect(next.tier).toBe('cold')
    expect(tierChanged).toBe(true)
  })
})

describe('节点侧事件（applyNodeUseEvent / applyNodeSuppressEvent）', () => {
  it('无 counters 的旧节点：按全 0 容忍初始化并使用', () => {
    const node = makeNode({ S: 4 })
    const next = applyNodeUseEvent(node, 0.2, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(4 + 0.5 * 1.2, 12) // 无 arousal 门
    expect(next.counters).toEqual({
      used_count: 1, suppressed_count: 0, last_used_at: 1_000, last_cue_quality: 0.2, last_decay_at: 1_000,
    })
  })
  it('抑制：S −= ε、counters 延续', () => {
    const node = makeNode({ S: 4, counters: { used_count: 3, suppressed_count: 1 } })
    const next = applyNodeSuppressEvent(node, 1_000, BASE_CONFIG)
    expect(next.S).toBeCloseTo(3.95, 12)
    expect(next.counters).toEqual({ used_count: 3, suppressed_count: 2, last_decay_at: 1_000 })
  })
})

describe('applyDecay / applyNodeDecay（睡眠周期批处理）', () => {
  it('记录：S·e^(−Δt/τ)、tier 重算、锚点前移', () => {
    const record = makeRecord({ S: 5, tier: 'warm' })
    const next = applyDecay(record, 1_000 + 180 * DAY, BASE_CONFIG)
    expect(next.S).toBeCloseTo(5 * Math.exp(-1), 12)
    expect(next.tier).toBe('cold') // 1.84 < 2
    expect(next.counters.last_decay_at).toBe(1_000 + 180 * DAY)
    expect(next.counters.used_count).toBe(0) // 不动事件计数
  })
  it('节点：同规；counters 缺省初始化', () => {
    const node = makeNode({ S: 4 })
    const next = applyNodeDecay(node, 1_000 + 180 * DAY, BASE_CONFIG)
    expect(next.S).toBeCloseTo(4 * Math.exp(-1), 12)
    expect(next.counters).toEqual({ ...zeroCounters(), last_decay_at: 1_000 + 180 * DAY })
  })
  it('锚点已前移后重复衰减是幂等的（elapsed=0 → 因子 1）', () => {
    const once = applyDecay(makeRecord({ S: 5 }), 2_000, BASE_CONFIG)
    const twice = applyDecay(once, 2_000, BASE_CONFIG)
    expect(twice.S).toBe(once.S)
  })
})

// ---- ScoreKeeper.recordRetrieval ----

describe('ScoreKeeper.recordRetrieval（两通道共用的记账入口）', () => {
  it('情景层 top：使用事件入账（S/counters/tier）', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', S: 5, arousal: 0 }))
    const llm = new FakeLlm()
    const keeper = new ScoreKeeper({ llm, store, config: BASE_CONFIG, now: () => 1_000 })

    await keeper.recordRetrieval(makeRetrieval([makeCandidate('m1', 'episodic', 0.6)]))

    const record = store.records.get('m1')!
    expect(record.S).toBeCloseTo(5.4, 12)
    expect(record.counters.used_count).toBe(1)
    expect(llm.calls).toHaveLength(0) // warm→warm 无跃迁，不重判
  })

  it('语义层 top：节点使用事件入账（无 arousal 门、无重判）', async () => {
    const store = new FakeStore()
    store.nodes.set('n1', makeNode({ id: 'n1', S: 4 }))
    const llm = new FakeLlm()
    const keeper = new ScoreKeeper({ llm, store, config: BASE_CONFIG, now: () => 1_000 })

    await keeper.recordRetrieval(makeRetrieval([makeCandidate('n1', 'semantic', 0.2)]))

    const node = store.nodes.get('n1')!
    expect(node.S).toBeCloseTo(4 + 0.6, 12)
    expect(node.counters?.used_count).toBe(1)
    expect(llm.calls).toHaveLength(0)
  })

  it('suppressed：情景层 ε 惩罚、语义层同规', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', S: 5 }))
    store.nodes.set('n1', makeNode({ id: 'n1', S: 4 }))
    const keeper = new ScoreKeeper({ llm: new FakeLlm(), store, config: BASE_CONFIG, now: () => 1_000 })

    await keeper.recordRetrieval(makeRetrieval([], [
      makeCandidate('m1', 'episodic'),
      makeCandidate('n1', 'semantic'),
    ]))

    expect(store.records.get('m1')!.S).toBeCloseTo(4.95, 12)
    expect(store.records.get('m1')!.counters.suppressed_count).toBe(1)
    expect(store.nodes.get('n1')!.S).toBeCloseTo(3.95, 12)
    expect(store.nodes.get('n1')!.counters?.suppressed_count).toBe(1)
  })

  it('层级跃迁触发 M5 重判：新判分按 EMA 混入 imp/valence/arousal', async () => {
    const store = new FakeStore()
    // S=7.9 + gain(cue=0.6, arousal=0)=0.4 → 8.3 hot：warm→hot 跃迁
    store.records.set('m1', makeRecord({ id: 'm1', S: 7.9, imp: 5, valence: 0, arousal: 0 }))
    const llm = new FakeLlm(['{"importance":9,"valence":-1,"arousal":1,"reasons":["decision"]}'])
    const keeper = new ScoreKeeper({ llm, store, config: BASE_CONFIG, now: () => 1_000 })

    await keeper.recordRetrieval(makeRetrieval([makeCandidate('m1', 'episodic', 0.6)]))

    const record = store.records.get('m1')!
    expect(record.tier).toBe('hot')
    expect(record.S).toBeCloseTo(8.3, 12)
    // EMA β=0.5：imp=(5+9)/2、valence=(0−1)/2、arousal=(0+1)/2
    expect(record.imp).toBeCloseTo(7, 12)
    expect(record.valence).toBeCloseTo(-0.5, 12)
    expect(record.arousal).toBeCloseTo(0.5, 12)
    expect(llm.calls).toHaveLength(1)
    expect(llm.calls[0]!.user).toContain('用户偏好简洁的回复') // 重判对象是记忆文本
  })

  it('抑制事件引起的跃迁同样重判', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', S: 2.02, imp: 5 }))
    const llm = new FakeLlm(['{"importance":3,"valence":0,"arousal":0,"reasons":[]}'])
    const keeper = new ScoreKeeper({ llm, store, config: BASE_CONFIG, now: () => 1_000 })

    await keeper.recordRetrieval(makeRetrieval([], [makeCandidate('m1', 'episodic')]))

    expect(store.records.get('m1')!.tier).toBe('cold')
    expect(store.records.get('m1')!.imp).toBeCloseTo(4, 12) // (5+3)/2
  })

  it('重判失败：保留旧判分、记日志、不上抛', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', S: 7.9, imp: 5 }))
    const llm = new FakeLlm() // 耗尽即抛
    const logs: string[] = []
    const keeper = new ScoreKeeper({ llm, store, config: BASE_CONFIG, now: () => 1_000, log: (m) => logs.push(m) })

    await keeper.recordRetrieval(makeRetrieval([makeCandidate('m1', 'episodic', 0.6)]))

    expect(store.records.get('m1')!.imp).toBe(5) // 旧值保留
    expect(store.records.get('m1')!.S).toBeCloseTo(8.3, 12) // S 记账不受重判失败影响
    expect(logs.length).toBe(1)
  })

  it('软失效记录与缺失 id 静默跳过（检索与记账之间可能被推翻）', async () => {
    const store = new FakeStore()
    store.records.set('dead', makeRecord({ id: 'dead', S: 5, time_meta: { created_at: 1_000, valid_at: 1_000, invalid_at: 1_500 } }))
    const logs: string[] = []
    const keeper = new ScoreKeeper({ llm: new FakeLlm(), store, config: BASE_CONFIG, now: () => 2_000, log: (m) => logs.push(m) })

    await keeper.recordRetrieval(makeRetrieval(
      [makeCandidate('dead', 'episodic'), makeCandidate('ghost', 'episodic')],
      [makeCandidate('nobody', 'semantic')],
    ))

    expect(store.records.get('dead')!.S).toBe(5) // 未动
    expect(store.records.get('dead')!.counters.used_count).toBe(0)
    // ghost/nobody：情景层经 updateRecord 抛错吞掉、语义层 getNode undefined 直接返回
    expect(logs.length).toBe(1)
  })
})

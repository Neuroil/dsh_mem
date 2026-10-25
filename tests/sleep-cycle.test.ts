/**
 * 离线睡眠巩固周期（P1，src/sleep-cycle.ts）的行为测试。
 *
 * 三层覆盖：
 *  1. 纯函数：buildClusters（连通分量 + 平均 pairwise 余弦闸门 + 不可压缩过滤）、
 *     conAgainstBaseline（gist 对语义网络峰值余弦的 z 分数、平局 id 字典序决胜）、
 *     gistEpiScore（M6 完整 epi = σ(z 实体密度 + z 时间密度 − z 探针相关度) 的方向性）；
 *  2. qaF1 的 token 切分（大小写/标点不敏感、Han 单字、多重集截断）与 M6_PROBES 常量；
 *  3. SleepCycle.run 端到端（FakeStore 实现 SleepStoreLike + 按序回放的 FakeLlm +
 *     按文本映射向量的嵌入 stub）：空网引导建首节点、QA/epi 闸门拒收、M4 挂入与
 *     LLM 兼容性复核、con 中性留层、建新节点 imp 门控、signal 中止上抛、
 *     池不足仍刷新统计、单簇失败不中断整轮。
 *
 * 约定：数值断言用 toBeCloseTo(x, 10)；向量一律 8 维，前两维复用 2D 单位向量
 * plane(c) = [c, √(1−c²), 0…]，使 cos(v, e1) = c 可手工精确控制。
 */
import { describe, expect, it, vi } from 'vitest'
import {
  M6_PROBES,
  SleepCycle,
  buildClusters,
  conAgainstBaseline,
  gistEpiScore,
  type SleepStoreLike,
} from '../src/sleep-cycle.js'
import { qaF1 } from '../src/metrics.js'
import type { Config } from '../src/config.js'
import type { EmbeddingProvider, InternalLlm, MemoryRecord, SemanticNode } from '../src/types.js'

// ---- 测试基建 ----

const BASE_CONFIG: Config = {
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: false },
  embedding: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000, dimensions: 8, batchSize: 8 },
  reranker: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 5_000 },
  llm: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 2048 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: { l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6, cslsK: 10, contextTurns: 3, coldCueBoost: 0.1, cueThreshold: 0.3, pprSeedK: 8, pprDamping: 0.5, pprWeight: 0.2 },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: {
    enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3,
    qaF1Threshold: 0.8, epiThreshold: 0.5, conWindow: 1000, conAttachSigma: 0.5,
    conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5,
  },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
  emotion: { workspaces: [], alpha: 0.15, tauHours: 4, disclose: false },
  graph: { synonymCos: 0.8, maxDegree: 8 },
  inspire: { counterProbability: 0.25 },
}

/** 覆写 sleep 段超参（其余段沿用 BASE_CONFIG）。 */
function withSleep(over: Partial<Config['sleep']>): Config {
  return { ...BASE_CONFIG, sleep: { ...BASE_CONFIG.sleep, ...over } }
}

/** 8 维正交基向量 e_i。 */
function unit(i: number): number[] {
  const v = new Array<number>(8).fill(0)
  v[i] = 1
  return v
}

/** 前两维平面上的单位向量 [c, √(1−c²), 0…]；cos(v, e1) = c，便于手工设定余弦。 */
function plane(c: number): number[] {
  const s = Math.sqrt(Math.max(0, 1 - c * c))
  return [c, s, 0, 0, 0, 0, 0, 0]
}

/** 与 e1 夹角 45° 的单位向量（前两维各 1/√2）。 */
const DIAGONAL: readonly number[] = [Math.SQRT1_2, Math.SQRT1_2, 0, 0, 0, 0, 0, 0]

function makeRecord(over: Partial<MemoryRecord> = {}): MemoryRecord {
  const now = 1_000
  return {
    id: over.id ?? 'm1',
    text: over.text ?? '用户偏好简洁的回复',
    embedding: over.embedding ?? unit(0),
    imp: 5, valence: 0, arousal: 0,
    epi: 0.5, red: 0, r_m: 0.3, tier: 'warm',
    con_route: { kind: 'pending' }, links: [], S: 5,
    counters: { used_count: 0, suppressed_count: 0 },
    time_meta: { created_at: now, valid_at: now },
    flags: {},
    ...over,
  }
}

function makeNode(over: Partial<SemanticNode> = {}): SemanticNode {
  const now = 900
  return {
    id: over.id ?? 'n1',
    gist: over.gist ?? '已有节点的摘要',
    embedding: over.embedding ?? unit(0),
    epi: 0.4, links: [], member_refs: over.member_refs ?? ['x0'],
    time_meta: { created_at: now, valid_at: now },
    S: 4, r_m: 0.5,
    ...over,
  }
}

/** 满足 SleepStoreLike 的假存储：updateRecord 应用 fn（缺记录抛错），刷新只记录调用。 */
class FakeStore implements SleepStoreLike {
  readonly records = new Map<string, MemoryRecord>()
  readonly nodes = new Map<string, SemanticNode>()
  readonly refreshed: Array<{ id: string; k: number }> = []
  readonly nodeRefreshed: Array<{ id: string; k: number }> = []

  validMemories(): readonly MemoryRecord[] {
    return [...this.records.values()].filter((r) => r.time_meta.invalid_at === undefined)
  }
  validNodes(): readonly SemanticNode[] {
    return [...this.nodes.values()].filter((n) => n.time_meta.invalid_at === undefined)
  }
  async updateRecord(id: string, fn: (r: MemoryRecord) => MemoryRecord): Promise<MemoryRecord> {
    const record = this.records.get(id)
    if (record === undefined) throw new Error(`记录不存在: ${id}`)
    const next = fn(record)
    this.records.set(id, next)
    return next
  }
  async putNode(node: SemanticNode): Promise<void> {
    this.nodes.set(node.id, node)
  }
  async refreshNeighborhood(id: string, k: number): Promise<{ r_m: number; red: number }> {
    this.refreshed.push({ id, k })
    return { r_m: 0.42, red: 0.3 }
  }
  async refreshNodeNeighborhood(id: string, k: number): Promise<number> {
    this.nodeRefreshed.push({ id, k })
    return 0.7
  }
}

/** 按调用序回放预置回复的假 LLM；耗尽即抛错（兼作“不应再被调用”的哨兵）。 */
class FakeLlm implements InternalLlm {
  readonly calls: Array<{ system: string; user: string }> = []
  private readonly queue: string[]
  constructor(replies: readonly string[]) { this.queue = [...replies] }
  async complete(system: string, user: string, _signal?: AbortSignal): Promise<string> {
    this.calls.push({ system, user })
    const reply = this.queue.shift()
    if (reply === undefined) throw new Error('FakeLlm 无更多预置回复')
    return reply
  }
}

/** M6 探针的固定向量：正交基循环（20 条 → 每维 2~3 条）。 */
const PROBE_MAP: Readonly<Record<string, readonly number[]>> = Object.fromEntries(
  M6_PROBES.map((probe, i) => [probe, unit(i % 8)]),
)

/** 按文本映射向量的嵌入 stub；gist 文本由 FakeLlm 脚本预知，未映射文本落到 e1。 */
function makeEmbedding(extra: Readonly<Record<string, readonly number[]>> = {}) {
  const map: Readonly<Record<string, readonly number[]>> = { ...PROBE_MAP, ...extra }
  const embed = vi.fn(async (texts: readonly string[], _signal?: AbortSignal): Promise<readonly number[][]> =>
    texts.map((text) => map[text] ?? unit(0)),
  )
  const provider: EmbeddingProvider = { dimensions: 8, embed }
  return { provider, embed }
}

const gistReply = (gist: string): string => JSON.stringify({ gist })

/** 一次“QA 全对”的脚本：QA 对生成 + 每对一次 gist 作答（答案与参考一致，F1=1）。 */
function qaPerfect(pairs: ReadonlyArray<{ q: string; a: string }>): string[] {
  return [
    JSON.stringify(pairs.map(({ q, a }) => ({ question: q, answer: a }))),
    ...pairs.map(({ a }) => JSON.stringify({ answer: a })),
  ]
}

// ---- buildClusters（第 1 步聚类，纯函数） ----

describe('buildClusters', () => {
  const sleep = BASE_CONFIG.sleep

  it('三条两两余弦 ≥ 阈值的记录成一簇：成员按 id 升序，平均 pairwise 余弦与手工 2D 计算一致', () => {
    // r1 = e1；r2 = [4,1]/√17；r3 = [2,1]/√5（均单位向量）
    // cos(r1,r2) = 4/√17，cos(r1,r3) = 2/√5，cos(r2,r3) = (4·2+1·1)/√85 = 9/√85
    const pool = [
      makeRecord({ id: 'r3', embedding: plane(2 / Math.sqrt(5)) }),
      makeRecord({ id: 'r1', embedding: plane(1) }),
      makeRecord({ id: 'r2', embedding: plane(4 / Math.sqrt(17)) }),
    ]
    const clusters = buildClusters(pool, sleep)
    expect(clusters).toHaveLength(1)
    expect(clusters[0]!.members.map((m) => m.id)).toEqual(['r1', 'r2', 'r3'])
    expect(clusters[0]!.meanPairwiseCos).toBeCloseTo(
      (4 / Math.sqrt(17) + 2 / Math.sqrt(5) + 9 / Math.sqrt(85)) / 3, 10,
    )
  })

  it('链式传导被平均值闸门拦截：A-B、B-C 达阈值而 A-C 很低 → 同一连通分量但整体不成簇', () => {
    // θ = arccos(0.85)：cos(A,B) = cos(B,C) = 0.85 连边；cos(A,C) = cos2θ ≈ 0.445 不连边但经 B 同分量
    // 平均 pairwise = (0.85 + 0.85 + 0.445)/3 = 0.715 < 0.75 → 拒收
    const theta = Math.acos(0.85)
    const pool = [
      makeRecord({ id: 'c1', embedding: plane(1) }),
      makeRecord({ id: 'c2', embedding: plane(0.85) }),
      makeRecord({ id: 'c3', embedding: plane(Math.cos(2 * theta)) }),
    ]
    expect(buildClusters(pool, sleep)).toEqual([])
  })

  it('默认 clusterMinSize=3 下，两条近重复（cos=1）不成簇', () => {
    const pool = [makeRecord({ id: 'd1' }), makeRecord({ id: 'd2' })] // 默认向量均为 e1
    expect(buildClusters(pool, sleep)).toEqual([])
  })

  it('两个互不相连的 3-团 → 2 簇，按首成员 id 排序，簇内成员升序', () => {
    const pool = [
      makeRecord({ id: 'b2', embedding: unit(1) }),
      makeRecord({ id: 'a3', embedding: unit(0) }),
      makeRecord({ id: 'b1', embedding: unit(1) }),
      makeRecord({ id: 'a1', embedding: unit(0) }),
      makeRecord({ id: 'b3', embedding: unit(1) }),
      makeRecord({ id: 'a2', embedding: unit(0) }),
    ]
    const clusters = buildClusters(pool, sleep)
    expect(clusters.map((c) => c.members.map((m) => m.id))).toEqual([
      ['a1', 'a2', 'a3'],
      ['b1', 'b2', 'b3'],
    ])
    expect(clusters[0]!.meanPairwiseCos).toBeCloseTo(1, 10)
    expect(clusters[1]!.meanPairwiseCos).toBeCloseTo(1, 10)
  })

  it('全员 incompressible 的簇被跳过；含一名未标记成员的簇保留', () => {
    const pool = [
      makeRecord({ id: 'a1', embedding: unit(0), flags: { incompressible: true } }),
      makeRecord({ id: 'a2', embedding: unit(0), flags: { incompressible: true } }),
      makeRecord({ id: 'a3', embedding: unit(0), flags: { incompressible: true } }),
      makeRecord({ id: 'b1', embedding: unit(1), flags: { incompressible: true } }),
      makeRecord({ id: 'b2', embedding: unit(1), flags: { incompressible: true } }),
      makeRecord({ id: 'b3', embedding: unit(1) }),
    ]
    const clusters = buildClusters(pool, sleep)
    expect(clusters.map((c) => c.members.map((m) => m.id))).toEqual([['b1', 'b2', 'b3']])
  })
})

// ---- conAgainstBaseline（M4 con，纯函数） ----

describe('conAgainstBaseline', () => {
  it('手算 z：基线峰值 0.5/0.7（μ=0.6 σ=0.1），gist 峰值 0.8 → con=2；平局取 id 字典序小者', () => {
    // 两个同向节点造成峰值平局；基线两条记录对节点峰值余弦恰为 0.5 / 0.7
    const nodes = [
      makeNode({ id: 'nb', embedding: unit(0) }),
      makeNode({ id: 'na', embedding: unit(0) }),
    ]
    const baseline = [
      makeRecord({ id: 'base-1', embedding: plane(0.5) }),
      makeRecord({ id: 'base-2', embedding: plane(0.7) }),
    ]
    const { con, attachTo } = conAgainstBaseline(plane(0.8), nodes, baseline, 1000)
    expect(con).toBeCloseTo(2, 10)
    expect(attachTo.id).toBe('na')
  })

  it('基线为空 → con=0（中性，不挂入也不建新节点）', () => {
    const nodes = [makeNode({ id: 'n1', embedding: unit(0) })]
    const { con, attachTo } = conAgainstBaseline(plane(0.8), nodes, [], 1000)
    expect(con).toBeCloseTo(0, 10)
    expect(attachTo.id).toBe('n1')
  })
})

// ---- gistEpiScore（M6 完整 epi，纯函数） ----

describe('gistEpiScore', () => {
  // 探针取正交基前两维：G(v) = (v0 + v1)/2，可用池/gist 向量精确控制
  const probes: readonly (readonly number[])[] = [unit(0), unit(1)]

  it('方向性：池词法密集且 G 低、gist 纯抽象且 G 高 → epi < 0.5', () => {
    const pool = [
      // 词法重（日期/数字实体），向量 e3 → G = 0
      makeRecord({ id: 'p1', text: '2026年10月24日和张三开会', embedding: unit(2) }),
      // 词法重，G = 0.1
      makeRecord({ id: 'p2', text: '2027年3月5日和李四通话', embedding: [0.2, 0, Math.sqrt(0.96), 0, 0, 0, 0, 0] }),
      // 词法中等，G = 0.2
      makeRecord({ id: 'p3', text: '服务器部署在 AWS 东京区', embedding: [0, 0.4, Math.sqrt(0.84), 0, 0, 0, 0, 0] }),
    ]
    // gist 纯抽象（词法 0），向量与两探针均值同向 → G = 1/√2 ≈ 0.707（池分布高端）
    const epi = gistEpiScore('用户偏好简洁的回复', DIAGONAL, pool, probes)
    expect(epi).toBeLessThan(0.5)
  })

  it('反向构造：gist 词法密集且 G 低、池偏抽象且 G 高 → epi > 0.5', () => {
    const pool = [
      makeRecord({ id: 'q1', text: '用户喜欢简洁的回复', embedding: [0.6, 0, 0.8, 0, 0, 0, 0, 0] }), // G = 0.3
      makeRecord({ id: 'q2', text: '服务器部署在 AWS 东京区', embedding: [0.8, 0.2, Math.sqrt(0.32), 0, 0, 0, 0, 0] }), // G = 0.5
      makeRecord({ id: 'q3', text: '2026年10月24日和张三开会', embedding: DIAGONAL }), // G = 1/√2
    ]
    // gist 词法重且与探针正交（G = 0，池分布低端）
    const epi = gistEpiScore('2027年3月5日和李四通话', unit(2), pool, probes)
    expect(epi).toBeGreaterThan(0.5)
  })

  it('gist 特征恰等于池均值（单条同文同向量的池）→ 三个 z 全 0 → epi=0.5', () => {
    const gist = '用户偏好简洁的回复'
    const pool = [makeRecord({ id: 's1', text: gist, embedding: unit(3) })]
    expect(gistEpiScore(gist, unit(3), pool, probes)).toBeCloseTo(0.5, 10)
  })
})

// ---- qaF1 与 M6_PROBES ----

describe('qaF1', () => {
  it('完全一致 → 1', () => {
    expect(qaF1('部署延迟源于配置', '部署延迟源于配置')).toBeCloseTo(1, 10)
  })

  it('部分重叠手算 precision/recall：对称与不对称两种', () => {
    // ref [a,b,c] vs cand [a,b,d]：overlap=2，P=R=2/3 → F1=2/3
    expect(qaF1('a b c', 'a b d')).toBeCloseTo(2 / 3, 10)
    // ref [a,b,c,d] vs cand [a,b]：P=1、R=0.5 → F1=2/3
    expect(qaF1('a b c d', 'a b')).toBeCloseTo(2 / 3, 10)
  })

  it('大小写与标点不敏感', () => {
    expect(qaF1('Hello, World!', 'hello world')).toBeCloseTo(1, 10)
  })

  it('Han 字按单字 token：“部署延迟” vs “部署延时” 有部分重叠（3/4）', () => {
    // [部,署,延,迟] vs [部,署,延,时]：overlap=3，P=R=3/4
    expect(qaF1('部署延迟', '部署延时')).toBeCloseTo(0.75, 10)
  })

  it('空串 / 无 token → 0', () => {
    expect(qaF1('', '答案')).toBeCloseTo(0, 10)
    expect(qaF1('答案', '')).toBeCloseTo(0, 10)
    expect(qaF1('!!!', '???')).toBeCloseTo(0, 10) // 标点不产 token，双侧皆空
  })

  it('重复 token 多重集截断：“a a” vs “a” → 2/3（且对称）', () => {
    // ref [a,a] vs cand [a]：overlap=1，P=1、R=1/2 → F1=2/3
    expect(qaF1('a a', 'a')).toBeCloseTo(2 / 3, 10)
    expect(qaF1('a', 'a a')).toBeCloseTo(2 / 3, 10)
  })
})

describe('M6_PROBES', () => {
  it('固定 20 条且互不相同', () => {
    expect(M6_PROBES.length).toBe(20)
    expect(new Set(M6_PROBES).size).toBe(20)
  })
})

// ---- SleepCycle.run 端到端 ----

describe('SleepCycle.run', () => {
  it('引导路径：空网络无条件建首节点（new_node），节点与成员归档字段完整', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', text: '用户偏好简洁的回复', imp: 6, epi: 0.7 }))
    store.records.set('m2', makeRecord({ id: 'm2', text: '用户希望回答保持简短', imp: 6, epi: 0.8 }))
    store.records.set('m3', makeRecord({ id: 'm3', text: '用户喜欢直接的答案', imp: 6, epi: 0.9 }))
    const gistText = '用户偏好简洁直接的回复'
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([
        { q: '用户偏好怎样的回复？', a: '简洁' },
        { q: '回答应当保持多长？', a: '简短' },
        { q: '用户喜欢什么样的答案？', a: '直接' },
      ]),
    ])
    const { provider, embed } = makeEmbedding({ [gistText]: unit(0) })
    const cycle = new SleepCycle({
      llm, embedding: provider, store,
      config: withSleep({ epiThreshold: 0.99 }), // 抬高阈值强制通过 M6 闸门
      now: () => 4242, newId: () => 'node-1',
    })

    const outcome = await cycle.run()

    expect(outcome.clusters).toHaveLength(1)
    const c = outcome.clusters[0]!
    expect(c.members).toEqual(['m1', 'm2', 'm3'])
    expect(c.gist).toBe(gistText)
    expect(c.qaF1).toBeCloseTo(1, 10)
    expect(c.memberMeanEpi).toBeCloseTo(0.8, 10)
    expect(c.gistEpi).toBeCloseTo(0.5, 10) // 池/ gist 词法与探针相关度全同 → logit=0
    expect(c.route).toBe('new_node')
    expect(c.nodeId).toBe('node-1')

    const node = store.nodes.get('node-1')!
    expect(node.S).toBeCloseTo(6, 10) // S = 簇均 imp
    expect(node.member_refs).toEqual(['m1', 'm2', 'm3'])
    expect(node.time_meta).toEqual({ created_at: 4242, valid_at: 4242 }) // 注入的 now
    expect(node.embedding).toEqual(unit(0)) // 注入的 gist 向量
    expect(node.epi).toBeCloseTo(c.gistEpi, 10)
    expect(node.gist).toBe(gistText)
    expect(node.links).toEqual([])

    for (const id of ['m1', 'm2', 'm3']) {
      const r = store.records.get(id)!
      expect(r.flags).toEqual({ compressed_into: 'node-1', qa_passed: true })
      expect(r.con_route).toEqual({ kind: 'new_node', nodeId: 'node-1' })
      // P3-2 聚类建边：同簇成员互连 related（3 成员簇，每人 2 条，远低于度数上限）
      const mates = ['m1', 'm2', 'm3'].filter((m) => m !== id)
      expect(r.links.filter((l) => l.kind === 'related')).toEqual(
        mates.map((m) => ({ kind: 'related', target: m })),
      )
    }

    expect(outcome.refreshedRecords).toBe(0) // 全部被压缩，退出检索宇宙
    expect(outcome.refreshedNodes).toEqual(['node-1'])
    expect(store.refreshed).toEqual([])
    expect(store.nodeRefreshed).toEqual([{ id: 'node-1', k: 10 }]) // 第 7 步用 cslsK

    expect(llm.calls).toHaveLength(5) // 1 gist + 1 QA 生成 + 3 作答；引导路径无兼容性复核
    expect(llm.calls[0]!.user).toContain('用户偏好简洁的回复')
    expect(llm.calls[0]!.user).toContain('用户喜欢直接的答案')
    expect(embed.mock.calls).toHaveLength(2) // 探针批量 + gist 单条
    expect(embed.mock.calls[0]![0]).toHaveLength(20)
    expect(embed.mock.calls[1]![0]).toEqual([gistText])
  })

  it('QA 闸门失败（作答 unknown）→ qa_failed，全簇标 incompressible，不压缩不建节点', async () => {
    const store = new FakeStore()
    for (let i = 1; i <= 3; i++) store.records.set(`m${i}`, makeRecord({ id: `m${i}` }))
    const gistText = '用户偏好简洁直接的回复'
    const llm = new FakeLlm([
      gistReply(gistText),
      '[{"question":"用户偏好怎样的回复？","answer":"简洁"}]',
      '{"answer":"unknown"}', // F1=0
    ])
    const { provider } = makeEmbedding({ [gistText]: unit(0) })
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: BASE_CONFIG })

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('qa_failed')
    expect(c.qaF1).toBeCloseTo(0, 10)
    expect(c.nodeId).toBeUndefined()
    for (const id of ['m1', 'm2', 'm3']) {
      const r = store.records.get(id)!
      expect(r.flags).toEqual({ incompressible: true }) // 不带 compressed_into：记录保留
      expect(r.con_route).toEqual({ kind: 'pending' })
    }
    expect(store.nodes.size).toBe(0)
    expect(outcome.refreshedRecords).toBe(3) // 未压缩 → 仍在刷新范围
    expect(outcome.refreshedNodes).toEqual([])
  })

  it('generateQaPairs 返回空数组 → qa_failed 且 qaF1=0，无作答调用', async () => {
    const store = new FakeStore()
    for (let i = 1; i <= 3; i++) store.records.set(`m${i}`, makeRecord({ id: `m${i}` }))
    const gistText = '用户偏好简洁直接的回复'
    const llm = new FakeLlm([gistReply(gistText), '[]'])
    const { provider } = makeEmbedding({ [gistText]: unit(0) })
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: BASE_CONFIG })

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('qa_failed')
    expect(c.qaF1).toBeCloseTo(0, 10)
    expect(llm.calls).toHaveLength(2) // gist + QA 生成，没有 answerFromGist
    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.flags).toEqual({ incompressible: true })
    }
  })

  it('M6 闸门失败（gistEpi ≥ epiThreshold）→ epi_failed 且标 incompressible（QA 已先通过）', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1', text: '2026年10月24日和张三开会' })) // 词法密集端
    store.records.set('m2', makeRecord({ id: 'm2', text: '用户喜欢简洁的回复' })) // 词法稀疏端
    store.records.set('m3', makeRecord({ id: 'm3', text: '服务器部署在 AWS 东京区' })) // 词法中等
    const gistText = '2026年10月24日和张三开会' // 与 m1 同文：词法密度位于池分布高端 → epi 高
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([{ q: '会议在哪天？', a: '10月24日' }]),
    ])
    const { provider } = makeEmbedding({ [gistText]: unit(0) })
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: withSleep({ epiThreshold: 0.001 }) })

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('epi_failed')
    expect(c.qaF1).toBeCloseTo(1, 10) // QA 闸门前置，确实先通过了
    expect(c.gistEpi).toBeGreaterThanOrEqual(0.001)
    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.flags).toEqual({ incompressible: true })
    }
    expect(store.nodes.size).toBe(0)
    expect(outcome.refreshedRecords).toBe(3)
  })

  // 挂入场景共用基建：预置 n1（S=4、member_refs=['x0']、向量 e1）；
  // 池内三记录对 n1 峰值余弦 0.1/0.2/0.3（μ=0.2 σ≈0.0816），gist 峰值 0.98 → con ≈ 9.55 > 0.5
  function attachSetup(compatReply: string) {
    const store = new FakeStore()
    store.nodes.set('n1', makeNode({ id: 'n1', gist: '用户重视已上线服务的质量', embedding: unit(0), S: 4, member_refs: ['x0'] }))
    store.records.set('m1', makeRecord({ id: 'm1', text: '用户重视部署的稳定性', embedding: plane(0.1), imp: 6 }))
    store.records.set('m2', makeRecord({ id: 'm2', text: '用户关注服务的可用性', embedding: plane(0.2), imp: 6 }))
    store.records.set('m3', makeRecord({ id: 'm3', text: '用户在意系统的可靠性', embedding: plane(0.3), imp: 6 }))
    const gistText = '用户重视部署的可观测性' // 与 n1 极近（cos=0.98）
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([
        { q: '用户重视部署的什么？', a: '稳定性' },
        { q: '用户关注服务的什么？', a: '可用性' },
        { q: '用户在意系统的什么？', a: '可靠性' },
      ]),
      compatReply,
    ])
    const { provider } = makeEmbedding({ [gistText]: plane(0.98) })
    const cycle = new SleepCycle({
      llm, embedding: provider, store,
      config: withSleep({ epiThreshold: 0.99, attachGain: 2.5 }), // 覆写增益更直观：S 应 +2.5
      now: () => 900, // 与节点锚点同刻：第 7 步兜底衰减不介入，S 断言只反映 attachGain
    })
    return { store, llm, cycle }
  }

  it('挂入：con 超阈值且 LLM 兼容 → attached，节点 S += attachGain、member_refs 追加', async () => {
    const { store, llm, cycle } = attachSetup('{"compatible":true}')

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('attached')
    expect(c.nodeId).toBe('n1')
    expect(store.nodes.size).toBe(1) // 未建新节点
    const node = store.nodes.get('n1')!
    expect(node.S).toBeCloseTo(4 + 2.5, 10)
    expect(node.member_refs).toEqual(['x0', 'm1', 'm2', 'm3'])
    expect(node.embedding).toEqual(unit(0)) // 挂入不改节点向量与 gist
    expect(node.gist).toBe('用户重视已上线服务的质量')
    for (const id of ['m1', 'm2', 'm3']) {
      const r = store.records.get(id)!
      expect(r.flags).toEqual({ compressed_into: 'n1', qa_passed: true })
      expect(r.con_route).toEqual({ kind: 'attached', nodeIds: ['n1'] })
    }
    expect(outcome.refreshedRecords).toBe(0)
    expect(outcome.refreshedNodes).toEqual(['n1'])
    expect(llm.calls).toHaveLength(6) // 1 gist + 1 QA 生成 + 3 作答 + 1 兼容性复核
    expect(llm.calls[5]!.user).toContain('用户重视已上线服务的质量') // 复核提示带节点 gist
    expect(llm.calls[5]!.user).toContain('用户重视部署的可观测性') // 与新语句
  })

  it('con 超阈值但 LLM 不兼容 → stayed：节点与成员都不动', async () => {
    const { store, llm, cycle } = attachSetup('{"compatible":false}')

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('stayed')
    expect(c.nodeId).toBeUndefined()
    const node = store.nodes.get('n1')!
    expect(node.S).toBeCloseTo(4, 10)
    expect(node.member_refs).toEqual(['x0'])
    for (const id of ['m1', 'm2', 'm3']) {
      const r = store.records.get(id)!
      expect(r.flags).toEqual({}) // 留层不标 incompressible，也不压缩
      expect(r.con_route).toEqual({ kind: 'pending' })
    }
    expect(outcome.refreshedRecords).toBe(3)
    expect(outcome.refreshedNodes).toEqual(['n1'])
    expect(llm.calls).toHaveLength(6)
  })

  it('con 中性（−1 < con < 0.5）→ stayed 且不发起兼容性复核', async () => {
    const store = new FakeStore()
    store.nodes.set('n1', makeNode({ id: 'n1', embedding: unit(0), S: 4 }))
    store.records.set('m1', makeRecord({ id: 'm1', embedding: plane(0.5) }))
    store.records.set('m2', makeRecord({ id: 'm2', embedding: plane(0.6) }))
    store.records.set('m3', makeRecord({ id: 'm3', embedding: plane(0.7) }))
    // 基线峰值 0.5/0.6/0.7（μ=0.6 σ≈0.0816），gist 峰值 0.58 → con ≈ −0.245
    const gistText = '用户重视部署的可观测性'
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([
        { q: '用户偏好怎样的回复？', a: '简洁' },
        { q: '回答应当保持多长？', a: '简短' },
        { q: '用户喜欢什么样的答案？', a: '直接' },
      ]),
    ])
    const { provider } = makeEmbedding({ [gistText]: plane(0.58) })
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: withSleep({ epiThreshold: 0.99 }) })

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('stayed')
    expect(c.nodeId).toBeUndefined()
    expect(llm.calls).toHaveLength(5) // 1 gist + 1 QA 生成 + 3 作答，无兼容性调用
    expect(llm.calls.filter((call) => call.system.includes('compatible'))).toHaveLength(0)
    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.flags).toEqual({})
    }
    expect(store.nodes.size).toBe(1)
    expect(outcome.refreshedRecords).toBe(3)
    expect(outcome.refreshedNodes).toEqual(['n1'])
  })

  // 建新节点场景共用基建：基线对 n1 峰值 0.9/0.95/1.0（μ=0.95 σ≈0.0408），gist 峰值 0.7
  // → con ≈ −6.12 < conNewNodeSigma(−1)；imp 决定是否过 conNewNodeMinImp=7 的闸
  function conNewNodeSetup(imp: number) {
    const store = new FakeStore()
    store.nodes.set('n1', makeNode({ id: 'n1', embedding: unit(0), S: 4, member_refs: ['x0'] }))
    store.records.set('m1', makeRecord({ id: 'm1', text: '用户重视部署的稳定性', imp, embedding: plane(0.9) }))
    store.records.set('m2', makeRecord({ id: 'm2', text: '用户关注服务的可用性', imp, embedding: plane(0.95) }))
    store.records.set('m3', makeRecord({ id: 'm3', text: '用户在意系统的可靠性', imp, embedding: plane(1) }))
    const gistText = '用户重视整体的运行质量' // 对 n1 余弦 0.7，远离基线
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([
        { q: '用户重视部署的什么？', a: '稳定性' },
        { q: '用户关注服务的什么？', a: '可用性' },
        { q: '用户在意系统的什么？', a: '可靠性' },
      ]),
    ])
    const { provider } = makeEmbedding({ [gistText]: plane(0.7) })
    const cycle = new SleepCycle({
      llm, embedding: provider, store,
      config: withSleep({ epiThreshold: 0.99 }), newId: () => 'node-2',
      now: () => 900, // 与锚点同刻：衰减不介入，S 断言只反映建节点初值
    })
    return { store, llm, cycle }
  }

  it('con < conNewNodeSigma 且簇均 imp=8 ≥ 7 → 建第二个节点（new_node）', async () => {
    const { store, llm, cycle } = conNewNodeSetup(8)

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('new_node')
    expect(c.nodeId).toBe('node-2')
    expect(store.nodes.size).toBe(2)
    const node2 = store.nodes.get('node-2')!
    expect(node2.S).toBeCloseTo(8, 10) // S = 簇均 imp
    expect(node2.member_refs).toEqual(['m1', 'm2', 'm3'])
    expect(node2.embedding).toEqual(plane(0.7))
    const n1 = store.nodes.get('n1')!
    expect(n1.S).toBeCloseTo(4, 10) // 原节点不动
    expect(n1.member_refs).toEqual(['x0'])
    for (const id of ['m1', 'm2', 'm3']) {
      const r = store.records.get(id)!
      expect(r.flags).toEqual({ compressed_into: 'node-2', qa_passed: true })
      expect(r.con_route).toEqual({ kind: 'new_node', nodeId: 'node-2' })
    }
    expect(outcome.refreshedRecords).toBe(0)
    expect(outcome.refreshedNodes).toEqual(['n1', 'node-2'])
    expect(llm.calls).toHaveLength(5) // 无兼容性复核（未走挂入分支）
  })

  it('con < conNewNodeSigma 但簇均 imp=5 < 7 → stayed，不建节点', async () => {
    const { store, llm, cycle } = conNewNodeSetup(5)

    const outcome = await cycle.run()

    const c = outcome.clusters[0]!
    expect(c.route).toBe('stayed')
    expect(c.nodeId).toBeUndefined()
    expect(store.nodes.size).toBe(1)
    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.flags).toEqual({})
    }
    expect(outcome.refreshedRecords).toBe(3)
    expect(outcome.refreshedNodes).toEqual(['n1'])
    expect(llm.calls).toHaveLength(5)
  })

  it('signal.aborted 且 LLM 抛错 → run() rejects（错误不被吞，未走到刷新）', async () => {
    const store = new FakeStore()
    for (let i = 1; i <= 3; i++) store.records.set(`m${i}`, makeRecord({ id: `m${i}` }))
    const llm = new FakeLlm([]) // 第一次 complete 即抛（解析重试一次后仍失败）
    const { provider } = makeEmbedding()
    const controller = new AbortController()
    controller.abort()
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: BASE_CONFIG })

    await expect(cycle.run(controller.signal)).rejects.toThrow()
    expect(llm.calls).toHaveLength(2) // 解析失败重试一次
    expect(store.refreshed).toEqual([]) // 中止上抛 → 未执行第 7 步
    expect(store.nodes.size).toBe(0)
    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.flags).toEqual({})
    }
  })

  it('池不足 clusterMinSize → 无簇、无 LLM/嵌入调用，但统计刷新照常执行', async () => {
    const store = new FakeStore()
    store.records.set('m1', makeRecord({ id: 'm1' }))
    store.records.set('m2', makeRecord({ id: 'm2' }))
    const llm = new FakeLlm([]) // 若被调用会抛错，兼作哨兵
    const { provider, embed } = makeEmbedding()
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: BASE_CONFIG })

    const outcome = await cycle.run()

    expect(outcome).toEqual({
      clusters: [], refreshedRecords: 2, refreshedNodes: [],
      decayedRecords: 2, decayedNodes: 0,
    })
    expect(llm.calls).toHaveLength(0)
    expect(embed.mock.calls).toHaveLength(0) // 探针嵌入只在有簇时发生
    expect(store.refreshed).toEqual([{ id: 'm1', k: 10 }, { id: 'm2', k: 10 }])
  })

  it('单簇 gist 两次解析失败不中断整轮：log 一次，另一团正常产出', async () => {
    const store = new FakeStore()
    // 甲团沿 e1、乙团沿 e2：互不相连的两个 3-团；甲团（id 靠前）先处理且 gist 两次返回非 JSON
    store.records.set('a1', makeRecord({ id: 'a1', text: '甲团关于部署节奏的记忆', embedding: unit(0) }))
    store.records.set('a2', makeRecord({ id: 'a2', text: '甲团关于部署的又一条记忆', embedding: unit(0) }))
    store.records.set('a3', makeRecord({ id: 'a3', text: '甲团关于部署的第三条记忆', embedding: unit(0) }))
    store.records.set('b1', makeRecord({ id: 'b1', text: '乙团关于代码风格的第一条记忆', embedding: unit(1) }))
    store.records.set('b2', makeRecord({ id: 'b2', text: '乙团关于代码风格的又一条记忆', embedding: unit(1) }))
    store.records.set('b3', makeRecord({ id: 'b3', text: '乙团关于代码风格的第三条记忆', embedding: unit(1) }))
    const gistText = '用户注重代码的可读性'
    const llm = new FakeLlm([
      '不是 JSON（甲第一次）', // 甲 gist 首次失败
      '仍不是 JSON（甲重试）', // 甲 gist 重试仍失败 → 抛出，被 run 捕获
      gistReply(gistText), // 乙 gist
      '[{"question":"用户注重代码的什么？","answer":"可读性"}]',
      '{"answer":"可读性"}',
    ])
    const { provider } = makeEmbedding({ [gistText]: unit(1) })
    const logs: string[] = []
    const cycle = new SleepCycle({
      llm, embedding: provider, store,
      config: withSleep({ epiThreshold: 0.99 }),
      newId: () => 'node-1', log: (m) => logs.push(m),
    })

    const outcome = await cycle.run()

    expect(logs).toHaveLength(1)
    expect(logs[0]).toContain('簇处理失败')
    expect(logs[0]).toContain('a1')
    expect(outcome.clusters).toHaveLength(1)
    const c = outcome.clusters[0]!
    expect(c.members).toEqual(['b1', 'b2', 'b3'])
    expect(c.route).toBe('new_node')
    expect(c.nodeId).toBe('node-1')
    expect(c.qaF1).toBeCloseTo(1, 10)
    expect(store.nodes.get('node-1')!.member_refs).toEqual(['b1', 'b2', 'b3'])
    for (const id of ['a1', 'a2', 'a3']) {
      // LLM 失败 ≠ 闸门拒收：不标 incompressible、不压缩，留下轮重试
      const r = store.records.get(id)!
      expect(r.flags).toEqual({})
      expect(r.con_route).toEqual({ kind: 'pending' })
    }
    for (const id of ['b1', 'b2', 'b3']) {
      expect(store.records.get(id)!.flags.compressed_into).toBe('node-1')
    }
    expect(outcome.refreshedRecords).toBe(3) // 只剩甲团三条
    expect(store.refreshed.map((x) => x.id)).toEqual(['a1', 'a2', 'a3'])
    expect(outcome.refreshedNodes).toEqual(['node-1'])
    expect(llm.calls).toHaveLength(5) // 甲耗 2 次 + 乙 3 次（1 gist + 1 生成 + 1 作答）
  })

  it('第 7 步兜底衰减（P2 §2.1）：留层记录与节点 S·e^(−Δt/τ)、tier 重算、锚点前移；已压缩记录不动', async () => {
    const store = new FakeStore()
    // 距 now 180 天的两条记录：m1 未压缩（衰减）、m2 已压缩（不动）；节点同样衰减
    const old = 1_000
    const now = old + 180 * 24 * 3600 * 1000
    store.records.set('m1', makeRecord({ id: 'm1', S: 5, tier: 'warm' }))
    store.records.set('m2', makeRecord({ id: 'm2', S: 5, flags: { compressed_into: 'n0' } }))
    store.nodes.set('n1', makeNode({ id: 'n1', S: 4, time_meta: { created_at: old, valid_at: old } }))
    // 池只有 1 条未压缩记录 < clusterMinSize → 无簇路径，衰减照常执行
    const { provider } = makeEmbedding()
    const cycle = new SleepCycle({ llm: new FakeLlm([]), embedding: provider, store, config: BASE_CONFIG, now: () => now })

    const outcome = await cycle.run()

    expect(outcome.decayedRecords).toBe(1)
    expect(outcome.decayedNodes).toBe(1)
    expect(store.records.get('m1')!.S).toBeCloseTo(5 * Math.exp(-1), 10)
    expect(store.records.get('m1')!.tier).toBe('cold') // 1.84 < 2
    expect(store.records.get('m1')!.counters.last_decay_at).toBe(now)
    expect(store.records.get('m2')!.S).toBe(5) // 已压缩：S 惰性，不参与分层
    expect(store.nodes.get('n1')!.S).toBeCloseTo(4 * Math.exp(-1), 10)
    expect(store.nodes.get('n1')!.counters?.last_decay_at).toBe(now)
  })

  it('迁移成员带 time_meta.compressed_at 归档时点（时间旅行视图消费，§2.5）', async () => {
    const store = new FakeStore()
    for (let i = 1; i <= 3; i++) store.records.set(`m${i}`, makeRecord({ id: `m${i}` }))
    const gistText = '用户偏好简洁直接的回复'
    const llm = new FakeLlm([
      gistReply(gistText),
      ...qaPerfect([{ q: '用户偏好怎样的回复？', a: '简洁' }]),
    ])
    const { provider } = makeEmbedding({ [gistText]: unit(0) })
    const at = 123_456
    const cycle = new SleepCycle({ llm, embedding: provider, store, config: withSleep({ epiThreshold: 0.99 }), now: () => at })

    await cycle.run()

    for (const id of ['m1', 'm2', 'm3']) {
      expect(store.records.get(id)!.time_meta.compressed_at).toBe(at)
    }
  })
})

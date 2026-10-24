/**
 * 存储层单测：MemoryStore / memoryUnitDescriptor / tierOf。
 * 自造内存版 FakeKvUnit（同一 backing 模拟重开）与 FakeVectorIndex（暴力余弦，平局按 id 升序）。
 */
import { describe, expect, it, vi } from 'vitest'
import { MemoryStore, memoryUnitDescriptor, tierOf } from '../src/memory-store.js'
import type { Config } from '../src/config.js'
import type { MemoryRecord, Neighbor, SemanticNode, VectorIndex } from '../src/types.js'
import type { KvUnit, KvUnitDescriptor } from '@deepseek-ai/dsh-storage'

// ---------------------------------------------------------------- 测试基建

/** 完整 Config 字面量（阈值等与断言耦合的字段显式写出）。 */
const testConfig = (): Config => ({
  storage: { backend: 'json', unit: 'dsh_mem_test', version: 7, workspaceScope: false },
  embedding: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 1000, dimensions: 2, batchSize: 1 },
  reranker: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 1000 },
  llm: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 512 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: { l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6, cslsK: 10, contextTurns: 3, coldCueBoost: 0.1, cueThreshold: 0.3 },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: { enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3, qaF1Threshold: 0.8, epiThreshold: 0.5, conWindow: 1000, conAttachSigma: 0.5, conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5 },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5, clear: false },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
})

/** 全字段 MemoryRecord（对照 types.ts；r_m/red/epi 为 0，con_route pending，counters 全 0）。 */
const makeRecord = (
  overrides: Partial<MemoryRecord> & Pick<MemoryRecord, 'id' | 'embedding'>,
): MemoryRecord => ({
  text: `text-${overrides.id}`,
  imp: 5,
  valence: 0,
  arousal: 0,
  epi: 0,
  red: 0,
  r_m: 0,
  tier: 'warm',
  con_route: { kind: 'pending' },
  links: [],
  S: 5,
  counters: { used_count: 0, suppressed_count: 0 },
  time_meta: { created_at: 1000, valid_at: 1000 },
  ...overrides,
})

const makeNode = (id: string, embedding: readonly number[], overrides: Partial<SemanticNode> = {}): SemanticNode => ({
  id,
  gist: `gist-${id}`,
  embedding,
  epi: 0,
  links: [],
  member_refs: [],
  time_meta: { created_at: 1000, valid_at: 1000 },
  S: 5,
  r_m: 0,
  ...overrides,
})

/** 共享介质：多个 FakeKvUnit 先后打开同一 backing，模拟"持久化后重开"。 */
class FakeBacking {
  readonly tables = new Map<string, Map<string, unknown>>()
  constructor(readonly descriptor: KvUnitDescriptor) {
    for (const table of descriptor.tables) this.tables.set(table, new Map())
  }
  put(table: string, key: string, value: unknown): void {
    const target = this.tables.get(table)
    if (target === undefined) throw new Error(`undeclared table: ${table}`)
    target.set(key, JSON.parse(JSON.stringify(value))) // 介质只存 JSON（与真实后端一致的往返语义）
  }
}

/** KvUnit 内存实现：数据存 backing Map，close 后一切调用抛错，close 幂等。 */
class FakeKvUnit implements KvUnit {
  private closed = false
  constructor(private readonly backing: FakeBacking) {}
  private assertOpen(): void {
    if (this.closed) throw new Error('closed')
  }
  async loadAll(): Promise<{ tables: Record<string, Record<string, unknown>>; global: unknown }> {
    this.assertOpen()
    const tables: Record<string, Record<string, unknown>> = {}
    for (const [name, table] of this.backing.tables) {
      tables[name] = Object.fromEntries([...table.entries()].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]))
    }
    return { tables, global: null }
  }
  async putRecord(table: string, key: string, value: unknown): Promise<void> {
    this.assertOpen()
    this.backing.put(table, key, value)
  }
  async deleteRecord(table: string, key: string): Promise<void> {
    this.assertOpen()
    this.backing.tables.get(table)?.delete(key)
  }
  async setGlobal(value: unknown): Promise<void> {
    this.assertOpen()
    if (!this.backing.descriptor.hasGlobal) throw new Error('unit declares no global slot')
    this.backing.global = value
  }
  async close(): Promise<void> {
    this.closed = true
  }
}

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (const x of a) na += x * x
  for (const x of b) nb += x * x
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) dot += a[i]! * b[i]!
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

/** VectorIndex 内存实现：暴力余弦，余弦降序、平局按 id 升序。 */
class FakeVectorIndex implements VectorIndex {
  private readonly vectors = new Map<string, readonly number[]>()
  add(id: string, vector: readonly number[]): void {
    this.vectors.set(id, vector)
  }
  remove(id: string): boolean {
    return this.vectors.delete(id)
  }
  size(): number {
    return this.vectors.size
  }
  search(query: readonly number[], k: number, filter?: (id: string) => boolean): readonly Neighbor[] {
    const hits: Neighbor[] = []
    for (const [id, vector] of this.vectors) {
      if (filter !== undefined && !filter(id)) continue
      hits.push({ id, cos: cosine(query, vector) })
    }
    hits.sort((a, b) => (a.cos !== b.cos ? b.cos - a.cos : a.id < b.id ? -1 : 1))
    return hits.slice(0, k)
  }
}

/** 开一套 store 环境：backing 跨 open 共享；index 由测试持有以便检查一致性。 */
async function makeEnv() {
  const config = testConfig()
  const backing = new FakeBacking(memoryUnitDescriptor(config.storage.unit, config))
  const open = async () => {
    const index = new FakeVectorIndex()
    const store = await MemoryStore.open(new FakeKvUnit(backing), index, config)
    return { index, store }
  }
  return { config, backing, open }
}

/** 独立环境快速打开（每个 it 各自全新 backing）；需跨重开共享介质的用例用 makeEnv()。 */
const open = async () => (await makeEnv()).open()

// ---------------------------------------------------------------- 用例

describe('memoryUnitDescriptor', () => {
  it('形状：表名 episodic+semantic，name 来自入参、version 来自 config，无 global 槽', () => {
    const config = testConfig()
    expect(memoryUnitDescriptor(config.storage.unit, config)).toEqual({
      name: 'dsh_mem_test',
      version: 7,
      tables: ['episodic', 'semantic'],
      hasGlobal: false,
    })
  })

  it('unit 名由入参决定（M2：工作区 unit 名由 scope.ts 派生后传入）', () => {
    const config = testConfig()
    config.storage = { backend: 'sqlite', unit: 'other_unit', version: 3, workspaceScope: true }
    const d = memoryUnitDescriptor('dsh_mem_ws_ab12cd34', config)
    expect(d.name).toBe('dsh_mem_ws_ab12cd34')
    expect(d.version).toBe(3)
  })

  it('unit 名连字符归一为下划线（宿主不允许连字符）', () => {
    const config = testConfig()
    expect(memoryUnitDescriptor('dsh-mem-test', config).name).toBe('dsh_mem_test')
  })

  it('归一化后仍不合规的 unit 名在启动期抛错', () => {
    const config = testConfig()
    expect(() => memoryUnitDescriptor('1bad Unit', config)).toThrow(/invalid storage unit name/)
  })
})

describe('tierOf（§2.1 边界）', () => {
  const tier = { hotThreshold: 8, coldThreshold: 2 }
  it('S ≥ hotThreshold → hot（含等号）', () => {
    expect(tierOf(8, tier)).toBe('hot')
    expect(tierOf(10, tier)).toBe('hot')
  })
  it('coldThreshold ≤ S < hotThreshold → warm（含下界等号）', () => {
    expect(tierOf(7.999, tier)).toBe('warm')
    expect(tierOf(2, tier)).toBe('warm')
    expect(tierOf(5, tier)).toBe('warm')
  })
  it('S < coldThreshold → cold（不含等号）', () => {
    expect(tierOf(1.999, tier)).toBe('cold')
    expect(tierOf(0, tier)).toBe('cold')
    expect(tierOf(-3, tier)).toBe('cold')
  })
})

describe('MemoryStore 持久化往返', () => {
  it('open → putRecord → 重开（同一 backing 的新 FakeKvUnit）能读回记录', async () => {
    const { backing, open } = await makeEnv()
    const record = makeRecord({ id: 'a', embedding: [1, 0], text: '用户偏好简洁回答' })
    {
      const { store } = await open()
      await store.putRecord(record)
      await store.close()
    }
    const reopened = new FakeKvUnit(backing) // 新 unit 实例，同一介质
    const index2 = new FakeVectorIndex()
    const store2 = await MemoryStore.open(reopened, index2, testConfig())
    expect(store2.get('a')).toEqual(record)
    expect(store2.allMemories().map((r) => r.id)).toEqual(['a'])
    expect(index2.size()).toBe(1) // 重开时 valid 记录重建进索引
    await store2.close()
  })

  it('softInvalidate 后重开：记录保留且不进索引（§2.5 记录永不删除）', async () => {
    const { backing, open } = await makeEnv()
    {
      const { store } = await open()
      await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
      await store.softInvalidate('a', 'b', 4242)
      await store.close()
    }
    const { index, store } = await open()
    expect(store.get('a')?.time_meta.invalid_at).toBe(4242)
    expect(store.get('a')?.flags.superseded_by).toBe('b')
    expect(store.validMemories()).toHaveLength(0)
    expect(index.size()).toBe(0)
    await store.close()
  })

  it('open 时跳过不合规记录并逐条 console.warn，不抛错', async () => {
    const { backing, open } = await makeEnv()
    backing.put('episodic', 'bad-id', { id: 42, text: 'x', embedding: [1, 0] })
    backing.put('episodic', 'bad-emb', { id: 'bad-emb', text: 'x', embedding: 'not-array' })
    backing.put('episodic', 'bad-emb2', { id: 'bad-emb2', text: 'x', embedding: [1, 'two'] })
    backing.put('episodic', 'good', makeRecord({ id: 'good', embedding: [0, 1] }))
    backing.put('semantic', 'bad-node', { id: 'bad-node', gist: 7, embedding: [1, 0] })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { store } = await open()
    expect(store.allMemories().map((r) => r.id)).toEqual(['good'])
    expect(store.allNodes()).toHaveLength(0)
    expect(warn).toHaveBeenCalledTimes(4)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('bad-id'))
    warn.mockRestore()
    await store.close()
  })
})

describe('MemoryStore 软失效（§2.5）', () => {
  it('validMemories 排除软失效记录；allMemories 仍含', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [0, 1] }))
    await store.softInvalidate('a', 'b', 1111)
    expect(store.validMemories().map((r) => r.id)).toEqual(['b'])
    expect(store.allMemories().map((r) => r.id).sort()).toEqual(['a', 'b'])
    await store.close()
  })

  it('软失效后 similar() 不再返回该记忆；重复调用幂等不抛且保持首次时间戳', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [1, 0.0001] }))
    expect(store.similar([1, 0], 10).map((h) => h.id)).toContain('a')
    await store.softInvalidate('a', 'b', 777)
    expect(store.similar([1, 0], 10).map((h) => h.id)).not.toContain('a')
    await expect(store.softInvalidate('a', 'b', 999999)).resolves.toBeUndefined()
    expect(store.get('a')?.time_meta.invalid_at).toBe(777)
    await store.close()
  })

  it('softInvalidate 对不存在的 id 抛错', async () => {
    const { store } = await open()
    await expect(store.softInvalidate('nope', 'x', 1)).rejects.toThrow('memory not found')
    await store.close()
  })
})

describe('MemoryStore updateRecord', () => {
  it('读改写：fn 作用于当前值并写回（内存 + 持久化）', async () => {
    const { open } = await makeEnv()
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    const updated = await store.updateRecord('a', (r) => ({
      ...r,
      counters: { ...r.counters, used_count: r.counters.used_count + 1, last_used_at: 55 },
      S: 9,
    }))
    expect(updated.counters.used_count).toBe(1)
    expect(store.get('a')?.counters.used_count).toBe(1)
    expect(store.get('a')?.S).toBe(9)
    await store.close()
    // 重开验证持久化
    const { store: store2 } = await open()
    expect(store2.get('a')?.counters.used_count).toBe(1)
    expect(store2.get('a')?.S).toBe(9)
    await store2.close()
  })

  it('不存在的 id 抛错', async () => {
    const { store } = await open()
    await expect(store.updateRecord('nope', (r) => r)).rejects.toThrow('memory not found')
    await store.close()
  })

  it('updateRecord 把记录改为 invalid 后索引同步移除；改回 valid 后恢复', async () => {
    const { index, store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [0, 1] }))
    expect(index.size()).toBe(2)
    await store.updateRecord('a', (r) => ({ ...r, time_meta: { ...r.time_meta, invalid_at: 88 } }))
    expect(index.size()).toBe(1)
    expect(store.similar([1, 0], 10).map((h) => h.id)).toEqual(['b'])
    await store.updateRecord('a', (r) => {
      const { invalid_at: _drop, ...timeMeta } = r.time_meta
      return { ...r, time_meta: timeMeta }
    })
    expect(index.size()).toBe(2)
    expect(store.similar([1, 0], 10).map((h) => h.id)).toContain('a')
    await store.close()
  })
})

describe('MemoryStore refreshNeighborhood（§3.2/§3.3）', () => {
  it('数值正确：二维单位向量夹角 0°/60°/-45° 手算 r_m 与 red', async () => {
    const { store } = await open()
    // a=0°, b=60°, c=-45°（单位向量）→ cos(a,c)=cos45°, cos(a,b)=cos60°
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [Math.cos(Math.PI / 3), Math.sin(Math.PI / 3)] }))
    await store.putRecord(makeRecord({ id: 'c', embedding: [Math.cos(-Math.PI / 4), Math.sin(-Math.PI / 4)] }))

    // k=1：r_m = top-1 近邻余弦 = cos45°；red = top-5（实际 2 个）均值 = (cos45°+cos60°)/2
    const { r_m, red } = await store.refreshNeighborhood('a', 1)
    expect(r_m).toBeCloseTo(Math.cos(Math.PI / 4), 10)
    expect(red).toBeCloseTo((Math.cos(Math.PI / 4) + Math.cos(Math.PI / 3)) / 2, 10)
    // 写回记录
    expect(store.get('a')?.r_m).toBeCloseTo(Math.cos(Math.PI / 4), 10)
    expect(store.get('a')?.red).toBeCloseTo((Math.cos(Math.PI / 4) + Math.cos(Math.PI / 3)) / 2, 10)
    await store.close()
  })

  it('k 大于近邻数时取实际数（不足 k 取实际数）', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [0, 1] }))
    const { r_m, red } = await store.refreshNeighborhood('a', 10)
    const expected = Math.cos(Math.PI / 2)
    expect(r_m).toBeCloseTo(expected, 10)
    expect(red).toBeCloseTo(expected, 10)
    await store.close()
  })

  it('近邻数为 0 时 r_m=0（且写回 0）', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    const { r_m, red } = await store.refreshNeighborhood('a', 5)
    expect(r_m).toBe(0)
    expect(red).toBe(0)
    expect(store.get('a')?.r_m).toBe(0)
    await store.close()
  })

  it('近邻只统计 valid 记忆（排除自身与软失效记录）', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'b', embedding: [Math.cos(Math.PI / 3), Math.sin(Math.PI / 3)] }))
    await store.putRecord(makeRecord({ id: 'c', embedding: [Math.cos(-Math.PI / 4), Math.sin(-Math.PI / 4)] }))
    await store.softInvalidate('c', 'a', 1) // 最近邻 c 失效 → 只剩 b
    const { r_m, red } = await store.refreshNeighborhood('a', 5)
    expect(r_m).toBeCloseTo(Math.cos(Math.PI / 3), 10)
    expect(red).toBeCloseTo(Math.cos(Math.PI / 3), 10)
    await store.close()
  })

  it('不存在的 id 抛错', async () => {
    const { store } = await open()
    await expect(store.refreshNeighborhood('nope', 5)).rejects.toThrow('memory not found')
    await store.close()
  })
})

describe('MemoryStore 语义层节点（P1 睡眠周期填充）', () => {
  it('putNode/getNode/allNodes 往返，且重开后仍可读', async () => {
    const { open } = await makeEnv()
    const node = makeNode('n1', [1, 0])
    {
      const { store } = await open()
      await store.putNode(node)
      expect(store.getNode('n1')).toEqual(node)
      expect(store.allNodes().map((n) => n.id)).toEqual(['n1'])
      await store.close()
    }
    const { store: store2 } = await open()
    expect(store2.getNode('n1')?.gist).toBe('gist-n1')
    expect(store2.allNodes()).toHaveLength(1)
    await store2.close()
  })

  it('有效节点进入索引；软失效节点（invalid_at）移出且退出检索宇宙', async () => {
    const { index, store } = await open()
    await store.putNode(makeNode('n1', [1, 0]))
    expect(index.size()).toBe(1)
    await store.putNode(makeNode('n2', [0, 1], { time_meta: { created_at: 1, valid_at: 1, invalid_at: 9 } }))
    expect(index.size()).toBe(1) // 软失效节点不入索引
    expect(store.validNodes().map((n) => n.id)).toEqual(['n1'])
    expect(store.retrievable().map((i) => i.id)).toEqual(['n1'])
    await store.close()
  })

  it('retrievable()：有效未压缩记录 + 有效节点；tier 由 S 分层（§2.1）；压缩记录退出', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'r1', embedding: [1, 0], text: 'raw memory' }))
    await store.putRecord(makeRecord({ id: 'r2', embedding: [0, 1], flags: { compressed_into: 'n1' } }))
    await store.putNode(makeNode('n1', [0.6, 0.8], { gist: 'distilled gist', S: 9, r_m: 0.3 }))
    await store.putNode(makeNode('n2', [0, 1], { S: 1, time_meta: { created_at: 1, valid_at: 1, invalid_at: 5 } }))
    const items = store.retrievable()
    expect(items.map((i) => i.id)).toEqual(['r1', 'n1'])
    const [raw, node] = items
    expect(raw).toMatchObject({ id: 'r1', text: 'raw memory', layer: 'episodic', tier: 'warm', createdAt: 1000, r_m: 0 })
    expect(node).toMatchObject({ id: 'n1', text: 'distilled gist', layer: 'semantic', tier: 'hot', r_m: 0.3, createdAt: 1000 })
    expect(node?.embedding).toEqual([0.6, 0.8])
    await store.close()
  })

  it('refreshNeighborhood 近邻取自检索宇宙：节点计入记录 r_m；压缩记录不计入', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putNode(makeNode('n1', [Math.cos(Math.PI / 6), Math.sin(Math.PI / 6)])) // 与 a 夹角 30°
    // b 与 a 近乎同向但已压缩归档：若过滤器失效，r_m 会是 (≈1 + cos30°)/2 而非 cos30°
    await store.putRecord(makeRecord({ id: 'b', embedding: [Math.cos(0.01), Math.sin(0.01)], flags: { compressed_into: 'n1' } }))
    const { r_m, red } = await store.refreshNeighborhood('a', 5)
    expect(r_m).toBeCloseTo(Math.cos(Math.PI / 6), 10)
    expect(red).toBeCloseTo(Math.cos(Math.PI / 6), 10)
    await store.close()
  })

  it('refreshNodeNeighborhood：检索宇宙内 top-k（排除自身），写回节点并持久化', async () => {
    const { open } = await makeEnv()
    {
      const { store } = await open()
      await store.putNode(makeNode('n1', [1, 0]))
      await store.putNode(makeNode('n2', [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)]))
      await store.putRecord(makeRecord({ id: 'r1', embedding: [0, 1] })) // 与 n1 正交，k=1 时不如 n2
      const r_m = await store.refreshNodeNeighborhood('n1', 1)
      expect(r_m).toBeCloseTo(Math.cos(Math.PI / 4), 10)
      expect(store.getNode('n1')?.r_m).toBeCloseTo(Math.cos(Math.PI / 4), 10)
      await store.close()
    }
    const { store: store2 } = await open()
    expect(store2.getNode('n1')?.r_m).toBeCloseTo(Math.cos(Math.PI / 4), 10)
    await store2.close()
  })

  it('refreshNodeNeighborhood 对不存在的 id 抛错', async () => {
    const { store } = await open()
    await expect(store.refreshNodeNeighborhood('nope', 5)).rejects.toThrow('node not found')
    await store.close()
  })
})

describe('MemoryStore 语义节点软失效与近邻（支柱五完整版，P2）', () => {
  it('similarNodes：有效节点按余弦返回，排除软失效节点与记录', async () => {
    const { store } = await open()
    await store.putNode(makeNode('n1', [1, 0]))
    await store.putNode(makeNode('n2', [0.6, 0.8]))
    await store.putNode(makeNode('dead', [1, 0], { time_meta: { created_at: 1, valid_at: 1, invalid_at: 9 } }))
    await store.putRecord(makeRecord({ id: 'r1', embedding: [1, 0] }))
    const hits = store.similarNodes([1, 0.1], 10)
    expect(hits.map((h) => h.id)).toEqual(['n1', 'n2'])
    expect(hits[0]!.cos).toBeGreaterThan(hits[1]!.cos)
    expect(hits[0]!.node.id).toBe('n1')
    await store.close()
  })

  it('softInvalidateNode：退出索引与检索宇宙、保留存储、幂等、写 superseded_by；重开后仍可读', async () => {
    const { backing, open } = await makeEnv()
    {
      const { index, store } = await open()
      await store.putNode(makeNode('n1', [1, 0]))
      await store.putRecord(makeRecord({ id: 'r1', embedding: [0, 1] }))
      await store.softInvalidateNode('n1', 'r2', 4242)
      expect(store.getNode('n1')?.time_meta.invalid_at).toBe(4242)
      expect(store.getNode('n1')?.superseded_by).toBe('r2')
      expect(store.validNodes()).toHaveLength(0)
      expect(store.retrievable().map((i) => i.id)).toEqual(['r1'])
      expect(index.size()).toBe(1) // 只剩 r1
      // 幂等：重复调用不抛、保持首次时间戳
      await store.softInvalidateNode('n1', 'other', 9999)
      expect(store.getNode('n1')?.time_meta.invalid_at).toBe(4242)
      await store.close()
    }
    const { store: store2 } = await open()
    expect(store2.getNode('n1')?.superseded_by).toBe('r2')
    expect(store2.validNodes()).toHaveLength(0)
    await store2.close()
  })

  it('softInvalidateNode 对不存在的 id 抛错', async () => {
    const { store } = await open()
    await expect(store.softInvalidateNode('nope', 'x', 1)).rejects.toThrow('node not found')
    await store.close()
  })
})

describe('MemoryStore retrievableAt（时间旅行，§2.5 完整版）', () => {
  it('有效性窗口重建：valid_at ≤ t < invalid_at 的记录在列，窗口外不在', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'live', embedding: [1, 0] })) // valid 1000 →
    await store.putRecord(makeRecord({ id: 'old', embedding: [1, 0], time_meta: { created_at: 500, valid_at: 500, invalid_at: 900 } }))
    await store.putRecord(makeRecord({ id: 'dying', embedding: [1, 0], time_meta: { created_at: 500, valid_at: 500, invalid_at: 1500 } }))
    expect(store.retrievableAt(600).map((i) => i.id).sort()).toEqual(['dying', 'old'])
    expect(store.retrievableAt(1000).map((i) => i.id).sort()).toEqual(['dying', 'live']) // invalid_at=1500 > 1000 仍在
    expect(store.retrievableAt(1500).map((i) => i.id)).toEqual(['live']) // invalid_at ≤ t 排除
    expect(store.retrievableAt(400)).toEqual([]) // valid_at 500 > 400
    await store.close()
  })

  it('压缩时点重建：compressed_at ≤ t 排除、compressed_at > t 仍在；当前视图恒排除', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({
      id: 'r1', embedding: [1, 0],
      time_meta: { created_at: 500, valid_at: 500, compressed_at: 1200 },
      flags: { compressed_into: 'n1' }, // 迁移时 flags 与 compressed_at 同笔写入
    }))
    await store.putNode(makeNode('n1', [0, 1], { time_meta: { created_at: 1300, valid_at: 1300 } }))
    expect(store.retrievableAt(1100).map((i) => i.id)).toEqual(['r1']) // 压缩前：记录还在
    expect(store.retrievableAt(1250)).toEqual([]) // 记录已压缩、节点未创建：间隙期
    expect(store.retrievableAt(1400).map((i) => i.id)).toEqual(['n1'])
    expect(store.retrievable().map((i) => i.id)).toEqual(['n1']) // 当前视图：压缩记录退出
    await store.close()
  })

  it('P1 遗留归档（compressed_into 无 compressed_at）：保守视为历来已压缩，任何时点不复活', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'legacy', embedding: [1, 0], flags: { compressed_into: 'n1' } }))
    expect(store.retrievableAt(1000)).toEqual([])
    await store.close()
  })

  it('节点窗口：valid_at ≤ t < invalid_at；软失效节点在失效前时点仍可检索', async () => {
    const { store } = await open()
    await store.putNode(makeNode('n1', [1, 0], { time_meta: { created_at: 1000, valid_at: 1000, invalid_at: 2000 } }))
    await store.putNode(makeNode('n2', [0, 1], { time_meta: { created_at: 1000, valid_at: 1000 } }))
    expect(store.retrievableAt(1500).map((i) => i.id).sort()).toEqual(['n1', 'n2']) // n1 于 2000 失效，1500 时仍有效
    expect(store.retrievableAt(2500).map((i) => i.id)).toEqual(['n2'])
    await store.softInvalidateNode('n2', 'x', 3000)
    expect(store.retrievableAt(2999).map((i) => i.id)).toEqual(['n2']) // 失效前的时点仍可检索（时间旅行语义）
    expect(store.retrievableAt(3000).map((i) => i.id)).toEqual([]) // 失效时点起排除
    await store.close()
  })
})

describe('MemoryStore 生命周期', () => {
  it('close 幂等；close 后所有写方法抛 Error(\'closed\')', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.close()
    await expect(store.close()).resolves.toBeUndefined()
    await expect(store.putRecord(makeRecord({ id: 'b', embedding: [1, 0] }))).rejects.toThrow('closed')
    await expect(store.updateRecord('a', (r) => r)).rejects.toThrow('closed')
    await expect(store.softInvalidate('a', 'b', 1)).rejects.toThrow('closed')
    await expect(store.putNode(makeNode('n', [1, 0]))).rejects.toThrow('closed')
    await expect(store.refreshNeighborhood('a', 5)).rejects.toThrow('closed')
    await expect(store.refreshNodeNeighborhood('n', 5)).rejects.toThrow('closed')
  })
})

describe('MemoryStore clearAll（memory_clear 后端：唯一硬删除路径）', () => {
  it('双表全清（含软失效与已压缩记录）：计数正确、镜像与索引同步清空', async () => {
    const { index, store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'dead', embedding: [0, 1], time_meta: { created_at: 1, valid_at: 1, invalid_at: 9 } }))
    await store.putRecord(makeRecord({ id: 'arch', embedding: [1, 1], flags: { compressed_into: 'n1' } }))
    await store.putNode(makeNode('n1', [0.6, 0.8]))
    await store.putNode(makeNode('n2', [0, 1], { time_meta: { created_at: 1, valid_at: 1, invalid_at: 9 } }))
    // 索引恒含有效记录（含已压缩）与有效节点：a + arch + n1
    expect(index.size()).toBe(3)
    const counts = await store.clearAll()
    expect(counts).toEqual({ episodic: 3, semantic: 2 })
    expect(store.allMemories()).toHaveLength(0)
    expect(store.allNodes()).toHaveLength(0)
    expect(store.validMemories()).toHaveLength(0)
    expect(store.retrievable()).toHaveLength(0)
    expect(store.retrievableAt(500)).toHaveLength(0)
    expect(index.size()).toBe(0)
    expect(store.similar([1, 0], 10)).toHaveLength(0)
    await store.close()
  })

  it('清空是持久化的：重开（同一介质）后两表为空', async () => {
    const { open } = await makeEnv()
    {
      const { store } = await open()
      await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
      await store.putNode(makeNode('n1', [0, 1]))
      await store.clearAll()
      await store.close()
    }
    const { store: store2 } = await open()
    expect(store2.allMemories()).toHaveLength(0)
    expect(store2.allNodes()).toHaveLength(0)
    expect(store2.retrievable()).toHaveLength(0)
    await store2.close()
  })

  it('空库清空是干净的无操作（0/0）', async () => {
    const { store } = await open()
    await expect(store.clearAll()).resolves.toEqual({ episodic: 0, semantic: 0 })
    await store.close()
  })

  it('close 后抛 Error(\'closed\')', async () => {
    const { store } = await open()
    await store.close()
    await expect(store.clearAll()).rejects.toThrow('closed')
  })
})

describe('索引与 valid 集一致性', () => {
  it('putRecord 直接写入 invalid 记录时不入索引', async () => {
    const { index, store } = await open()
    await store.putRecord(makeRecord({ id: 'a', embedding: [1, 0] }))
    await store.putRecord(
      makeRecord({ id: 'dead', embedding: [0, 1], time_meta: { created_at: 1, valid_at: 1, invalid_at: 9 } }),
    )
    expect(index.size()).toBe(1)
    // kNN 不按余弦阈值截断：查询 [0,1] 仍会命中 valid 的 a（cos=0），但不含 dead
    expect(store.similar([0, 1], 10).map((h) => h.id)).toEqual(['a'])
    await store.close()
  })

  it('similar 按余弦降序返回并携带完整记录', async () => {
    const { store } = await open()
    await store.putRecord(makeRecord({ id: 'near', embedding: [1, 0] }))
    await store.putRecord(makeRecord({ id: 'far', embedding: [0, 1] }))
    const hits = store.similar([1, 0.1], 10)
    expect(hits.map((h) => h.id)).toEqual(['near', 'far'])
    expect(hits[0]!.cos).toBeGreaterThan(hits[1]!.cos)
    expect(hits[0]!.record.id).toBe('near')
    await store.close()
  })
})

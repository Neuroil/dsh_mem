/**
 * 语义图边工具（P3-2）单测：去重追加、度数上限、密度统计。
 * P3-3 增：邻接构建（池外过滤/无向对称）与 PPR（手算链/归一化/边界/确定性）。
 */
import { describe, expect, it } from 'vitest'
import { addLink, addRelatedCapped, buildAdjacency, graphStats, personalizedPagerank, relatedDegree } from '../src/graph.js'
import type { AdjacentItem, MemoryLink, MemoryRecord, SemanticNode } from '../src/types.js'

const record = (id: string, links: MemoryLink[]): MemoryRecord => ({
  id, text: `t-${id}`, embedding: [1, 0],
  imp: 5, valence: 0, arousal: 0, epi: 0, red: 0, r_m: 0, tier: 'warm',
  con_route: { kind: 'pending' }, links, S: 5,
  counters: { used_count: 0, suppressed_count: 0 },
  time_meta: { created_at: 1, valid_at: 1 }, flags: {},
})

const node = (id: string, memberRefs: string[], links: MemoryLink[] = []): SemanticNode => ({
  id, gist: `g-${id}`, embedding: [0, 1], epi: 0, links, member_refs: memberRefs,
  time_meta: { created_at: 1, valid_at: 1 }, S: 5, r_m: 0,
})

describe('addLink / relatedDegree', () => {
  it('kind+target 去重；不同 kind 或不同 target 不算重复', () => {
    let links: readonly MemoryLink[] = []
    links = addLink(links, { kind: 'related', target: 'a' })
    links = addLink(links, { kind: 'related', target: 'a' }) // 重复：无效果
    links = addLink(links, { kind: 'supersedes', target: 'a' }) // 不同 kind：保留
    expect(links).toEqual([{ kind: 'related', target: 'a' }, { kind: 'supersedes', target: 'a' }])
    expect(relatedDegree(links)).toBe(1) // 度数只数 related
  })
})

describe('addRelatedCapped（度数上限）', () => {
  it('按序消费额度，超上限即停；已有边不重复计', () => {
    const links = addRelatedCapped([{ kind: 'related', target: 'x' }], ['a', 'b', 'c', 'd'], 3)
    expect(links).toEqual([
      { kind: 'related', target: 'x' },
      { kind: 'related', target: 'a' },
      { kind: 'related', target: 'b' },
      // c、d 超过 maxDegree=3（x,a,b 已占 3）
    ])
  })

  it('maxDegree 足够大时全量追加', () => {
    const links = addRelatedCapped([], ['a', 'b'], 8)
    expect(links).toHaveLength(2)
  })
})

describe('graphStats（密度统计）', () => {
  it('边 = 记录/节点 links + 节点 member_refs；平均度 = 2E/N', () => {
    const records = [
      record('r1', [{ kind: 'related', target: 'r2' }, { kind: 'supersedes', target: 'n1' }]),
      record('r2', []),
    ]
    const nodes = [node('n1', ['r1', 'r2']), node('n2', [])]
    const stats = graphStats(records, nodes)
    expect(stats.items).toBe(4)
    // 逐项：r1.links=2, r2.links=0, n1.links=0+member_refs=2, n2=0
    expect(stats.edges).toBe(4)
    expect(stats.avgDegree).toBeCloseTo((2 * 4) / 4, 10)
  })

  it('空图返回 0 且不除零', () => {
    expect(graphStats([], [])).toEqual({ items: 0, edges: 0, avgDegree: 0 })
  })
})

describe('buildAdjacency（P3-3 邻接构建）', () => {
  const item = (id: string, neighbors: string[]): AdjacentItem => ({ id, neighbors })

  it('无向对称：任一侧声明都双向连通', () => {
    const adjacency = buildAdjacency([item('a', ['b']), item('b', []), item('c', [])])
    expect([...adjacency.get('a')!]).toEqual(['b'])
    expect([...adjacency.get('b')!]).toEqual(['a'])
    expect(adjacency.get('c')!).toEqual(new Set())
  })

  it('池外与自环 id 被过滤（软失效/归档出池后图遍历不越池）', () => {
    const adjacency = buildAdjacency([item('a', ['gone', 'a', 'b']), item('b', [])])
    expect([...adjacency.get('a')!]).toEqual(['b'])
    expect(adjacency.has('gone')).toBe(false)
  })

  it('空池返回空 Map', () => {
    expect(buildAdjacency([])).toEqual(new Map())
  })
})

describe('personalizedPagerank（P3-3，幂迭代）', () => {
  /** A—B—C 路径图（A 为唯一种子）的邻接表。 */
  function pathABC(): Map<string, Set<string>> {
    return buildAdjacency([
      { id: 'A', neighbors: ['B'] },
      { id: 'B', neighbors: ['A', 'C'] },
      { id: 'C', neighbors: ['B'] },
    ])
  }

  it('A-B-C 链、damping 0.5、种子 A：收敛到解析稳态 7/12 / 1/3 / 1/12', () => {
    // 稳态方程：r = 0.5·seed + 0.5·W·r → r(B)=1/3、r(A)=7/12、r(C)=1/12
    const rank = personalizedPagerank(pathABC(), new Map([['A', 1]]), 0.5, 100, 1e-12)
    expect(rank.get('A')).toBeCloseTo(7 / 12, 6)
    expect(rank.get('B')).toBeCloseTo(1 / 3, 6)
    expect(rank.get('C')).toBeCloseTo(1 / 12, 6)
  })

  it('种子权重归一化为分布：{A:2,B:1} 与 {A:2/3,B:1/3} 结果一致；总质量守恒为 1', () => {
    const twoSeeds = personalizedPagerank(pathABC(), new Map([['A', 2], ['B', 1]]), 0.5, 100, 1e-12)
    const normalized = personalizedPagerank(pathABC(), new Map([['A', 2 / 3], ['B', 1 / 3]]), 0.5, 100, 1e-12)
    for (const id of ['A', 'B', 'C']) {
      expect(twoSeeds.get(id)).toBeCloseTo(normalized.get(id)!, 10)
    }
    const total = [...twoSeeds.values()].reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 8)
  })

  it('悬挂质量回撒：孤立节点的质量不丢失（总和仍为 1）', () => {
    const adjacency = buildAdjacency([
      { id: 'A', neighbors: ['B'] },
      { id: 'B', neighbors: [] },
      { id: 'C', neighbors: [] }, // C 完全孤立（无向图里 A-B 仍连通）：度数 0，质量走回撒
    ])
    const rank = personalizedPagerank(adjacency, new Map([['A', 1]]), 0.5, 100, 1e-12)
    const total = [...rank.values()].reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 8)
    expect(rank.size).toBe(3)
  })

  it('空图或种子全不在图中返回空 Map（调用方据此精确跳过 PPR）', () => {
    expect(personalizedPagerank(new Map(), new Map([['A', 1]]), 0.5)).toEqual(new Map())
    expect(personalizedPagerank(pathABC(), new Map([['X', 1]]), 0.5)).toEqual(new Map())
    // 种子权重非正同样视为无种子
    expect(personalizedPagerank(pathABC(), new Map([['A', 0]]), 0.5)).toEqual(new Map())
  })

  it('确定性：同输入同输出（Map 按插入序遍历，无迭代顺序抖动）', () => {
    const first = personalizedPagerank(pathABC(), new Map([['A', 1]]), 0.5)
    const second = personalizedPagerank(pathABC(), new Map([['A', 1]]), 0.5)
    expect([...first.entries()]).toEqual([...second.entries()])
  })
})

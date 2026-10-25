/**
 * 语义图边工具（P3-2）单测：去重追加、度数上限、密度统计。
 */
import { describe, expect, it } from 'vitest'
import { addLink, addRelatedCapped, graphStats, relatedDegree } from '../src/graph.js'
import type { MemoryLink, MemoryRecord, SemanticNode } from '../src/types.js'

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

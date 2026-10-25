/**
 * 灵感通道（P3-4）单测：PRNG 确定性、seed 分量、采样器的去重/上限与
 * 两个分布臂的偏置统计（冷门优先、red 加成、topic 锚定、counter 对冲）。
 */
import { describe, expect, it } from 'vitest'
import { inspireSeed, localDateKey, mulberry32, sampleInspirations } from '../src/inspire.js'
import type { InspireItem } from '../src/types.js'

let next = 0
function makeItem(overrides: Partial<InspireItem> = {}): InspireItem {
  return {
    id: `m${next++}`,
    text: 'memory',
    embedding: [1, 0],
    r_m: 0,
    tier: 'warm',
    createdAt: 0,
    layer: 'episodic',
    usedCount: 0,
    red: 0,
    S: 5,
    ...overrides,
  }
}

describe('mulberry32', () => {
  it('同 seed 输出序列逐位一致；异 seed 序列不同', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    const c = mulberry32(43)
    const seqA = Array.from({ length: 8 }, () => a())
    const seqB = Array.from({ length: 8 }, () => b())
    const seqC = Array.from({ length: 8 }, () => c())
    expect(seqA).toEqual(seqB)
    expect(seqA).not.toEqual(seqC)
  })
  it('输出落在 [0,1)', () => {
    const rng = mulberry32(7)
    for (let i = 0; i < 1000; i++) {
      const v = rng()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })
})

describe('localDateKey / inspireSeed', () => {
  it('本地日期键 YYYY-MM-DD，月日补零', () => {
    expect(localDateKey(new Date(2026, 0, 5, 23, 59))).toBe('2026-01-05')
    expect(localDateKey(new Date(2026, 10, 25, 0, 0))).toBe('2026-11-25')
  })
  it('seed 对三分量各自敏感：同分量同 seed、任一分量变化即漂移', () => {
    const base = inspireSeed('scope-a', '2026-10-25', 'cats')
    expect(inspireSeed('scope-a', '2026-10-25', 'cats')).toBe(base)
    expect(inspireSeed('scope-b', '2026-10-25', 'cats')).not.toBe(base)
    expect(inspireSeed('scope-a', '2026-10-26', 'cats')).not.toBe(base)
    expect(inspireSeed('scope-a', '2026-10-25', 'dogs')).not.toBe(base)
    expect(inspireSeed(undefined, '2026-10-25', 'cats')).not.toBe(base)
  })
})

describe('sampleInspirations', () => {
  it('空池 / count ≤ 0 返回空数组', () => {
    const rng = mulberry32(1)
    expect(sampleInspirations([], { count: 3, rng })).toEqual([])
    expect(sampleInspirations([makeItem()], { count: 0, rng })).toEqual([])
    expect(sampleInspirations([makeItem()], { count: -1, rng })).toEqual([])
  })

  it('确定性：同池同 seed 逐位一致；无图无锚点时跨 seed 结果来自同一池', () => {
    const pool = [makeItem(), makeItem(), makeItem()]
    const run = () => sampleInspirations(pool, { count: 2, rng: mulberry32(99) })
    expect(run()).toEqual(run())
    for (const seed of [0, 1, 2, 3]) {
      const picked = sampleInspirations(pool, { count: 3, rng: mulberry32(seed) })
      expect(picked).toHaveLength(3)
      expect(new Set(picked.map((item) => item.id)).size).toBe(3)
    }
  })

  it('去重与上限：count 超池规模时返回全部池项且各只一次', () => {
    const pool = [makeItem(), makeItem(), makeItem(), makeItem()]
    const picked = sampleInspirations(pool, { count: 10, rng: mulberry32(5) })
    expect(picked).toHaveLength(4)
    expect(new Set(picked.map((item) => item.id)).size).toBe(4)
  })

  /** seeds 0..N-1 各采 1 条，统计首抽分布。 */
  function firstPickCounts(pool: readonly InspireItem[], seeds = 1000, opts?: Parameters<typeof sampleInspirations>[1]): Map<string, number> {
    const counts = new Map<string, number>()
    for (const seed of Array.from({ length: seeds }, (_, i) => i)) {
      const picked = sampleInspirations(pool, { count: 1, rng: mulberry32(seed), ...opts })
      counts.set(picked[0]!.id, (counts.get(picked[0]!.id) ?? 0) + 1)
    }
    return counts
  }

  it('冷门偏置：usedCount 0 远压过 usedCount 50（1 vs 1/51）', () => {
    const cold = makeItem({ id: 'cold', usedCount: 0 })
    const hot = makeItem({ id: 'hot', usedCount: 50 })
    const counts = firstPickCounts([cold, hot])
    expect(counts.get('cold')!).toBeGreaterThan(900)
  })

  it('red 加成：red=5 约为 red=0 的 6 倍权重', () => {
    const plain = makeItem({ id: 'plain', red: 0 })
    const linked = makeItem({ id: 'linked', red: 5 })
    const counts = firstPickCounts([plain, linked])
    expect(counts.get('linked')!).toBeGreaterThan(700)
  })

  it('topic 弱锚定：与 topic 同向者线性放大（×2 vs ×1）', () => {
    const aligned = makeItem({ id: 'aligned', embedding: [1, 0] })
    const other = makeItem({ id: 'other', embedding: [0, 1] })
    const counts = firstPickCounts([aligned, other], 1000, { topicVector: [1, 0] })
    expect(counts.get('aligned')!).toBeGreaterThan(600)
  })

  it('counter 臂 p=1：valence 与 mood 反号者占绝对多数', () => {
    const opposite = makeItem({ id: 'opposite', valence: 0.9 })
    const same = makeItem({ id: 'same', valence: -0.9 })
    const counts = firstPickCounts([opposite, same], 1000, {
      counterMood: { valence: -0.8, probability: 1 },
    })
    // 反号项权重 ×1.9（S 项对两者同乘 ×2），理论占比 ≈ 65%
    expect(counts.get('opposite')!).toBeGreaterThan(600)
  })

  it('counter 臂 p=0 与不传 counterMood 逐位一致（中性回退）', () => {
    const pool = [makeItem({ id: 'a', valence: 0.9 }), makeItem({ id: 'b', valence: -0.9 })]
    const neutral = sampleInspirations(pool, { count: 2, rng: mulberry32(11) })
    const zero = sampleInspirations(pool, {
      count: 2, rng: mulberry32(11), counterMood: { valence: -0.8, probability: 0 },
    })
    expect(zero.map((item) => item.id)).toEqual(neutral.map((item) => item.id))
  })

  it('counter 臂高 S 优先：反号候选里 S=10 压过 S=0（×2 vs ×1）', () => {
    const strong = makeItem({ id: 'strong', valence: 0.9, S: 10 })
    const weak = makeItem({ id: 'weak', valence: 0.9, S: 0 })
    const counts = firstPickCounts([strong, weak], 1000, {
      counterMood: { valence: -0.8, probability: 1 },
    })
    expect(counts.get('strong')!).toBeGreaterThan(600)
  })
})

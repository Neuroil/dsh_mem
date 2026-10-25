/**
 * 灵感通道（P3-4，设计 §10.1）：带种子的随机探索采样。
 *
 * 确定性契约（与 PPR 检索特征相反，随机性属于灵感通道本身）：
 * - seed = fnv1a(scope + 日期 + 查询锚点)——同作用域同日同锚点可复现
 *   （同一话题一天内不换灵感），跨日漂移自带间隔节奏（无定时器）；
 * - mulberry32 PRNG：给定 seed 输出序列完全确定，测试可复现。
 *
 * 采样器 = 图上带偏随机游走：权重偏向低 used_count / 高 red（冷门但语义
 * 网络中心的候选），topic 弱锚定（余弦 ≥ 0 线性放大，不硬过滤）；counter
 * 情绪对冲与 wild-card 同机制换分布——以概率切换到「valence 与当前 mood
 * 相反的高 S」分布。
 */
import { buildAdjacency } from './graph.js'
import { cosine } from './metrics.js'
import { fnv1a } from './scope.js'
import type { InspireItem } from './types.js'

/**
 * mulberry32：32 位种子的确定性 PRNG，输出 [0,1)。足够小的周期与均匀性
 * 对"换一个灵感"的用途绰绰有余，且实现短到可以逐行审计。
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 本地日期键（YYYY-MM-DD）：seed 的漂移分量，跨日自然换灵感。 */
export function localDateKey(now: Date): string {
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${now.getFullYear()}-${month}-${day}`
}

/** 灵感种子 = fnv1a(scope ⊕ 日期 ⊕ 查询锚点)；无锚点（纯随机）用空串占位。 */
export function inspireSeed(scopeKey: string | undefined, dateKey: string, anchor: string): number {
  return fnv1a(`${scopeKey ?? '_'}\n${dateKey}\n${anchor}`)
}

/** 游走每跳的继续概率（工程常数：期望步长 2 跳，同 MOOD_EMA_WEIGHT 的处理不进 Config）。 */
const WALK_CONTINUE_PROB = 0.5
/** 游走步数硬上限（防病态长游走；期望步长下极少触达）。 */
const WALK_MAX_HOPS = 3

export interface InspireOptions {
  /** 采样条数（池小于该值时返回全部池项）。 */
  readonly count: number
  /** 种子化 PRNG（mulberry32(seed)）。 */
  readonly rng: () => number
  /** 弱锚定向量（topic 嵌入）：余弦 ≥ 0 线性放大权重，不硬过滤。缺省 = 纯随机。 */
  readonly topicVector?: readonly number[]
  /** counter 情绪对冲：每次抽样以 probability 切换到反向 valence 高 S 分布。 */
  readonly counterMood?: { readonly valence: number; readonly probability: number }
}

/** 加权抽取：累计权重二分不需（池规模小），线性扫即可；全零权重均匀回退。 */
function pickWeighted<T>(pool: readonly T[], weight: (item: T) => number, rng: () => number): T {
  let total = 0
  for (const item of pool) total += Math.max(0, weight(item))
  if (total <= 0) return pool[Math.floor(rng() * pool.length)]!
  let roll = rng() * total
  for (const item of pool) {
    roll -= Math.max(0, weight(item))
    if (roll < 0) return item
  }
  return pool[pool.length - 1]!
}

/**
 * 图上带偏随机游走采样（P3-4）：不重复地采 count 条。
 *
 * 每次抽取两臂：
 * - 常规臂（wild-card）：全池按 baseWeight = 1/(1+usedCount) · (1+red) · (1+max(0,topicCos))
 *   加权起跳，再以 WALK_CONTINUE_PROB 概率向邻居迈步（邻居同样按 baseWeight 偏置）；
 * - counter 臂（情绪对冲，probability 概率触发）：全池按
 *   1/(1+usedCount) · (1+S/maxS) · (1+oppositeValence) 加权——valence 与当前
 *   mood 反号者线性放大，高 S 优先。
 */
export function sampleInspirations(items: readonly InspireItem[], opts: InspireOptions): InspireItem[] {
  if (items.length === 0 || opts.count <= 0) return []
  const adjacency = buildAdjacency(items)
  const topicCos = new Map<string, number>()
  if (opts.topicVector !== undefined) {
    for (const item of items) topicCos.set(item.id, cosine(opts.topicVector, item.embedding))
  }
  const baseWeight = (item: InspireItem): number =>
    (1 / (1 + item.usedCount)) * (1 + item.red) * (1 + Math.max(0, topicCos.get(item.id) ?? 0))

  let counterWeight: ((item: InspireItem) => number) | undefined
  if (opts.counterMood !== undefined) {
    const moodValence = opts.counterMood.valence
    const maxS = Math.max(1, ...items.map((item) => item.S))
    counterWeight = (item: InspireItem): number => {
      const opposite =
        item.valence !== undefined && moodValence !== 0 && Math.sign(item.valence) === -Math.sign(moodValence)
          ? Math.abs(item.valence)
          : 0
      return (1 / (1 + item.usedCount)) * (1 + item.S / maxS) * (1 + opposite)
    }
  }

  const remaining = new Map(items.map((item) => [item.id, item]))
  const picked: InspireItem[] = []
  while (picked.length < Math.min(opts.count, items.length)) {
    const pool = [...remaining.values()]
    let item: InspireItem
    if (counterWeight !== undefined && opts.rng() < opts.counterMood!.probability) {
      item = pickWeighted(pool, counterWeight, opts.rng)
    } else {
      item = pickWeighted(pool, baseWeight, opts.rng)
      let hops = 0
      while (hops < WALK_MAX_HOPS && opts.rng() < WALK_CONTINUE_PROB) {
        const neighbors = [...(adjacency.get(item.id) ?? [])]
          .filter((id) => remaining.has(id))
          .map((id) => remaining.get(id)!)
        if (neighbors.length === 0) break
        item = pickWeighted(neighbors, baseWeight, opts.rng)
        hops++
      }
    }
    picked.push(item)
    remaining.delete(item.id)
  }
  return picked
}

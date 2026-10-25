/**
 * 语义图边工具（P3-2，设计 §10.1 P3-2 包）：`related` 边的三个生产者共用
 * 的纯函数——去重追加、度数上限（防 hub 复发：CSLS 治理过的 hub 现象会在
 * 图上重现）、密度统计（决定 P3-3 PPR 何时有意义）。
 *
 * 图的边来源（全景图 §7.6/§5.2）：`supersedes`/`complements`（矛盾判定，
 * P0 起）+ `related`（P3-2 三件套：聚类建边 / 矛盾调用扩产出 / 同义边）+
 * 节点 `member_refs`（成员关系边，P1 起——图遍历侧由邻接构建读取）。
 */
import type { MemoryLink, MemoryRecord, SemanticNode } from './types.js'

/** 去重追加一条边（kind+target 相同视为已存在）；返回新数组，原数组不动。 */
export function addLink(links: readonly MemoryLink[], link: MemoryLink): MemoryLink[] {
  if (links.some((l) => l.kind === link.kind && l.target === link.target)) return [...links]
  return [...links, link]
}

/** 记录/节点当前 related 边数（度数上限只约束 related，supersedes/complements 不占额度）。 */
export function relatedDegree(links: readonly MemoryLink[]): number {
  let n = 0
  for (const l of links) if (l.kind === 'related') n++
  return n
}

/**
 * 受度数上限约束地批量补 related 边：目标按入参顺序消费额度，超出上限的
 * 目标跳过（不截断整批）。返回（可能未变的）新数组。
 */
export function addRelatedCapped(
  links: readonly MemoryLink[],
  targets: readonly string[],
  maxDegree: number,
): MemoryLink[] {
  let out = [...links]
  for (const target of targets) {
    if (relatedDegree(out) >= maxDegree) break
    out = addLink(out, { kind: 'related', target })
  }
  return out
}

/** 图密度统计：边数（links 全部种类 + member_refs 成员边）/ 参与项数 / 平均度。 */
export function graphStats(
  records: readonly MemoryRecord[],
  nodes: readonly SemanticNode[],
): { items: number; edges: number; avgDegree: number } {
  let edges = 0
  for (const r of records) edges += r.links.length
  for (const n of nodes) edges += n.links.length + n.member_refs.length
  const items = records.length + nodes.length
  return { items, edges, avgDegree: items === 0 ? 0 : (2 * edges) / items }
}

// ---- P3-3：邻接构建与 PPR ----

/** 池内项的最小结构（RetrievableItem 满足；neighbors = links 目标 + 节点 member_refs）。 */
export interface AdjacentItem {
  readonly id: string
  /** 邻居 id 列表（可能含池外/软失效 id——构建时过滤）。 */
  readonly neighbors?: readonly string[]
}

/**
 * 无向邻接表：只连**池内**的边（池 = 检索宇宙：有效未压缩记录 + 有效节点；
 * 软失效/归档出池的 id 自动被 `ids.has` 过滤，图遍历天然不越池）。
 */
export function buildAdjacency(items: readonly AdjacentItem[]): Map<string, Set<string>> {
  const ids = new Set(items.map((i) => i.id))
  const adjacency = new Map<string, Set<string>>()
  for (const item of items) adjacency.set(item.id, new Set<string>())
  for (const item of items) {
    const own = adjacency.get(item.id)!
    for (const target of item.neighbors ?? []) {
      if (target === item.id || !ids.has(target)) continue
      own.add(target)
      adjacency.get(target)!.add(item.id)
    }
  }
  return adjacency
}

/**
 * Personalized PageRank（幂迭代，确定性：Map 按插入序遍历）。
 * seed 权重归一化为分布；悬挂质量均匀回撒；收敛于容差或 maxIterations。
 * 返回空 Map 当且仅当图为空或 seed 全不在图中（调用方据此精确跳过 PPR）。
 */
export function personalizedPagerank(
  adjacency: ReadonlyMap<string, ReadonlySet<string>>,
  seeds: ReadonlyMap<string, number>,
  damping: number,
  maxIterations = 20,
  tolerance = 1e-8,
): Map<string, number> {
  const nodes = [...adjacency.keys()]
  const seedTotal = [...seeds.values()].reduce((a, b) => a + b, 0)
  const result = new Map<string, number>()
  if (nodes.length === 0 || seedTotal <= 0) return result
  const seedDist = new Map<string, number>()
  let inGraph = 0
  for (const n of nodes) seedDist.set(n, 0)
  for (const [id, weight] of seeds) {
    if (seedDist.has(id)) {
      seedDist.set(id, weight / seedTotal)
      inGraph += weight / seedTotal
    }
  }
  if (inGraph <= 0) return result
  let rank = new Map(seedDist)
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const next = new Map<string, number>()
    for (const n of nodes) next.set(n, 0)
    let dangling = 0
    for (const n of nodes) {
      const current = rank.get(n)!
      const out = adjacency.get(n)!
      if (out.size === 0) {
        dangling += current
        continue
      }
      const share = current / out.size
      for (const neighbor of out) next.set(neighbor, next.get(neighbor)! + share)
    }
    const danglingShare = dangling / nodes.length
    let delta = 0
    for (const n of nodes) {
      const value =
        (1 - damping) * (seedDist.get(n) ?? 0) +
        damping * ((next.get(n) ?? 0) + danglingShare)
      next.set(n, value)
      delta += Math.abs(value - rank.get(n)!)
    }
    rank = next
    if (delta < tolerance) break
  }
  return rank
}

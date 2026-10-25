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

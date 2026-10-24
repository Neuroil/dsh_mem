/**
 * 精确 kNN 向量索引（§3.1 L0 召回的 P0 实现）：
 * 写入时归一化、查询时暴力扫全量；接口为后续替换 ANN 留出空间。
 */
import { cosine, normalize } from './metrics.js'
import type { Neighbor, VectorIndex } from './types.js'

/** id 字典序比较，作为余弦平局的确定性决胜。 */
function compareId(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

export function createVectorIndex(): VectorIndex {
  const vectors = new Map<string, number[]>()
  return {
    add(id, vector) {
      vectors.set(id, normalize(vector))
    },
    remove(id) {
      return vectors.delete(id)
    },
    size() {
      return vectors.size
    },
    search(query, k, filter) {
      const hits: Neighbor[] = []
      for (const [id, vector] of vectors) {
        if (filter !== undefined && !filter(id)) continue // 过滤发生在排序之前
        hits.push({ id, cos: cosine(query, vector) })
      }
      hits.sort((a, b) => b.cos - a.cos || compareId(a.id, b.id))
      return hits.slice(0, Math.max(0, k))
    },
  }
}

/**
 * 检索管线与睡眠周期用到的纯数学工具（设计 §3.1–§3.3、§2.3、§3.6）：
 * 余弦相似度 / L2 归一化、CSLS hubness 校正、MMR 多样性选择、
 * z 标准化、sigmoid、QA 闸门 token 级 F1。
 * 不依赖宿主框架与其他 src 模块，可独立单测。
 */

/** 余弦相似度；任一零向量返回 0（方向未定义，按不相似处理）。 */
export function cosine(a: readonly number[], b: readonly number[]): number {
  let normA = 0
  let normB = 0
  for (const x of a) normA += x * x
  for (const x of b) normB += x * x
  if (normA === 0 || normB === 0) return 0
  let dot = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0)
  }
  return dot / Math.sqrt(normA * normB)
}

/** L2 归一化；零向量（范数为 0）无方向可言，按原值返回。 */
export function normalize(v: readonly number[]): number[] {
  let normSq = 0
  for (const x of v) normSq += x * x
  if (normSq === 0) return [...v]
  const inv = 1 / Math.sqrt(normSq)
  return v.map(x => x * inv)
}

/** CSLS 校正后的线索区分度（§3.2）：cue(q,m) = 2·cos(q,m) − r(q) − r(m)。 */
export function cslsCue(cosQM: number, rQ: number, rM: number): number {
  return 2 * cosQM - rQ - rM
}

/** 算术均值；空数组返回 0（如候选池为空时的 r(q)）。 */
export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0
  let sum = 0
  for (const x of values) sum += x
  return sum / values.length
}

/** 总体标准差；空数组返回 0。z 标准化的分母（§3.4/§3.6）。 */
export function std(values: readonly number[]): number {
  if (values.length === 0) return 0
  const mu = mean(values)
  let sumSq = 0
  for (const x of values) sumSq += (x - mu) * (x - mu)
  return Math.sqrt(sumSq / values.length)
}

/** 逻辑函数 σ(x)（§3.6 epi 合成的外层）。 */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

/** z 分数：除以 max(σ, floor) 防 0 除；空分布（μ=σ=0）时返回 0。 */
export function zScore(value: number, values: readonly number[], sigmaFloor = 1e-9): number {
  if (values.length === 0) return 0
  return (value - mean(values)) / Math.max(std(values), sigmaFloor)
}

/**
 * QA 闸门 token 级 F1（§2.3 第 3 步，QAFactEval 式）：
 * 小写化后 Han 字按单字、其余按字母数字连串切 token（中英混合均可），
 * 多重集交叠算 precision/recall；任一侧无 token 返回 0。
 */
export function qaF1(reference: string, candidate: string): number {
  const refTokens = reference.toLowerCase().match(/[\p{Script=Han}]|[a-z0-9]+/gu) ?? []
  const candTokens = candidate.toLowerCase().match(/[\p{Script=Han}]|[a-z0-9]+/gu) ?? []
  if (refTokens.length === 0 || candTokens.length === 0) return 0
  const counts = new Map<string, number>()
  for (const token of refTokens) counts.set(token, (counts.get(token) ?? 0) + 1)
  let overlap = 0
  for (const token of candTokens) {
    const left = counts.get(token) ?? 0
    if (left > 0) {
      overlap++
      counts.set(token, left - 1)
    }
  }
  const precision = overlap / candTokens.length
  const recall = overlap / refTokens.length
  return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall)
}

/**
 * MMR 贪心选择（§3.3）：argmax [ λ·R(m) − (1−λ)·max_{s∈已选} sim(m,s) ]。
 * 首轮（已选为空）max 项视为 0；同分取先遍历者，输入序确定则结果确定。
 * 返回按选择顺序排列的前 k 项。
 */
export function mmrSelect<T>(items: readonly T[], k: number, lambda: number,
  relevance: (item: T) => number,
  similarity: (a: T, b: T) => number): T[] {
  const pool = [...items]
  const selected: T[] = []
  while (selected.length < k && pool.length > 0) {
    let chosen: T | undefined
    let bestScore = -Infinity
    for (const candidate of pool) {
      // 已选为空时 max 项为 0；否则取真实最大值（允许为负）
      let maxSim = selected.length === 0 ? 0 : -Infinity
      for (const s of selected) {
        maxSim = Math.max(maxSim, similarity(candidate, s))
      }
      const score = lambda * relevance(candidate) - (1 - lambda) * maxSim
      if (score > bestScore) {
        bestScore = score
        chosen = candidate
      }
    }
    if (chosen === undefined) break
    selected.push(chosen)
    pool.splice(pool.indexOf(chosen), 1)
  }
  return selected
}

/**
 * 离线巩固周期（"睡眠"，设计 §2.3 / §5.3）：
 *
 *   1. 聚类：pairwise 余弦 ≥ clusterCos 的有效未压缩记忆构成连通分量，
 *      簇大小 ≥ clusterMinSize 且簇内平均 pairwise 余弦 ≥ clusterCos（§3.3 压缩触发）
 *   2. gist 抽取：LLM 对簇摘要抽象
 *   3. QA 闸门：从簇生成 QA 对、仅以 gist 作答，平均 token 级 F1 ≥ qaF1Threshold
 *      才通过；不通过 → 保留原簇并标记"不可压缩"
 *   4. M4 路由（§3.4）：con = gist 对语义网络峰值余弦的 z 分数，
 *      挂入已有节点（+LLM 兼容性复核）或建新节点（imp 门控）；其余留情景层
 *   5. M6 评估（§3.6）：gist 的完整 epi（z 词法特征 − z(G 探针相关度)），
 *      低于 epiThreshold 才迁移；epi 下降量 = "细节遗忘量"的量化
 *   6. 迁移：建/挂 SemanticNode，成员 flags.compressed_into 指向节点并退出检索
 *   7. 刷新 r_m / red / con 统计（检索宇宙 = 有效未压缩记录 + 有效语义节点）
 *
 * 触发由 runtime 的 turn/end 信号门控（§5.3，minNewMemories 计数）；本模块
 * 只做批处理本身。全部依赖注入（InternalLlm / EmbeddingProvider / 窄存储接口），
 * 不依赖宿主框架，可离线单测。
 */
import type { Config } from './config.js'
import {
  answerFromGist,
  extractGist,
  generateQaPairs,
  judgeNodeCompatibility,
  lexicalDensities,
} from './judges.js'
import { cosine, mean, qaF1, sigmoid, zScore } from './metrics.js'
import { applyDecay, applyNodeDecay } from './scoring.js'
import type {
  EmbeddingProvider,
  InternalLlm,
  MemoryRecord,
  SemanticNode,
  SleepClusterOutcome,
  SleepOutcome,
} from './types.js'

/** 存储层最小依赖（由 MemoryStore 结构满足；窄接口保持可测）。 */
export interface SleepStoreLike {
  validMemories(): readonly MemoryRecord[]
  validNodes(): readonly SemanticNode[]
  updateRecord(id: string, fn: (r: MemoryRecord) => MemoryRecord): Promise<MemoryRecord>
  putNode(node: SemanticNode): Promise<void>
  refreshNeighborhood(id: string, k: number): Promise<{ r_m: number; red: number }>
  /** 刷新语义节点 r_m（第 7 步），返回新值。 */
  refreshNodeNeighborhood(id: string, k: number): Promise<number>
}

/**
 * §3.6 G(m) 的预置多样化探针集（|P| = 20，设计固定 20 条）。与 epiLexical 的
 * 词表同属度量工具的固定常量，非可调超参：语义记忆对多样问题都有中等相关
 * （gist 通用），情景记忆只对少数特定查询尖峰相关。
 */
export const M6_PROBES: readonly string[] = [
  'What does the user prefer in general?',
  'What is the user\u2019s job or team?',
  'What tools or software does the user use?',
  'What decisions have been made?',
  'What problems keep recurring?',
  'What is the user working on these days?',
  'What does the user like to do in free time?',
  'What should I know about the user\u2019s health?',
  'Who does the user work or live with?',
  'What are the user\u2019s long-term goals?',
  'What did we discuss about databases?',
  'What did we discuss about deployment or infrastructure?',
  'What coding habits does the user have?',
  'What meetings or plans were scheduled?',
  'What did the user buy or plan to buy?',
  'Where does the user live or travel?',
  'What opinions does the user hold?',
  'What is currently broken or being fixed?',
  'What did we discuss about the team or company?',
  'What general lessons came out of past incidents?',
]

// ---- 聚类（§2.3 第 1 步） ----

/** 一个压缩候选簇：成员（按 id 序）与簇内平均 pairwise 余弦。 */
export interface Cluster {
  readonly members: readonly MemoryRecord[]
  readonly meanPairwiseCos: number
}

/** id 字典序，平局时的确定性决胜。 */
function compareId(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

/**
 * 连通分量聚类：pairwise 余弦 ≥ clusterCos 的记忆连边；分量需同时满足
 * 簇大小 ≥ clusterMinSize、簇内平均 pairwise 余弦 ≥ clusterCos（链式传导被
 * 平均值闸门拦截），且至少一名成员未标"不可压缩"（全簇已标 = 上一轮闸门
 * 拒收，需新成员加入才值得重试，§2.3 第 3 步）。
 */
export function buildClusters(pool: readonly MemoryRecord[], config: Config['sleep']): readonly Cluster[] {
  const sorted = [...pool].sort((a, b) => compareId(a.id, b.id))
  const parent = sorted.map((_, i) => i)
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i] ?? 0] ?? 0
      i = parent[i] ?? 0
    }
    return i
  }
  const union = (a: number, b: number): void => {
    parent[find(a)] = find(b)
  }
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      if (cosine(sorted[i]!.embedding, sorted[j]!.embedding) >= config.clusterCos) union(i, j)
    }
  }
  const components = new Map<number, MemoryRecord[]>()
  for (const [i, record] of sorted.entries()) {
    const root = find(i)
    const bucket = components.get(root)
    if (bucket === undefined) components.set(root, [record])
    else bucket.push(record)
  }
  const clusters: Cluster[] = []
  for (const members of components.values()) {
    if (members.length < config.clusterMinSize) continue
    // 平均 pairwise 余弦取分量内全部 C(n,2) 对（含低于边阈值的对，链式传导由此拦截）
    let sum = 0
    let pairs = 0
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        sum += cosine(members[i]!.embedding, members[j]!.embedding)
        pairs++
      }
    }
    const meanPairwiseCos = sum / pairs
    if (meanPairwiseCos < config.clusterCos) continue
    if (!members.some((m) => m.flags.incompressible !== true)) continue
    clusters.push({ members, meanPairwiseCos })
  }
  return clusters.sort((a, b) => compareId(a.members[0]!.id, b.members[0]!.id))
}

// ---- M4 con（§3.4） ----

/** 对语义网络的峰值余弦；平局取 id 字典序最小者（确定性）。 */
function maxNodeCos(vector: readonly number[], nodes: readonly SemanticNode[]): { cos: number; node: SemanticNode } {
  const sorted = [...nodes].sort((a, b) => compareId(a.id, b.id))
  let best = sorted[0]!
  let bestCos = cosine(vector, best.embedding)
  for (const node of sorted.slice(1)) {
    const cos = cosine(vector, node.embedding)
    if (cos > bestCos) {
      bestCos = cos
      best = node
    }
  }
  return { cos: bestCos, node: best }
}

export interface ConResult {
  /** z 分数；语义网络为空时由调用方走引导路径，不会调到这里。 */
  readonly con: number
  /** 峰值余弦对应的节点（挂入候选）。 */
  readonly attachTo: SemanticNode
}

/**
 * con = z( gist 对语义网络的峰值余弦 )，基线 = 最近 conWindow 条有效未压缩
 * 情景记忆各自对网络的峰值余弦分布（§3.4"相对近期记忆总体的滚动分布"）。
 * σ 加下限防 0 除；基线为空时 con = 0（中性 → 留情景层）。
 */
export function conAgainstBaseline(
  gistVector: readonly number[],
  nodes: readonly SemanticNode[],
  baselinePool: readonly MemoryRecord[],
  conWindow: number,
): ConResult {
  const gistMax = maxNodeCos(gistVector, nodes)
  const window = [...baselinePool]
    .sort((a, b) => b.time_meta.created_at - a.time_meta.created_at || compareId(a.id, b.id))
    .slice(0, conWindow)
  const values = window.map((record) => maxNodeCos(record.embedding, nodes).cos)
  return { con: zScore(gistMax.cos, values), attachTo: gistMax.node }
}

// ---- M6 完整 epi（§3.6） ----

/** 平均探针相关度 G（§3.6，仅簇代表/基线分布上算）。 */
function probeAffinity(vector: readonly number[], probeVectors: readonly (readonly number[])[]): number {
  return mean(probeVectors.map((probe) => cosine(vector, probe)))
}

/**
 * gist 的完整 M6 epi = σ( z(实体密度) + z(时间表达密度) − z(G) )，a=b=c=1
 * （设计固定系数）。三个特征都相对"当前有效未压缩情景记忆"的分布做 z 标准化
 * ——G 也 z 化：裸余弦绝对值跨模型不稳定（§3.4 同理），z 化后 epiThreshold
 * 才有尺度不变的含义。z 标准化的合成依据 §3.6"特征先 z 标准化再合成"。
 */
export function gistEpiScore(
  gist: string,
  gistVector: readonly number[],
  pool: readonly MemoryRecord[],
  probeVectors: readonly (readonly number[])[],
): number {
  const poolTime: number[] = []
  const poolEntity: number[] = []
  const poolG: number[] = []
  for (const record of pool) {
    const densities = lexicalDensities(record.text)
    poolTime.push(densities.time)
    poolEntity.push(densities.entity)
    poolG.push(probeAffinity(record.embedding, probeVectors))
  }
  const gistDensities = lexicalDensities(gist)
  const logit =
    zScore(gistDensities.entity, poolEntity) +
    zScore(gistDensities.time, poolTime) -
    zScore(probeAffinity(gistVector, probeVectors), poolG)
  return sigmoid(logit)
}

// ---- 周期编排 ----

export interface SleepDeps {
  readonly llm: InternalLlm
  readonly embedding: EmbeddingProvider
  readonly store: SleepStoreLike
  readonly config: Config
  readonly now?: () => number
  /** 节点 id 生成器（默认 crypto.randomUUID；测试注入确定性序号）。 */
  readonly newId?: () => string
  readonly log?: (msg: string) => void
}

export class SleepCycle {
  private readonly deps: SleepDeps

  constructor(deps: SleepDeps) {
    this.deps = deps
  }

  /**
   * 执行一个完整周期。单簇失败不中断整轮（signal 中止除外）；统计刷新
   * （第 7 步）总是执行——即使本轮无簇可压缩，新写入也漂移了邻域统计
   * （§3.2/§3.3 的"夜间刷新"）。
   */
  async run(signal?: AbortSignal): Promise<SleepOutcome> {
    const { config } = this.deps
    // 聚类池 = 有效未压缩情景记忆；按 id 序保证确定性
    const pool = this.deps.store
      .validMemories()
      .filter((r) => r.flags.compressed_into === undefined)
      .sort((a, b) => compareId(a.id, b.id))
    const clusters = buildClusters(pool, config.sleep)

    let probeVectors: readonly (readonly number[])[] = []
    const outcomes: SleepClusterOutcome[] = []
    if (clusters.length > 0) {
      const embedded = await this.deps.embedding.embed(M6_PROBES, signal)
      probeVectors = embedded
    }
    for (const cluster of clusters) {
      try {
        outcomes.push(await this.processCluster(cluster, pool, probeVectors, signal))
      } catch (err) {
        if (signal?.aborted) throw err
        this.deps.log?.(`sleep-cycle: 簇处理失败（${cluster.members[0]!.id} 等 ${cluster.members.length} 条）：${String(err)}`)
      }
    }

    // 第 7 步：兜底衰减（§2.1，P2）+ 刷新 r_m / red（留情景层者）与节点 r_m
    // （检索宇宙内）。衰减只作用于仍在检索宇宙的记录/节点（软失效与已压缩
    // 归档者的 S 是惰性值，不参与分层）；衰减引起的层级跃迁不触发 M5 重判。
    const now = this.deps.now ?? Date.now
    const remaining = this.deps.store
      .validMemories()
      .filter((r) => r.flags.compressed_into === undefined)
    for (const record of remaining) {
      await this.deps.store.updateRecord(record.id, (r) => applyDecay(r, now(), this.deps.config))
      await this.deps.store.refreshNeighborhood(record.id, config.retrieval.cslsK)
    }
    const refreshedNodes: string[] = []
    for (const node of this.deps.store.validNodes()) {
      await this.deps.store.putNode(applyNodeDecay(node, now(), this.deps.config))
      await this.deps.store.refreshNodeNeighborhood(node.id, config.retrieval.cslsK)
      refreshedNodes.push(node.id)
    }

    return {
      clusters: outcomes,
      refreshedRecords: remaining.length,
      refreshedNodes,
      decayedRecords: remaining.length,
      decayedNodes: refreshedNodes.length,
    }
  }

  /**
   * 单簇七步（M6 闸门前置于 M4：两闸独立，纯计算的 M6 先行可省一次
   * LLM 兼容性复核调用，语义等价——两闸都过才迁移）。
   */
  private async processCluster(
    cluster: Cluster,
    pool: readonly MemoryRecord[],
    probeVectors: readonly (readonly number[])[],
    signal?: AbortSignal,
  ): Promise<SleepClusterOutcome> {
    const { config } = this.deps
    const cfg = config.sleep
    const now = this.deps.now ?? Date.now
    const members = cluster.members
    const memberIds = members.map((m) => m.id)
    const memberMeanEpi = mean(members.map((m) => m.epi))
    const memberMeanImp = mean(members.map((m) => m.imp))

    // 第 2 步：gist 抽取
    const gist = await extractGist(this.deps.llm, members.map((m) => m.text), signal)
    const gistVectors = await this.deps.embedding.embed([gist], signal)
    const gistVector = gistVectors[0]
    if (gistVector === undefined) {
      throw new Error(`gist 嵌入缺口（簇 ${members[0]!.id}）`)
    }

    // M6 epi（纯计算，先算好供所有分支报告）：gist 的语义化进度，§3.6
    const gistEpi = gistEpiScore(gist, gistVector, pool, probeVectors)

    // 第 3 步：QA 闸门 —— 从簇生成 QA 对，仅以 gist 作答，平均 F1 达标才放行
    const pairs = await generateQaPairs(this.deps.llm, members.map((m) => m.text), cfg.qaPairCount, signal)
    const f1s: number[] = []
    for (const pair of pairs) {
      const answer = await answerFromGist(this.deps.llm, gist, pair.question, signal)
      f1s.push(qaF1(pair.answer, answer))
    }
    const qaMean = f1s.length === 0 ? 0 : mean(f1s)
    if (pairs.length === 0 || qaMean < cfg.qaF1Threshold) {
      await this.markIncompressible(memberIds)
      return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'qa_failed' }
    }

    // 第 5 步（M6）：gist epi 够低才迁移；否则视为不可压缩（摘要未去情景化）
    if (gistEpi >= cfg.epiThreshold) {
      await this.markIncompressible(memberIds)
      return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'epi_failed' }
    }

    // 第 4 步（M4 路由）+ 第 6 步（迁移）
    const nodes = this.deps.store.validNodes()
    if (nodes.length === 0) {
      // 空网络引导：无条件建首节点，否则语义层永不为空（P1 政策，设计未定义空网情形）
      const nodeId = await this.createNode(gist, gistVector, gistEpi, memberIds, memberMeanImp, now())
      await this.compressMembers(memberIds, nodeId, 'new_node', now())
      return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'new_node', nodeId }
    }
    const { con, attachTo } = conAgainstBaseline(gistVector, nodes, pool, cfg.conWindow)
    if (con > cfg.conAttachSigma) {
      // 挂入需 LLM 兼容性复核（§3.4 表第 1 行）；不兼容 → 留情景层等下轮
      const compatible = await judgeNodeCompatibility(this.deps.llm, gist, attachTo.gist, signal)
      if (compatible) {
        await this.deps.store.putNode({
          ...attachTo,
          member_refs: [...attachTo.member_refs, ...memberIds],
          S: attachTo.S + cfg.attachGain, // §2.1 α 沿用的强化增益；使用/抑制/衰减动态见 scoring.ts
        })
        await this.compressMembers(memberIds, attachTo.id, 'attached', now())
        return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'attached', nodeId: attachTo.id }
      }
      return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'stayed' }
    }
    if (con < cfg.conNewNodeSigma && memberMeanImp >= cfg.conNewNodeMinImp) {
      const nodeId = await this.createNode(gist, gistVector, gistEpi, memberIds, memberMeanImp, now())
      await this.compressMembers(memberIds, nodeId, 'new_node', now())
      return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'new_node', nodeId }
    }
    // 其余：留情景层，等下个巩固周期（§3.4 表第 3 行）
    return { members: memberIds, gist, qaF1: qaMean, gistEpi, memberMeanEpi, route: 'stayed' }
  }

  /** §2.3 第 3 步：闸门拒收 → 全簇标记不可压缩（记录保留、永不删除）。 */
  private async markIncompressible(memberIds: readonly string[]): Promise<void> {
    for (const id of memberIds) {
      await this.deps.store.updateRecord(id, (r) => ({
        ...r,
        flags: { ...r.flags, incompressible: true },
      }))
    }
  }

  /** §2.3 第 6 步：建新语义节点（S = 簇均 imp，与记录"S 初始化为 imp"一致）。 */
  private async createNode(
    gist: string,
    gistVector: readonly number[],
    epi: number,
    memberIds: readonly string[],
    memberMeanImp: number,
    at: number,
  ): Promise<string> {
    // 注意不能写 (this.deps.newId ?? crypto.randomUUID)：摘出方法裸调用会丢 this（Node 下 ERR_INVALID_THIS）
    const id = this.deps.newId !== undefined ? this.deps.newId() : crypto.randomUUID()
    await this.deps.store.putNode({
      id,
      gist,
      embedding: gistVector,
      epi,
      links: [],
      member_refs: memberIds,
      time_meta: { created_at: at, valid_at: at },
      S: memberMeanImp,
      r_m: 0, // 第 7 步统一刷新
    })
    return id
  }

  /** §2.3 第 6 步：成员压缩归档 —— 指向 gist、记录归档时点（时间旅行视图消费）、退出检索，仅保留回溯索引。 */
  private async compressMembers(
    memberIds: readonly string[],
    nodeId: string,
    routeKind: 'new_node' | 'attached',
    at: number,
  ): Promise<void> {
    for (const id of memberIds) {
      await this.deps.store.updateRecord(id, (r) => ({
        ...r,
        time_meta: { ...r.time_meta, compressed_at: at },
        flags: { ...r.flags, compressed_into: nodeId, qa_passed: true },
        con_route:
          routeKind === 'new_node'
            ? { kind: 'new_node', nodeId }
            : { kind: 'attached', nodeIds: [nodeId] },
      }))
    }
  }
}

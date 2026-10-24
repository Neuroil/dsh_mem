/**
 * 存储层（设计 §5.0 存储契约、§4 数据模型、§2.5 软失效）。
 *
 * 一切持久化经宿主 KvUnit（ctx.storage.backend.get(name).kv.open(descriptor)），
 * 不自建数据库或裸文件：表 = episodic（MemoryRecord，§4.1）+ semantic
 * （SemanticNode，§4.2），键 = 记录 id，无 global 槽。
 *
 * 写入顺序统一为「先持久化、后改内存镜像」：KvUnit 的单次调用在 resolve 时
 * 即已原子且持久（崩溃后重开可观察到该写），因此镜像只在持久化成功后更新，
 * 二者不会出现"镜像有、介质无"的偏离。
 */
import type { KvUnit, KvUnitDescriptor } from '@deepseek-ai/dsh-storage'
import { UNIT_NAME_RE } from './config.js'
import type { Config } from './config.js'
import type {
  MemoryRecord,
  MemoryTier,
  Neighbor,
  RetrievableItem,
  SemanticNode,
  VectorIndex,
} from './types.js'

/**
 * KV unit 描述符：表 = episodic（MemoryRecord）+ semantic（SemanticNode），无 global 槽（§5.0）。
 * unit 名由调用方传入（M2：工作区 unit 名由 scope.ts 派生，回退用 config.storage.unit），
 * 在此归一化并本地校验：连字符归一为下划线（宿主不允许连字符），归一后仍不满足
 * UNIT_NAME_RE 的名字在此抛错——让不合规命名在启动期暴露，而不是等宿主
 * validateDescriptor 抛 malformed-medium 后被 apply() 的 catch 吞掉。
 */
export function memoryUnitDescriptor(unitName: string, config: Config): KvUnitDescriptor {
  const name = unitName.replace(/-/g, '_')
  if (!UNIT_NAME_RE.test(name)) {
    throw new Error(`invalid storage unit name '${unitName}' (normalized '${name}'): ` +
      `must match ${UNIT_NAME_RE} — lowercase letters/digits/underscores, no hyphens`)
  }
  return {
    name,
    version: config.storage.version,
    tables: ['episodic', 'semantic'],
    hasGlobal: false,
  }
}

/** 可及性分层（§2.1）：S ≥ hotThreshold → hot；S < coldThreshold → cold；否则 warm。阈值全部来自 Config。 */
export function tierOf(S: number, tier: Config['tier']): MemoryTier {
  if (S >= tier.hotThreshold) return 'hot'
  if (S < tier.coldThreshold) return 'cold'
  return 'warm'
}

/** 近邻检索命中项：id + 余弦 + 完整记录。 */
export interface SimilarMemory {
  readonly id: string
  readonly cos: number
  readonly record: MemoryRecord
}

/** 有效语义节点的近邻命中项（支柱五完整版：gist 被新记忆推翻的安全网）。 */
export interface SimilarNodeHit {
  readonly id: string
  readonly cos: number
  readonly node: SemanticNode
}

/** episodic 表名（MemoryRecord）。 */
const TABLE_EPISODIC = 'episodic'
/** semantic 表名（SemanticNode）。 */
const TABLE_SEMANTIC = 'semantic'
/**
 * red 的近邻宽度（§3.3：red(m) = m 与记忆库 top-5 近邻的平均余弦）。
 * 这是设计文档固定公式中的常数（如同 MMR 的 top-5 输出），Config 未将其参数化。
 */
const RED_NEIGHBOR_TOP = 5

/** 记录是否仍有效（§2.5）：time_meta.invalid_at 未设即有效；时间戳只是元数据。 */
function isValidRecord(record: MemoryRecord): boolean {
  return record.time_meta?.invalid_at === undefined
}

/** 语义节点是否仍有效（§4.2 同款双时间戳语义）。 */
function isValidNode(node: SemanticNode): boolean {
  return node.time_meta?.invalid_at === undefined
}

/** 记录/节点在时点 t 是否有效（时间旅行视图，§2.5 完整版）：valid_at ≤ t < invalid_at。 */
function validAt(timeMeta: { valid_at: number; invalid_at?: number }, t: number): boolean {
  return timeMeta.valid_at <= t && (timeMeta.invalid_at === undefined || timeMeta.invalid_at > t)
}

/** 记录是否已压缩归档（§2.3 第 6 步：内容已由语义节点承载，退出检索）。 */
function isCompressed(record: MemoryRecord): boolean {
  return record.flags?.compressed_into !== undefined
}

/** open() 时的最小形状校验（§4.1）：id/text 为 string、embedding 为 number[]；不合规记录跳过并 warn。 */
function asMemoryRecord(raw: unknown): MemoryRecord | undefined {
  if (typeof raw !== 'object' || raw === null) return undefined
  const v = raw as Record<string, unknown>
  if (typeof v.id !== 'string' || typeof v.text !== 'string') return undefined
  if (!Array.isArray(v.embedding) || !v.embedding.every((e) => typeof e === 'number')) return undefined
  return raw as MemoryRecord
}

/** open() 时的最小形状校验（§4.2）：id/gist 为 string、embedding 为 number[]。 */
function asSemanticNode(raw: unknown): SemanticNode | undefined {
  if (typeof raw !== 'object' || raw === null) return undefined
  const v = raw as Record<string, unknown>
  if (typeof v.id !== 'string' || typeof v.gist !== 'string') return undefined
  if (!Array.isArray(v.embedding) || !v.embedding.every((e) => typeof e === 'number')) return undefined
  return raw as SemanticNode
}

/** 余弦均值；空集返回 0（无近邻时 r_m = 0）。适用于任何带 cos 字段的命中项。 */
function meanCos(items: readonly { cos: number }[]): number {
  if (items.length === 0) return 0
  let sum = 0
  for (const item of items) sum += item.cos
  return sum / items.length
}

/**
 * 记忆库门面：内存镜像（Map）+ VectorIndex 同步 + KvUnit 持久化。
 *
 * 索引不变量：**有效 episodic 记录（含已压缩归档者）与有效语义节点进入索引**；
 * 软失效记录保留在镜像与介质中、永不删除（§2.5），仅从索引移除。
 * 两个消费视图各取所需：
 * - similar()（写入路径矛盾判定，§5.1）：有效 episodic 记录（含已压缩——
 *   归档记忆仍可被新信息软失效，保证 P0 安全网不因 P1 迁移而退化）；
 * - searchRetrievable()（邻域统计，§3.2/§3.3）与 retrievableItems()（L0 检索池，
 *   §5.2）：有效未压缩记录 + 有效语义节点，即"检索宇宙"。
 */
export class MemoryStore {
  private closed = false

  private constructor(
    private readonly unit: KvUnit,
    private readonly index: VectorIndex,
    private readonly config: Config,
    private readonly records: Map<string, MemoryRecord>,
    private readonly nodes: Map<string, SemanticNode>,
  ) {}

  /**
   * 从已打开的 KvUnit 加载快照并重建索引（open KvUnit 本身是调用方的事：
   * 插件入口经 ctx.storage.backend.get(name).kv.open(descriptor) 打开后传入）。
   * 不合规记录逐条跳过并 console.warn 一次，不抛错。
   */
  static async open(unit: KvUnit, index: VectorIndex, config: Config): Promise<MemoryStore> {
    const snapshot = await unit.loadAll()
    const records = new Map<string, MemoryRecord>()
    const episodic = snapshot.tables[TABLE_EPISODIC] ?? {}
    for (const [key, raw] of Object.entries(episodic)) {
      const record = asMemoryRecord(raw)
      if (record === undefined) {
        console.warn(`[dsh-mem] 跳过 episodic 表中不合规的记录：${key}`)
        continue
      }
      records.set(record.id, record)
    }
    const nodes = new Map<string, SemanticNode>()
    const semantic = snapshot.tables[TABLE_SEMANTIC] ?? {}
    for (const [key, raw] of Object.entries(semantic)) {
      const node = asSemanticNode(raw)
      if (node === undefined) {
        console.warn(`[dsh-mem] 跳过 semantic 表中不合规的节点：${key}`)
        continue
      }
      nodes.set(node.id, node)
    }
    // 有效 episodic 记录（含已压缩）与有效语义节点入索引；软失效者排除（§2.5）
    for (const record of records.values()) {
      if (isValidRecord(record)) index.add(record.id, record.embedding)
    }
    for (const node of nodes.values()) {
      if (isValidNode(node)) index.add(node.id, node.embedding)
    }
    return new MemoryStore(unit, index, config, records, nodes)
  }

  /** 关闭 unit；幂等。关闭后任何写方法抛 Error('closed')。 */
  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    await this.unit.close()
  }

  /** 写前置检查：close 后写方法一律抛错。 */
  private assertOpen(): void {
    if (this.closed) throw new Error('closed')
  }

  /** 按 id 取记录（含软失效记录）。 */
  get(id: string): MemoryRecord | undefined {
    return this.records.get(id)
  }

  /** 全部记录（含软失效记录——记录永不删除，§2.5）。 */
  allMemories(): readonly MemoryRecord[] {
    return [...this.records.values()]
  }

  /** 有效记录（time_meta.invalid_at 未设者）。 */
  validMemories(): readonly MemoryRecord[] {
    return this.allMemories().filter(isValidRecord)
  }

  /** 按新值同步索引：先移除，valid 才重新加入（保证 remove+re-add 语义，与索引实现无关）。 */
  private syncIndex(record: MemoryRecord): void {
    this.index.remove(record.id)
    if (isValidRecord(record)) this.index.add(record.id, record.embedding)
  }

  /**
   * upsert：先持久化（unit.putRecord('episodic', id, record)）再更新内存镜像与索引；
   * 仅 valid 记录入索引，invalid 记录确保从索引移除。
   */
  async putRecord(record: MemoryRecord): Promise<void> {
    this.assertOpen()
    await this.unit.putRecord(TABLE_EPISODIC, record.id, record)
    this.records.set(record.id, record)
    this.syncIndex(record)
  }

  /**
   * 读改写：fn 作用于当前值，写回（先持久化后改内存镜像，与 KvUnit 契约一致）；
   * 记录不存在时抛 Error。返回写入后的记录。
   */
  async updateRecord(id: string, fn: (r: MemoryRecord) => MemoryRecord): Promise<MemoryRecord> {
    this.assertOpen()
    const current = this.records.get(id)
    if (current === undefined) throw new Error(`memory not found: ${id}`)
    const next = fn(current)
    if (next.id !== id) throw new Error(`updateRecord: id 不可变（${String(next.id)} != ${id}）`)
    await this.unit.putRecord(TABLE_EPISODIC, id, next)
    this.records.set(id, next)
    // 按新值 valid/invalid 同步索引（remove + re-add）
    this.syncIndex(next)
    return next
  }

  /**
   * 软失效（§2.5）：设 time_meta.invalid_at = at、flags.superseded_by = supersededBy，
   * 从索引移除（记录保留，永不删除）。幂等：对已软失效的记录重复调用是无操作、不抛；
   * 记录不存在时抛 Error。
   */
  async softInvalidate(id: string, supersededBy: string, at: number): Promise<void> {
    this.assertOpen()
    const current = this.records.get(id)
    if (current === undefined) throw new Error(`memory not found: ${id}`)
    // 幂等：已软失效则保持首次的时间戳不变
    if (isValidRecord(current)) {
      await this.updateRecord(id, (r) => ({
        ...r,
        time_meta: { ...r.time_meta, invalid_at: at },
        flags: { ...r.flags, superseded_by: supersededBy },
      }))
    }
  }

  /**
   * 近邻检索（只对 valid episodic 记录——含已压缩归档者，经 VectorIndex）。
   * 写入路径矛盾判定（§5.1）的消费视图；索引 filter 再核对一次镜像中的
   * 有效性，防御索引与镜像的任何漂移。
   */
  similar(vector: readonly number[], k: number): readonly SimilarMemory[] {
    const hits: SimilarMemory[] = []
    const neighbors: readonly Neighbor[] = this.index.search(vector, k, (id) => {
      const record = this.records.get(id)
      return record !== undefined && isValidRecord(record)
    })
    for (const n of neighbors) {
      const record = this.records.get(n.id)
      if (record !== undefined) hits.push({ id: n.id, cos: n.cos, record })
    }
    return hits
  }

  /**
   * 有效语义节点的近邻检索（支柱五完整版，P2）：写入路径矛盾判定的第二路
   * 安全网——被新记忆推翻的 gist 也走软失效，不再永留检索宇宙。
   */
  similarNodes(vector: readonly number[], k: number): readonly SimilarNodeHit[] {
    const hits: SimilarNodeHit[] = []
    const neighbors: readonly Neighbor[] = this.index.search(vector, k, (id) => {
      const node = this.nodes.get(id)
      return node !== undefined && isValidNode(node)
    })
    for (const n of neighbors) {
      const node = this.nodes.get(n.id)
      if (node !== undefined) hits.push({ id: n.id, cos: n.cos, node })
    }
    return hits
  }

  /** id 是否属于检索宇宙（§5.2 L0 池）：有效未压缩记录或有效语义节点。 */
  private isRetrievable(id: string): boolean {
    const record = this.records.get(id)
    if (record !== undefined) return isValidRecord(record) && !isCompressed(record)
    const node = this.nodes.get(id)
    return node !== undefined && isValidNode(node)
  }

  /** 检索宇宙内近邻检索（邻域统计用，§3.2/§3.3 的"记忆库"即此宇宙）。 */
  private searchRetrievable(vector: readonly number[], k: number): readonly Neighbor[] {
    return this.index.search(vector, k, (id) => this.isRetrievable(id))
  }

  /** 检索宇宙（§5.2 L0 池，MemoryLookup.retrievable 的实现）：有效未压缩情景记录 + 有效语义节点（节点 tier 由 S 分层）。 */
  retrievable(): readonly RetrievableItem[] {
    const items: RetrievableItem[] = []
    for (const record of this.records.values()) {
      if (!isValidRecord(record) || isCompressed(record)) continue
      items.push(this.toRetrievableRecord(record))
    }
    for (const node of this.nodes.values()) {
      if (!isValidNode(node)) continue
      items.push(this.toRetrievableNode(node))
    }
    return items
  }

  /**
   * 时点 t 的检索宇宙（时间旅行，§2.5 完整版）：valid_at ≤ t < invalid_at 的
   * 记录（且当时未压缩——compressed_at 未设或晚于 t）+ 当时有效的语义节点。
   * P1 遗留的已归档记录没有 compressed_at：保守视为"历来已压缩"（无法定
   * 期的归档不复活）。近似边界：tier / r_m / 节点 gist 取当前落盘快照，历史
   * S 与邻域不重建（有效性区间是精确重建，成员集历史不重建）。
   */
  retrievableAt(t: number): readonly RetrievableItem[] {
    const items: RetrievableItem[] = []
    for (const record of this.records.values()) {
      if (!validAt(record.time_meta, t)) continue
      const compressedAt = record.flags?.compressed_into !== undefined
        ? (record.time_meta.compressed_at ?? Number.NEGATIVE_INFINITY)
        : record.time_meta.compressed_at
      if (compressedAt !== undefined && compressedAt <= t) continue
      items.push(this.toRetrievableRecord(record))
    }
    for (const node of this.nodes.values()) {
      if (!validAt(node.time_meta, t)) continue
      items.push(this.toRetrievableNode(node))
    }
    return items
  }

  /** 记录 → L0 候选视图（retrievable / retrievableAt 共用投影）。 */
  private toRetrievableRecord(record: MemoryRecord): RetrievableItem {
    return {
      id: record.id,
      text: record.text,
      embedding: record.embedding,
      r_m: record.r_m,
      tier: record.tier,
      createdAt: record.time_meta.created_at,
      layer: 'episodic',
    }
  }

  /** 节点 → L0 候选视图（tier 由 S 现算，不落盘）。 */
  private toRetrievableNode(node: SemanticNode): RetrievableItem {
    return {
      id: node.id,
      text: node.gist,
      embedding: node.embedding,
      r_m: node.r_m,
      tier: tierOf(node.S, this.config.tier),
      createdAt: node.time_meta.created_at,
      layer: 'semantic',
    }
  }

  /** 语义层节点写入（P1 睡眠周期填充）。先持久化再改镜像与索引。 */
  async putNode(node: SemanticNode): Promise<void> {
    this.assertOpen()
    await this.unit.putRecord(TABLE_SEMANTIC, node.id, node)
    this.nodes.set(node.id, node)
    this.index.remove(node.id)
    if (isValidNode(node)) this.index.add(node.id, node.embedding)
  }

  /**
   * 语义节点软失效（支柱五完整版，P2 / §2.5）：设 time_meta.invalid_at 与
   * superseded_by，移出索引与检索宇宙，节点本体保留、永不删除。幂等：对已
   * 失效节点重复调用是无操作、不抛；节点不存在时抛 Error。成员记录保持归档
   * 状态（其内容仍由存储回溯，不经节点检索）。
   */
  async softInvalidateNode(id: string, supersededBy: string, at: number): Promise<void> {
    this.assertOpen()
    const current = this.nodes.get(id)
    if (current === undefined) throw new Error(`node not found: ${id}`)
    if (isValidNode(current)) {
      await this.putNode({
        ...current,
        time_meta: { ...current.time_meta, invalid_at: at },
        superseded_by: supersededBy,
      })
    }
  }

  getNode(id: string): SemanticNode | undefined {
    return this.nodes.get(id)
  }

  allNodes(): readonly SemanticNode[] {
    return [...this.nodes.values()]
  }

  /** 有效语义节点（time_meta.invalid_at 未设者）。 */
  validNodes(): readonly SemanticNode[] {
    return this.allNodes().filter(isValidNode)
  }

  /**
   * 刷新单条记忆的邻域统计（§3.2 / §3.3）。近邻取自**检索宇宙**
   * （有效未压缩记录 + 有效语义节点，即 r_m/red 所服务的 CSLS/聚类视图）：
   * - r_m = 检索宇宙 top-k 近邻平均余弦（不足 k 取实际数，排除自身）；
   * - red = top-5 近邻平均余弦（不足 5 取实际数；§3.3 公式中的固定宽度）。
   * 结果写回该记录并持久化，同时返回 { r_m, red }。记录不存在时抛 Error。
   */
  async refreshNeighborhood(id: string, k: number): Promise<{ r_m: number; red: number }> {
    this.assertOpen()
    const record = this.records.get(id)
    if (record === undefined) throw new Error(`memory not found: ${id}`)
    const { r_m, red } = this.neighborhoodStats(record.embedding, id, k)
    await this.updateRecord(id, (r) => ({ ...r, r_m, red }))
    return { r_m, red }
  }

  /**
   * 刷新语义节点 r_m（§2.3 第 7 步）：检索宇宙 top-k 近邻平均余弦（排除自身），
   * 写回节点并持久化，返回新值。节点不存在时抛 Error。
   */
  async refreshNodeNeighborhood(id: string, k: number): Promise<number> {
    this.assertOpen()
    const node = this.nodes.get(id)
    if (node === undefined) throw new Error(`node not found: ${id}`)
    const { r_m } = this.neighborhoodStats(node.embedding, id, k)
    await this.putNode({ ...node, r_m })
    return r_m
  }

  /** 检索宇宙内 top-k / top-5 邻域统计（排除自身；不足取实际数，空集为 0）。 */
  private neighborhoodStats(
    vector: readonly number[],
    selfId: string,
    k: number,
  ): { r_m: number; red: number } {
    // +1 容纳可能命中的自身，随后排除自身再截断
    const width = Math.max(k, RED_NEIGHBOR_TOP) + 1
    const neighbors = this.searchRetrievable(vector, width)
      .filter((n) => n.id !== selfId)
      .slice(0, Math.max(k, RED_NEIGHBOR_TOP))
    return { r_m: meanCos(neighbors.slice(0, k)), red: meanCos(neighbors.slice(0, RED_NEIGHBOR_TOP)) }
  }
}

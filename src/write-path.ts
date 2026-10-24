/**
 * 写入路径（设计 §5.1）：turn/end 后对本轮消息批量抽取候选，再对每条候选
 * 执行 O(1) 次模型调用：嵌入（批量一次）→ M5 判分 → ANN 近邻 → 简化矛盾
 * 判定（矛盾即软失效，补充建链，重复 NOOP）→ S 初始化为 imp → 落库并刷新
 * r_m / red 邻域统计。
 */
import type { Config } from './config.js'
import {
  epiLexical,
  extractCandidates,
  judgeContradiction,
  judgeImportance,
  type TurnMessage,
} from './judges.js'
import type { EmbeddingProvider, InternalLlm, MemoryRecord, MemoryTier } from './types.js'

/** 存储层最小依赖（由 MemoryStore 结构满足；窄接口保持可测）。 */
export interface MemoryStoreLike {
  validMemories(): readonly MemoryRecord[]
  /** 同步签名：底层 VectorIndex 是同步精确 kNN（与 MemoryStore.similar 一致）。 */
  similar(vector: readonly number[], k: number): readonly { id: string; cos: number; record: MemoryRecord }[]
  putRecord(record: MemoryRecord): Promise<void>
  softInvalidate(id: string, supersededBy: string, at: number): Promise<void>
  refreshNeighborhood(id: string, k: number): Promise<{ r_m: number; red: number }>
}

export interface WriteOutcome {
  readonly turn: number
  readonly written: readonly MemoryRecord[]
  readonly superseded: readonly { id: string; by: string }[]
  /** 低分 + 重复(NOOP) + 单条失败的总数。 */
  readonly dropped: number
}

/** §2.1 分层：S ≥ hot → hot；S < cold → cold；否则 warm。 */
export function tierOf(S: number, tier: Config['tier']): MemoryTier {
  if (S >= tier.hotThreshold) return 'hot'
  if (S < tier.coldThreshold) return 'cold'
  return 'warm'
}

export interface WritePipelineDeps {
  readonly llm: InternalLlm
  readonly embedding: EmbeddingProvider
  readonly store: MemoryStoreLike
  readonly config: Config
  readonly now?: () => number
  readonly log?: (msg: string) => void
}

export class WritePipeline {
  private readonly deps: WritePipelineDeps

  constructor(deps: WritePipelineDeps) {
    this.deps = deps
  }

  async processTurn(
    sessionId: string,
    turn: number,
    messages: readonly TurnMessage[],
    signal?: AbortSignal,
  ): Promise<WriteOutcome> {
    const { config } = this.deps
    const now = this.deps.now ?? Date.now
    const written: MemoryRecord[] = []
    const superseded: { id: string; by: string }[] = []
    let dropped = 0

    const window = messages.slice(-config.write.messageWindow)
    const candidates = await extractCandidates(this.deps.llm, window, config.write, signal)

    // 与库中已有记忆文本完全相同的候选走 NOOP（§2.5 重复→合并）；候选内去重
    const existingTexts = new Set(this.deps.store.validMemories().map((record) => record.text))
    const seen = new Set<string>()
    const fresh = candidates.filter((text) => {
      if (existingTexts.has(text) || seen.has(text)) return false
      seen.add(text)
      return true
    })
    if (fresh.length === 0) return { turn, written, superseded, dropped }

    // 嵌入一次批量完成（§5.4 写入成本行）
    const vectors = await this.deps.embedding.embed(fresh, signal)

    for (const [i, text] of fresh.entries()) {
      const vector = vectors[i]
      if (!vector) {
        dropped++
        continue
      }
      try {
        const verdict = await judgeImportance(this.deps.llm, text, signal)
        if (verdict.importance < config.write.minImportance) {
          dropped++
          continue
        }

        const id = crypto.randomUUID()
        const record: MemoryRecord = {
          id,
          text,
          embedding: vector,
          imp: verdict.importance,
          valence: verdict.valence,
          arousal: verdict.arousal,
          epi: epiLexical(text),
          red: 0,
          r_m: 0,
          tier: tierOf(verdict.importance, config.tier),
          con_route: { kind: 'pending' },
          links: [],
          S: verdict.importance, // §2.1：S 初始化为 imp；完整动态在 P2
          counters: { used_count: 0, suppressed_count: 0 },
          time_meta: { created_at: now(), valid_at: now() },
          flags: {},
          origin: { session_id: sessionId, turn, seq: i },
        }

        // 简化矛盾判定（§5.1 第 5 步）：top 相似且有效的记忆做一次 LLM 判定
        const neighbors = this.deps.store.similar(vector, config.write.contradictionTopK)
          .filter((n) => n.cos >= config.write.contradictionMinCos)
          .map((n) => ({ id: n.id, text: n.record.text }))
        const judgment = await judgeContradiction(this.deps.llm, text, neighbors, signal)

        if (judgment?.verdict === 'duplicate') {
          dropped++ // NOOP 路径：既有记忆已覆盖，不重复入库
          continue
        }
        if (judgment?.verdict === 'contradicts') {
          await this.deps.store.softInvalidate(judgment.againstId, id, now())
          record.links.push({ kind: 'supersedes', target: judgment.againstId })
          superseded.push({ id: judgment.againstId, by: id })
        } else if (judgment?.verdict === 'complements') {
          record.links.push({ kind: 'complements', target: judgment.againstId })
        }

        await this.deps.store.putRecord(record)
        await this.deps.store.refreshNeighborhood(id, config.retrieval.cslsK)
        written.push(record)
      } catch (err) {
        // 单条失败不中断整轮（§5.1）；signal 中止则原样上抛
        if (signal?.aborted) throw err
        dropped++
        this.deps.log?.(`write-path: 候选写入失败（${text.slice(0, 40)}…）：${String(err)}`)
      }
    }

    return { turn, written, superseded, dropped }
  }
}

/**
 * 支柱一 S 完整动态（设计 §2.1，P2）：
 *
 *   初始：      S ← imp                                   （写入路径，P0 已有）
 *   成功使用：  S ← S + α·(1 + cue_quality)·g(arousal)    α = score.useGain
 *   被抑制：    S ← S − ε                                 ε = score.suppressPenalty
 *   兜底衰减：  S ← S·e^(−Δt/τ)                           τ = score.decayTauDays
 *
 * - 「成功使用」= 进入最终上下文（两条检索通道的 top 输出，§2.1）。设计所列
 *   「被采纳」信号（ctx.messageFeedback / feedback/committed）不在当前依赖集内
 *   （P1 对 ctx.schedule 的同款取舍）：按「未被纠正」默认成立处理，增益在
 *   使用时点即时入账；接入该 seam 后可改为确认制（架构文档记录该边界）。
 * - 兜底衰减是工程保险（§2.1）：只影响分层、永不触发删除。衰减以
 *   counters.last_decay_at（缺省 created_at）为锚**惰性求值**——指数衰减可乘
 *   复合，事件路径先补齐自锚点以来的衰减再加/减事件增量，睡眠周期第 7 步
 *   对全库批处理，两者合起来与逐日衰减数学等价。
 * - arousal 乘法门（§3.5「arousal 作为巩固增益的乘法门」）：g = 0.5 + 0.5·arousal
 *   ∈ [0.5,1]——高唤醒巩固更强、平静记忆不归零（归零会切断测试效应通路）。
 *   语义节点无 arousal（gist 已去情境化），门恒为 1。
 * - 分层跃迁（§3.5「只在层级跃迁时重判」）：tier 随 S 更新；**事件驱动**的
 *   跃迁触发 M5 重判（新判分按 EMA 混入）；兜底衰减引起的跃迁不重判——
 *   衰减是工程保险、不构成重要性的新证据（同时防边界反复穿越烧调用）。
 */
import type { Config } from './config.js'
import { judgeImportance } from './judges.js'
import { tierOf } from './memory-store.js'
import type {
  InternalLlm,
  MemoryRecord,
  RetrievalResult,
  ScoreCounters,
  ScoredCandidate,
  SemanticNode,
} from './types.js'

const MS_PER_DAY = 24 * 3600 * 1000

/** 兜底衰减因子 e^(−Δt/τ)；Δt ≤ 0（时钟回拨/同刻）返回 1（幂等）。 */
export function decayFactor(elapsedMs: number, tauDays: number): number {
  if (elapsedMs <= 0) return 1
  return Math.exp(-elapsedMs / (tauDays * MS_PER_DAY))
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x))

/** arousal 乘法门（§3.5）：0.5 + 0.5·arousal ∈ [0.5,1]，越界值夹取。 */
export function arousalGate(arousal: number): number {
  return 0.5 + 0.5 * clamp01(arousal)
}

/**
 * 使用事件增益 α·(1 + cue)·g(arousal)：(1 + cue) 下夹 0——线索质量为负的
 * "使用"不产生负巩固（§2.1 公式在 cue < −1 时会出负增量，非设计意图）。
 */
export function useGain(cue: number, arousal: number, config: Config['score']): number {
  return config.useGain * Math.max(0, 1 + cue) * arousalGate(arousal)
}

/** 语义节点无 arousal，门恒为 1。 */
export function nodeUseGain(cue: number, config: Config['score']): number {
  return config.useGain * Math.max(0, 1 + cue)
}

/** 节点记账缺省形状（旧节点无 counters 时按全 0 容忍，P2 新写全会带上）。 */
export function zeroCounters(): ScoreCounters {
  return { used_count: 0, suppressed_count: 0 }
}

/** 兜底衰减锚点：last_decay_at 缺省回退 created_at（旧记录容忍，不回填）。 */
function decayAnchor(timeMeta: { created_at: number }, counters?: ScoreCounters): number {
  return counters?.last_decay_at ?? timeMeta.created_at
}

export interface ScoreEventOutcome<R> {
  readonly record: R
  readonly tierChanged: boolean
}

/**
 * 使用事件（情景记录）：先补衰减再加增益，counters 全套更新，tier 随 S 跃迁。
 * `inspired`（P3-4）：本次使用来自灵感通道——S 增益照常（效用闭环），另计
 * inspired_count（wild-card 使用率 = inspired_count / used_count）。
 */
export function applyUseEvent(
  record: MemoryRecord,
  cue: number | undefined,
  now: number,
  config: Config,
  inspired = false,
): ScoreEventOutcome<MemoryRecord> {
  const effectiveCue = cue ?? record.counters.last_cue_quality ?? 0
  const S =
    record.S * decayFactor(now - decayAnchor(record.time_meta, record.counters), config.score.decayTauDays) +
    useGain(effectiveCue, record.arousal, config.score)
  const tier = tierOf(S, config.tier)
  return {
    record: {
      ...record,
      S,
      tier,
      counters: {
        ...record.counters,
        used_count: record.counters.used_count + 1,
        ...(inspired ? { inspired_count: (record.counters.inspired_count ?? 0) + 1 } : {}),
        last_used_at: now,
        last_cue_quality: effectiveCue,
        last_decay_at: now,
      },
    },
    tierChanged: tier !== record.tier,
  }
}

/** 抑制事件（情景记录）：先补衰减再减 ε；每查询至多计一次由调用方保证。 */
export function applySuppressEvent(
  record: MemoryRecord,
  now: number,
  config: Config,
): ScoreEventOutcome<MemoryRecord> {
  const S =
    record.S * decayFactor(now - decayAnchor(record.time_meta, record.counters), config.score.decayTauDays) -
    config.score.suppressPenalty
  const tier = tierOf(S, config.tier)
  return {
    record: {
      ...record,
      S,
      tier,
      counters: {
        ...record.counters,
        suppressed_count: record.counters.suppressed_count + 1,
        last_decay_at: now,
      },
    },
    tierChanged: tier !== record.tier,
  }
}

/** 使用事件（语义节点）：无 arousal 门、无 M5 重判，其余同记录侧（inspired 计数同款）。 */
export function applyNodeUseEvent(
  node: SemanticNode,
  cue: number | undefined,
  now: number,
  config: Config,
  inspired = false,
): SemanticNode {
  const counters = node.counters ?? zeroCounters()
  const effectiveCue = cue ?? counters.last_cue_quality ?? 0
  const S =
    node.S * decayFactor(now - decayAnchor(node.time_meta, node.counters), config.score.decayTauDays) +
    nodeUseGain(effectiveCue, config.score)
  return {
    ...node,
    S,
    counters: {
      ...counters,
      used_count: counters.used_count + 1,
      ...(inspired ? { inspired_count: (counters.inspired_count ?? 0) + 1 } : {}),
      last_used_at: now,
      last_cue_quality: effectiveCue,
      last_decay_at: now,
    },
  }
}

/** 抑制事件（语义节点）。 */
export function applyNodeSuppressEvent(node: SemanticNode, now: number, config: Config): SemanticNode {
  const counters = node.counters ?? zeroCounters()
  const S =
    node.S * decayFactor(now - decayAnchor(node.time_meta, node.counters), config.score.decayTauDays) -
    config.score.suppressPenalty
  return { ...node, S, counters: { ...counters, suppressed_count: counters.suppressed_count + 1, last_decay_at: now } }
}

/** 兜底衰减（情景记录，睡眠周期第 7 步批处理）：只影响分层、不重判。 */
export function applyDecay(record: MemoryRecord, now: number, config: Config): MemoryRecord {
  const S = record.S * decayFactor(now - decayAnchor(record.time_meta, record.counters), config.score.decayTauDays)
  return {
    ...record,
    S,
    tier: tierOf(S, config.tier),
    counters: { ...record.counters, last_decay_at: now },
  }
}

/** 兜底衰减（语义节点，睡眠周期第 7 步批处理）。 */
export function applyNodeDecay(node: SemanticNode, now: number, config: Config): SemanticNode {
  const counters = node.counters ?? zeroCounters()
  const S = node.S * decayFactor(now - decayAnchor(node.time_meta, node.counters), config.score.decayTauDays)
  return { ...node, S, counters: { ...counters, last_decay_at: now } }
}

// ---- 编排：检索结果的 S 记账（两条模型可见通道共用） ----

/** 存储层最小依赖（由 MemoryStore 结构满足；窄接口保持可测）。 */
export interface ScoreStoreLike {
  get(id: string): MemoryRecord | undefined
  updateRecord(id: string, fn: (r: MemoryRecord) => MemoryRecord): Promise<MemoryRecord>
  getNode(id: string): SemanticNode | undefined
  putNode(node: SemanticNode): Promise<void>
}

export interface ScoreDeps {
  readonly llm: InternalLlm
  readonly store: ScoreStoreLike
  readonly config: Config
  readonly now?: () => number
  readonly log?: (msg: string) => void
}

export class ScoreKeeper {
  private readonly deps: ScoreDeps

  constructor(deps: ScoreDeps) {
    this.deps = deps
  }

  /**
   * 检索结果记账（§2.1/§2.2）：top = 使用事件、suppressed = 抑制事件，
   * 情景层与语义层都消费；软失效/缺记录静默跳过（检索与记账之间可能被推翻）。
   * `result.source === 'inspire'`（P3-4）时使用事件另计 inspired_count。
   * fire-and-forget 调用方保证失败不上抛。
   */
  async recordRetrieval(result: RetrievalResult): Promise<void> {
    const now = (this.deps.now ?? Date.now)()
    const inspired = result.source === 'inspire'
    for (const winner of result.top) {
      if (winner.item.layer === 'semantic') await this.useNode(winner, now, inspired)
      else await this.useRecord(winner, now, inspired)
    }
    for (const loser of result.suppressed) {
      if (loser.item.layer === 'semantic') await this.suppressNode(loser, now)
      else await this.suppressRecord(loser, now)
    }
  }

  private async useRecord(candidate: ScoredCandidate, now: number, inspired: boolean): Promise<void> {
    try {
      let tierChanged = false
      const updated = await this.deps.store.updateRecord(candidate.id, (record) => {
        if (record.time_meta.invalid_at !== undefined) return record // 已被推翻：不记账
        const outcome = applyUseEvent(record, candidate.cue, now, this.deps.config, inspired)
        tierChanged = outcome.tierChanged
        return outcome.record
      })
      if (tierChanged) await this.rejudge(updated)
    } catch (err) {
      this.deps.log?.(`scoring: 使用事件记账失败（${candidate.id}）：${String(err)}`)
    }
  }

  private async suppressRecord(candidate: ScoredCandidate, now: number): Promise<void> {
    try {
      let tierChanged = false
      const updated = await this.deps.store.updateRecord(candidate.id, (record) => {
        if (record.time_meta.invalid_at !== undefined) return record
        const outcome = applySuppressEvent(record, now, this.deps.config)
        tierChanged = outcome.tierChanged
        return outcome.record
      })
      if (tierChanged) await this.rejudge(updated)
    } catch (err) {
      this.deps.log?.(`scoring: 抑制事件记账失败（${candidate.id}）：${String(err)}`)
    }
  }

  private async useNode(candidate: ScoredCandidate, now: number, inspired: boolean): Promise<void> {
    try {
      const node = this.deps.store.getNode(candidate.id)
      if (node === undefined || node.time_meta.invalid_at !== undefined) return
      await this.deps.store.putNode(applyNodeUseEvent(node, candidate.cue, now, this.deps.config, inspired))
    } catch (err) {
      this.deps.log?.(`scoring: 节点使用事件记账失败（${candidate.id}）：${String(err)}`)
    }
  }

  private async suppressNode(candidate: ScoredCandidate, now: number): Promise<void> {
    try {
      const node = this.deps.store.getNode(candidate.id)
      if (node === undefined || node.time_meta.invalid_at !== undefined) return
      await this.deps.store.putNode(applyNodeSuppressEvent(node, now, this.deps.config))
    } catch (err) {
      this.deps.log?.(`scoring: 节点抑制事件记账失败（${candidate.id}）：${String(err)}`)
    }
  }

  /**
   * 层级跃迁重判（§3.5）：M5 重跑一次，新判分按 EMA（rejudgeEma 为新值权重）
   * 混入 imp/valence/arousal；reasons 取新判分。失败保留旧值、记日志不抛
   * （重判是增强信号，不能阻塞记账链）。
   */
  private async rejudge(record: MemoryRecord): Promise<void> {
    if (record.time_meta.invalid_at !== undefined) return
    try {
      const verdict = await judgeImportance(this.deps.llm, record.text)
      const beta = this.deps.config.score.rejudgeEma
      const ema = (old: number, fresh: number) => (1 - beta) * old + beta * fresh
      await this.deps.store.updateRecord(record.id, (current) => ({
        ...current,
        imp: ema(current.imp, verdict.importance),
        valence: ema(current.valence, verdict.valence),
        arousal: ema(current.arousal, verdict.arousal),
      }))
    } catch (err) {
      this.deps.log?.(`scoring: 层级跃迁重判失败（${record.id}），保留旧判分：${String(err)}`)
    }
  }
}

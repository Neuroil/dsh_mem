/**
 * 情绪编码（P3-1，设计 §10.1）：按工作区白名单启用的 mood 状态与检索重排。
 *
 * 边界（稳定路线）：
 * - 写入侧恒开（M5 valence/arousal 照常落盘），读取侧按白名单闸；
 * - mood 只进 CSLS 排序键（cue' = cue + α·moodSim），**不进 L0 召回、不进
 *   cold 门控**（冷层不得凭情绪契合混过门槛——侧门封死）；
 * - 写入路径对 mood 全程无感知（判分提示词不含当前心境，杜绝情绪→语料
 *   污染反馈回路）；mood 非记忆：不进索引/检索宇宙，clearAll 也不清。
 */
import type { Config } from './config.js'
import { normalizeWorkspacePath } from './scope.js'
import type { MoodState } from './types.js'

const MS_PER_HOUR = 3_600_000

/**
 * turn 级情绪并入 mood 的 EMA 权重（新观测占比）。工程常数（同
 * RED_NEIGHBOR_TOP 的处理：设计公式固定值，不进 Config）。
 */
export const MOOD_EMA_WEIGHT = 0.5

/** cwd 是否在情绪编码白名单内（归一化路径精确匹配；无 cwd / 空白名单恒 false）。 */
export function isEmotionEnabled(cwd: string | undefined, config: Config): boolean {
  if (cwd === undefined || cwd === '') return false
  const normalized = normalizeWorkspacePath(cwd)
  return config.emotion.workspaces.some((entry) => normalizeWorkspacePath(entry) === normalized)
}

/**
 * 情绪相似度 ∈ [0,1]：mood 与记忆判分的逐维距离取补（越接近越高）。
 * 下夹 0：契合者获得提升、不契合者保持中性——不做主动降权（稳定路线的
 * 有界影响：排序键偏移 ∈ [0, α]）。
 */
export function moodSim(mood: MoodState, valence: number, arousal: number): number {
  return Math.max(0, 1 - (Math.abs(mood.valence - valence) + Math.abs(mood.arousal - arousal)) / 2)
}

/** 衰减因子 e^(−Δt/τ)；Δt ≤ 0（同刻/时钟回拨）返回 1（幂等）。 */
function decayFactor(elapsedMs: number, tauHours: number): number {
  if (elapsedMs <= 0) return 1
  return Math.exp(-elapsedMs / (tauHours * MS_PER_HOUR))
}

/**
 * 惰性衰减（INV-7 同架构，无定时器）：向基线 (0,0) 按 e^(−Δt/τ) 衰减，
 * 锚点 updated_at 保持不动（衰减是读取投影，下一次 blendMood 才前移锚点）。
 */
export function decayMood(mood: MoodState, now: number, tauHours: number): MoodState {
  const factor = decayFactor(now - mood.updated_at, tauHours)
  return { valence: mood.valence * factor, arousal: mood.arousal * factor, updated_at: mood.updated_at }
}

/**
 * turn 级观测并入 mood：先按 now 衰减历史值再 EMA 融合（新观测权重
 * MOOD_EMA_WEIGHT）；无历史时直接采用新观测。锚点前移到 now。
 */
export function blendMood(
  current: MoodState | null,
  turn: { readonly valence: number; readonly arousal: number },
  now: number,
  tauHours: number,
): MoodState {
  if (current === null) {
    return { valence: turn.valence, arousal: turn.arousal, updated_at: now }
  }
  const decayed = decayMood(current, now, tauHours)
  const w = MOOD_EMA_WEIGHT
  return {
    valence: decayed.valence * (1 - w) + turn.valence * w,
    arousal: decayed.arousal * (1 - w) + turn.arousal * w,
    updated_at: now,
  }
}

/** 判分输出的 turn_mood 兜底夹取：valence → [-1,1]、arousal → [0,1]；非有限数值返回 undefined。 */
export function clampAffect(
  valence: unknown,
  arousal: unknown,
): { valence: number; arousal: number } | undefined {
  if (typeof valence !== 'number' || typeof arousal !== 'number') return undefined
  if (!Number.isFinite(valence) || !Number.isFinite(arousal)) return undefined
  return {
    valence: Math.min(1, Math.max(-1, valence)),
    arousal: Math.min(1, Math.max(0, arousal)),
  }
}

/** disclose 的一行心境描述：粗档标签 + 数值（英文，与 section header 同语）。 */
export function describeMood(mood: MoodState): string {
  const vLabel =
    mood.valence <= -0.5 ? 'quite negative'
    : mood.valence <= -0.15 ? 'negative'
    : mood.valence < 0.15 ? 'neutral'
    : mood.valence < 0.5 ? 'positive'
    : 'very positive'
  const aLabel =
    mood.arousal < 0.2 ? 'calm'
    : mood.arousal < 0.5 ? 'settled'
    : mood.arousal < 0.8 ? 'animated'
    : 'highly aroused'
  return `current mood: ${vLabel}, ${aLabel} (valence=${mood.valence.toFixed(2)}, arousal=${mood.arousal.toFixed(2)})`
}

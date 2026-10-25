/**
 * 情绪编码（P3-1）单测：白名单解析、moodSim、惰性衰减、EMA 聚合、
 * 夹取与 disclose 文本。
 */
import { describe, expect, it } from 'vitest'
import {
  MOOD_EMA_WEIGHT,
  blendMood,
  clampAffect,
  decayMood,
  describeMood,
  isEmotionEnabled,
  moodSim,
} from '../src/emotion.js'
import type { Config } from '../src/config.js'

const config = (workspaces: string[]): Config => ({
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope: true },
  embedding: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 1000, dimensions: 2, batchSize: 1 },
  reranker: { mode: 'deterministic', endpoint: '', model: '', apiKey: '', timeoutMs: 1000 },
  llm: { provider: 'deepseek', model: 'deepseek-chat', temperature: 0, maxTokens: 512 },
  write: { messageWindow: 10, maxCandidates: 8, minImportance: 3, contradictionTopK: 5, contradictionMinCos: 0.8 },
  retrieval: { l0TopK: 100, cslsTopK: 20, rerankTopK: 8, mmrTopK: 5, mmrLambda: 0.6, cslsK: 10, contextTurns: 3, coldCueBoost: 0.1, cueThreshold: 0.3, pprSeedK: 8, pprDamping: 0.5, pprWeight: 0.2 },
  tier: { hotThreshold: 8, coldThreshold: 2 },
  score: { useGain: 0.5, suppressPenalty: 0.05, decayTauDays: 180, rejudgeEma: 0.5 },
  sleep: { enabled: true, minNewMemories: 10, clusterCos: 0.75, clusterMinSize: 3, qaPairCount: 3, qaF1Threshold: 0.8, epiThreshold: 0.5, conWindow: 1000, conAttachSigma: 0.5, conNewNodeSigma: -1, conNewNodeMinImp: 7, attachGain: 0.5 },
  section: { enabled: true, maxMemories: 5, header: 'memories:' },
  tool: { enabled: true, maxResults: 5 },
  intake: { session: { enabled: true, acceptSourceKinds: ['user'] } },
  emotion: { workspaces, alpha: 0.15, tauHours: 4, disclose: false },
})

const HOUR = 3_600_000

describe('isEmotionEnabled（白名单）', () => {
  it('归一化路径精确匹配：大小写/分隔符/尾斜杠不敏感', () => {
    const cfg = config(['C:/Users/Neuroil/QQBot'])
    expect(isEmotionEnabled('C:\\Users\\Neuroil\\QQBot', cfg)).toBe(true)
    expect(isEmotionEnabled('c:/users/neuroil/qqbot/', cfg)).toBe(true)
    expect(isEmotionEnabled('C:/Users/Neuroil/QQBot', cfg)).toBe(true)
  })

  it('不在名单内 / 无 cwd / 空白名单 恒 false（默认 bypass）', () => {
    expect(isEmotionEnabled('C:/work/other', config(['C:/Users/Neuroil/QQBot']))).toBe(false)
    expect(isEmotionEnabled(undefined, config(['C:/Users/Neuroil/QQBot']))).toBe(false)
    expect(isEmotionEnabled('C:/Users/Neuroil/QQBot', config([]))).toBe(false)
    expect(isEmotionEnabled('C:/Users/Neuroil/QQBot/sub', config(['C:/Users/Neuroil/QQBot']))).toBe(false) // 精确匹配，非前缀
  })
})

describe('moodSim', () => {
  const mood = { valence: -0.6, arousal: 0.8, updated_at: 0 }
  it('完全一致 = 1；逐维距离线性扣减', () => {
    expect(moodSim(mood, -0.6, 0.8)).toBe(1)
    expect(moodSim(mood, -0.6, 0.6)).toBeCloseTo(0.9, 10)
    expect(moodSim(mood, 0, 0.8)).toBeCloseTo(0.7, 10)
  })
  it('两维全异下夹 0（契合提升、不契合中性，不主动降权）', () => {
    expect(moodSim(mood, 1, 0)).toBe(0)
    expect(moodSim(mood, -0.6, 0.8)).toBe(1)
  })
})

describe('decayMood（τ=4h 惰性衰减，INV-7 同架构）', () => {
  const mood = { valence: -0.8, arousal: 0.6, updated_at: 0 }
  it('一个 τ 衰减到 1/e；锚点不动（读取投影不回写）', () => {
    const out = decayMood(mood, 4 * HOUR, 4)
    expect(out.valence).toBeCloseTo(-0.8 / Math.E, 10)
    expect(out.arousal).toBeCloseTo(0.6 / Math.E, 10)
    expect(out.updated_at).toBe(0)
  })
  it('Δt ≤ 0 幂等返回原值（同刻/时钟回拨）', () => {
    expect(decayMood(mood, 0, 4)).toEqual(mood)
    expect(decayMood(mood, -100, 4)).toEqual(mood)
  })
  it('隔日（24h = 6τ）基本归零', () => {
    const out = decayMood(mood, 24 * HOUR, 4)
    expect(Math.abs(out.valence)).toBeLessThan(0.0025)
    expect(Math.abs(out.arousal)).toBeLessThan(0.0025)
  })
})

describe('blendMood（EMA 聚合）', () => {
  it('无历史直接采用新观测', () => {
    expect(blendMood(null, { valence: 0.3, arousal: 0.2 }, 1000, 4)).toEqual({
      valence: 0.3, arousal: 0.2, updated_at: 1000,
    })
  })
  it('先按 now 衰减历史再 EMA 融合；锚点前移', () => {
    // 历史 (−0.8, 0.6) @0，4h 后衰减为 (−0.8/e, 0.6/e)，与新观测 (0.4, 0.1) 各半
    const out = blendMood({ valence: -0.8, arousal: 0.6, updated_at: 0 }, { valence: 0.4, arousal: 0.1 }, 4 * HOUR, 4)
    expect(out.valence).toBeCloseTo((-0.8 / Math.E) * (1 - MOOD_EMA_WEIGHT) + 0.4 * MOOD_EMA_WEIGHT, 10)
    expect(out.arousal).toBeCloseTo((0.6 / Math.E) * (1 - MOOD_EMA_WEIGHT) + 0.1 * MOOD_EMA_WEIGHT, 10)
    expect(out.updated_at).toBe(4 * HOUR)
  })
})

describe('clampAffect', () => {
  it('夹取 valence 到 [-1,1]、arousal 到 [0,1]', () => {
    expect(clampAffect(-2, 5)).toEqual({ valence: -1, arousal: 1 })
    expect(clampAffect(0.3, 0.4)).toEqual({ valence: 0.3, arousal: 0.4 })
  })
  it('非数值/非有限数值返回 undefined', () => {
    expect(clampAffect('sad', 0.5)).toBeUndefined()
    expect(clampAffect(Number.NaN, 0.5)).toBeUndefined()
    expect(clampAffect(0.5, Number.POSITIVE_INFINITY)).toBeUndefined()
  })
})

describe('describeMood（disclose 文本）', () => {
  it('输出含粗档标签与数值的英文一行', () => {
    const text = describeMood({ valence: -0.6, arousal: 0.85, updated_at: 0 })
    expect(text).toContain('quite negative')
    expect(text).toContain('highly aroused')
    expect(text).toContain('valence=-0.60')
    expect(describeMood({ valence: 0, arousal: 0.1, updated_at: 0 })).toContain('neutral')
  })
})

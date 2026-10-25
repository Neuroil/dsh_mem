import { describe, expect, it } from 'vitest'
import {
  answerFromGist,
  epiLexical,
  extractCandidates,
  extractGist,
  generateQaPairs,
  judgeContradiction,
  judgeImportance,
  judgeNodeCompatibility,
  lexicalDensities,
  parseJsonLoose,
} from '../src/judges.js'
import type { InternalLlm, SupersessionVerdict } from '../src/types.js'
import type { WriteConfig } from '../src/config.js'

const writeConfig: WriteConfig = {
  messageWindow: 10,
  maxCandidates: 4,
  minImportance: 3,
  contradictionTopK: 5,
  contradictionMinCos: 0.8,
}

/** 按调用序回放预置回复的假 LLM；记录收到的 signal 与调用数。 */
class FakeLlm implements InternalLlm {
  readonly calls: Array<{ system: string; user: string }> = []
  lastSignal?: AbortSignal
  constructor(private readonly replies: string[]) {}
  async complete(system: string, user: string, signal?: AbortSignal): Promise<string> {
    this.calls.push({ system, user })
    this.lastSignal = signal
    const reply = this.replies.shift()
    if (reply === undefined) throw new Error('FakeLlm 无更多预置回复')
    return reply
  }
}

describe('parseJsonLoose', () => {
  it('解析裸 JSON、围栏包裹、正文包围的 JSON', () => {
    expect(parseJsonLoose('{"a":1}')).toEqual({ a: 1 })
    expect(parseJsonLoose('```json\n[1,2]\n```')).toEqual([1, 2])
    expect(parseJsonLoose('好的，结果如下：{"x": "含 } 字符串"} 请查收')).toEqual({ x: '含 } 字符串' })
    expect(parseJsonLoose('前缀 ["a", "b]"] 后缀')).toEqual(['a', 'b]'])
  })

  it('对象数组以最先出现的 "[" 为起点（回归：曾被误截成首个对象元素）', () => {
    expect(parseJsonLoose('[{"question":"q","answer":"a"},{"question":"q2","answer":"a2"}]'))
      .toEqual([{ question: 'q', answer: 'a' }, { question: 'q2', answer: 'a2' }])
    // 正文包围时仍取更靠前者
    expect(parseJsonLoose('结果：[{"x":1}] 完')).toEqual([{ x: 1 }])
    expect(parseJsonLoose('结果：{"x":[1]} 完')).toEqual({ x: [1] })
  })

  it('无 JSON 时抛错', () => {
    expect(() => parseJsonLoose('完全不是 JSON')).toThrow()
  })
})

describe('extractCandidates', () => {
  it('空消息不调 LLM', async () => {
    const llm = new FakeLlm([])
    expect(await extractCandidates(llm, [], writeConfig)).toEqual({ candidates: [] })
    expect(llm.calls.length).toBe(0)
  })

  it('解析 memories 数组并截断到 maxCandidates；非字符串与重复项被过滤；turn_mood 夹取', async () => {
    const llm = new FakeLlm(['```json\n{"memories": ["m1", 42, "m1", "m2", "m3", "m4"], "turn_mood": {"valence": -2, "arousal": 5}}\n```'])
    const out = await extractCandidates(llm, [{ role: 'user', text: 'hi' }], writeConfig)
    expect(out.candidates).toEqual(['m1', 'm2', 'm3', 'm4'])
    expect(out.turnMood).toEqual({ valence: -1, arousal: 1 }) // 夹取到 [-1,1]/[0,1]
  })

  it('turn_mood 缺失/非法按无观测处理（宽容，不触发重试）', async () => {
    const missing = new FakeLlm(['{"memories": ["m1"]}'])
    const out1 = await extractCandidates(missing, [{ role: 'user', text: 'hi' }], writeConfig)
    expect(out1.candidates).toEqual(['m1'])
    expect(out1.turnMood).toBeUndefined()

    const invalid = new FakeLlm(['{"memories": ["m1"], "turn_mood": {"valence": "sad", "arousal": 0.5}}'])
    const out2 = await extractCandidates(invalid, [{ role: 'user', text: 'hi' }], writeConfig)
    expect(out2.candidates).toEqual(['m1'])
    expect(out2.turnMood).toBeUndefined()
    expect(invalid.calls.length).toBe(1) // 未重试
  })

  it('解析失败重试一次，仍失败抛错', async () => {
    const llm = new FakeLlm(['不是 JSON', '还是不是'])
    await expect(extractCandidates(llm, [{ role: 'user', text: 'hi' }], writeConfig)).rejects.toThrow()
    expect(llm.calls.length).toBe(2)
  })

  it('缺 memories 数组（结构校验失败）同样重试一次，仍失败抛错', async () => {
    const llm = new FakeLlm(['["m1"]', '["m2"]'])
    await expect(extractCandidates(llm, [{ role: 'user', text: 'hi' }], writeConfig)).rejects.toThrow(/缺少 memories 数组/)
    expect(llm.calls.length).toBe(2)
  })
})

describe('judgeImportance', () => {
  it('正常解析并 clamp 数值、过滤非法 reasons', async () => {
    const llm = new FakeLlm(['{"importance": 12, "valence": -3, "arousal": 2, "reasons": ["decision", "bogus", "decision"]}'])
    const verdict = await judgeImportance(llm, '用户决定把生产数据库迁到 Postgres')
    expect(verdict.importance).toBe(10)
    expect(verdict.valence).toBe(-1)
    expect(verdict.arousal).toBe(1)
    expect(verdict.reasons).toEqual(['decision'])
  })

  it('importance 非数值时抛错（含重试）', async () => {
    const llm = new FakeLlm(['{"importance": "high"}', '{"importance": null}'])
    await expect(judgeImportance(llm, 'x')).rejects.toThrow()
    expect(llm.calls.length).toBe(2)
  })

  it('提示词带锚定示例与 rubric 维度', async () => {
    const llm = new FakeLlm(['{"importance":5,"valence":0,"arousal":0,"reasons":[]}'])
    await judgeImportance(llm, 'x')
    expect(llm.calls[0]!.system).toMatch(/brushing\s+teeth.*2|2.*brushing/is)
    expect(llm.calls[0]!.system).toContain('user_identity')
  })
})

describe('judgeContradiction', () => {
  it('空近邻直接返回 null 不调 LLM', async () => {
    const llm = new FakeLlm([])
    expect(await judgeContradiction(llm, 'c', [])).toBeNull()
    expect(llm.calls.length).toBe(0)
  })

  it('unrelated / 越界 index / 非法 verdict 返回 null', async () => {
    const neighbors = [{ id: 'm1', text: 't1' }, { id: 'm2', text: 't2' }]
    const unrelated = new FakeLlm(['{"index":0,"verdict":"unrelated"}'])
    expect(await judgeContradiction(unrelated, 'c', neighbors)).toBeNull()
    const outOfRange = new FakeLlm(['{"index":5,"verdict":"contradicts"}'])
    expect(await judgeContradiction(outOfRange, 'c', neighbors)).toBeNull()
    const badVerdict = new FakeLlm(['{"index":0,"verdict":"maybe"}'])
    expect(await judgeContradiction(badVerdict, 'c', neighbors)).toBeNull()
  })

  it('合法判定映射回近邻 id', async () => {
    const neighbors = [{ id: 'm1', text: 't1' }, { id: 'm2', text: 't2' }]
    const llm = new FakeLlm(['{"index":1,"verdict":"contradicts"}'])
    const verdict: SupersessionVerdict | null = (await judgeContradiction(llm, 'c', neighbors))?.verdict ?? null
    const judgment = await judgeContradiction(new FakeLlm(['{"index":1,"verdict":"contradicts"}']), 'c', neighbors)
    expect(verdict).toBe('contradicts')
    // temporal 缺省按 overlapping（P0 安全网语义保持）
    expect(judgment).toEqual({ verdict: 'contradicts', againstId: 'm2', temporal: 'overlapping' })
  })
})

describe('epiLexical', () => {
  it('有界 [0,1] 且确定性', () => {
    for (const text of ['plain statement', '昨天 14:30 在北京和 Zhang 三开会', 'no dates here', '']) {
      const value = epiLexical(text)
      expect(value).toBeGreaterThanOrEqual(0)
      expect(value).toBeLessThanOrEqual(1)
      expect(epiLexical(text)).toBe(value)
    }
  })

  it('时间/实体密集文本的 epi 高于纯抽象文本', () => {
    const episodic = epiLexical('On 2026-10-24 at 9:30 Zhang Wei met the AWS team in Shanghai about the migration')
    const semantic = epiLexical('部署延迟通常源于配置复杂性')
    expect(episodic).toBeGreaterThan(semantic)
  })
})

// ---- 睡眠周期判定（§2.3 / §3.4） ----

describe('extractGist', () => {
  it('解析 {"gist": "..."} 并 trim；提示词要求去情景化', async () => {
    const llm = new FakeLlm(['```json\n{"gist": "  部署延迟与配置复杂性相关  "}\n```'])
    const gist = await extractGist(llm, ['记忆一', '记忆二'])
    expect(gist).toBe('部署延迟与配置复杂性相关')
    expect(llm.calls[0]!.system).toContain('semantic core')
    expect(llm.calls[0]!.user).toContain('记忆一')
    expect(llm.calls[0]!.user).toContain('记忆二')
  })

  it('空 gist 触发重试，仍失败抛错', async () => {
    const llm = new FakeLlm(['{"gist": ""}', '{"gist": "   "}'])
    await expect(extractGist(llm, ['m'])).rejects.toThrow()
    expect(llm.calls.length).toBe(2)
  })
})

describe('generateQaPairs', () => {
  it('解析对象数组、过滤缺字段/空串项、截断到 count', async () => {
    const llm = new FakeLlm([
      '[{"question":"q1","answer":"a1"},{"question":"","answer":"a2"},' +
      '{"question":"q3"},{"question":42,"answer":"a4"},{"question":"q5","answer":"a5"}]',
    ])
    const pairs = await generateQaPairs(llm, ['记忆'], 3)
    expect(pairs).toEqual([
      { question: 'q1', answer: 'a1' },
      { question: 'q5', answer: 'a5' },
    ])
  })

  it('返回非数组（被误截的对象等）触发重试', async () => {
    const llm = new FakeLlm(['{"question":"q","answer":"a"}', '[]'])
    expect(await generateQaPairs(llm, ['记忆'], 3)).toEqual([])
    expect(llm.calls.length).toBe(2)
  })
})

describe('answerFromGist', () => {
  it('解析 {"answer": "..."}；提示词仅允许依据 gist 作答', async () => {
    const llm = new FakeLlm(['{"answer": "部署延迟"}'])
    const answer = await answerFromGist(llm, '部署延迟与配置复杂性相关', '什么导致延迟？')
    expect(answer).toBe('部署延迟')
    expect(llm.calls[0]!.system).toContain('ONLY')
    expect(llm.calls[0]!.user).toContain('部署延迟与配置复杂性相关')
    expect(llm.calls[0]!.user).toContain('什么导致延迟？')
  })

  it('空答案触发重试；signal 透传', async () => {
    const signal = new AbortController().signal
    const llm = new FakeLlm(['{"answer": ""}', '{"answer": "ok"}'])
    expect(await answerFromGist(llm, 'g', 'q', signal)).toBe('ok')
    expect(llm.calls.length).toBe(2)
    expect(llm.lastSignal).toBe(signal)
  })
})

describe('judgeNodeCompatibility', () => {
  it('解析布尔；不兼容与兼容两分支', async () => {
    const yes = new FakeLlm(['{"compatible":true}'])
    expect(await judgeNodeCompatibility(yes, '新语句', '节点 gist')).toBe(true)
    const no = new FakeLlm(['{"compatible":false}'])
    expect(await judgeNodeCompatibility(no, '新语句', '节点 gist')).toBe(false)
    expect(no.calls[0]!.user).toContain('节点 gist')
    expect(no.calls[0]!.user).toContain('新语句')
  })

  it('compatible 非布尔触发重试，仍非法抛错', async () => {
    const llm = new FakeLlm(['{"compatible":"yes"}', '{"compatible":1}'])
    await expect(judgeNodeCompatibility(llm, 'a', 'b')).rejects.toThrow()
    expect(llm.calls.length).toBe(2)
  })
})

describe('lexicalDensities', () => {
  it('时间/实体密度独立输出，与 epiLexical 一致', () => {
    const episodic = lexicalDensities('On 2026-10-24 at 9:30 Zhang Wei met the AWS team in Shanghai')
    const semantic = lexicalDensities('部署延迟通常源于配置复杂性')
    expect(episodic.time).toBeGreaterThan(semantic.time)
    expect(episodic.entity).toBeGreaterThan(semantic.entity)
  })
})

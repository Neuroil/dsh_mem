/**
 * LLM 判定集合（设计 §3.5 / §2.5 / §5.1 / §2.3 / §3.4）：候选抽取（批量）、
 * M5 rubric 判分、简化矛盾判定，以及睡眠周期的 gist 抽取 / QA 对生成 /
 * gist 作答 / M4 节点兼容性复核。全部经注入的 InternalLlm（ctx.llm），
 * 提示词带结构化 JSON 约束，解析失败重试一次。另有纯词法的 epi 近似
 * （§3.6 的词法特征部分）。
 */
import type { WriteConfig } from './config.js'
import { clampAffect } from './emotion.js'
import { sigmoid } from './metrics.js'
import type {
  ContradictionJudgment,
  ImportanceVerdict,
  InternalLlm,
  QaPair,
  SupersessionVerdict,
} from './types.js'

export interface TurnMessage {
  readonly role: 'user' | 'assistant'
  readonly text: string
}

const IMPORTANCE_REASONS = ['task_impact', 'user_identity', 'decision', 'emotion', 'reversible'] as const

/** 健壮 JSON 提取：剥围栏、以文本中最先出现的 '{' 或 '[' 为起点截取首个平衡值后解析。 */
export function parseJsonLoose(raw: string): unknown {
  const text = raw.replace(/```(?:json)?/gi, '').trim()
  // 起点取更靠前的开括号：否则 [{...}] 会被误截成首个对象元素
  const objectStart = text.indexOf('{')
  const arrayStart = text.indexOf('[')
  let open: string
  let start: number
  if (objectStart === -1 && arrayStart === -1) {
    throw new Error(`回复中未找到 JSON：${raw.slice(0, 120)}`)
  } else if (arrayStart === -1 || (objectStart !== -1 && objectStart < arrayStart)) {
    open = '{'
    start = objectStart
  } else {
    open = '['
    start = arrayStart
  }
  const close = open === '{' ? '}' : ']'
  let depth = 0
  let inString = false
  let escaped = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (escaped) {
      escaped = false
    } else if (ch === '\\') {
      escaped = true
    } else if (ch === '"') {
      inString = !inString
    } else if (!inString && ch === open) {
      depth++
    } else if (!inString && ch === close) {
      depth--
      if (depth === 0) {
        return JSON.parse(text.slice(start, i + 1))
      }
    }
  }
  throw new Error(`回复中未找到 JSON：${raw.slice(0, 120)}`)
}

/** 调用 + 解析 + 结构校验，失败重试一次（校验也计入重试：模型给错形状同样值得重试）。 */
async function completeJson<T>(
  llm: InternalLlm,
  system: string,
  user: string,
  parse: (raw: unknown) => T,
  signal?: AbortSignal,
): Promise<T> {
  let lastError: unknown
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return parse(parseJsonLoose(await llm.complete(system, user, signal)))
    } catch (err) {
      lastError = err
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError))
}

const EXTRACT_SYSTEM = `You distill durable memories from a conversation.
Return ONLY a JSON object with exactly two keys:
- "memories": an array of at most the requested number of candidates. Each
  element is a plain string: a self-contained, third-person statement of a
  fact, preference, decision, or commitment worth remembering across sessions
  (user identity, stable preferences, decisions and their reasons, recurring
  tasks). Skip transient chatter, greetings, and anything derivable from the
  immediately visible context. Write each memory in the language it was
  expressed in. No numbering, no explanations.
- "turn_mood": your estimate of the OVERALL emotional tone of this exchange as
  {"valence": <number in -1..1, negative to positive>, "arousal": <number in
  0..1, calm to excited>}. Estimate it even when the exchange seems neutral
  (then use values near 0).
No text outside the JSON object.`

/** turn 级整体情绪估计（抽取调用顺带产出，P3-1）：非法/缺失时 undefined（宽容，不触发重试）。 */
export interface TurnExtraction {
  readonly candidates: readonly string[]
  readonly turnMood?: { readonly valence: number; readonly arousal: number }
}

/** 批量候选抽取（§5.1，每轮一次调用；P3-1 起顺带产出 turn 级情绪）。 */
export async function extractCandidates(
  llm: InternalLlm,
  messages: readonly TurnMessage[],
  config: WriteConfig,
  signal?: AbortSignal,
): Promise<TurnExtraction> {
  if (messages.length === 0) return { candidates: [] }
  const transcript = messages
    .map((m) => `${m.role === 'user' ? 'USER' : 'ASSISTANT'}: ${m.text}`)
    .join('\n\n')
  const parsed = await completeJson<{ memories: readonly unknown[]; turn_mood?: { valence?: unknown; arousal?: unknown } }>(
    llm,
    EXTRACT_SYSTEM,
    `Conversation:\n\n${transcript}\n\nReturn up to ${config.maxCandidates} memory candidates and the turn mood as specified.`,
    (raw) => {
      // 非「memories 数组」对象属结构校验失败：抛错触发重试（与 M5/矛盾判定的重试语义一致）
      if (typeof raw !== 'object' || raw === null || !Array.isArray((raw as { memories?: unknown }).memories)) {
        throw new Error('候选抽取返回的 JSON 缺少 memories 数组')
      }
      return raw as { memories: readonly unknown[]; turn_mood?: { valence?: unknown; arousal?: unknown } }
    },
    signal,
  )
  const seen = new Set<string>()
  const candidates: string[] = []
  for (const item of parsed.memories) {
    if (typeof item !== 'string') continue
    const text = item.trim()
    if (!text || seen.has(text)) continue
    seen.add(text)
    candidates.push(text)
    if (candidates.length >= config.maxCandidates) break
  }
  // turn_mood 宽容处理：缺失/非法按无观测处理（情绪聚合容忍空洞），夹取到合法域
  const rawMood = parsed.turn_mood
  const turnMood =
    typeof rawMood === 'object' && rawMood !== null
      ? clampAffect(rawMood.valence, rawMood.arousal)
      : undefined
  return { candidates, ...(turnMood !== undefined ? { turnMood } : {}) }
}

const IMPORTANCE_SYSTEM = `You score how strongly a memory deserves long-term consolidation.
Consider: impact on the user's tasks, relevance to user identity or stable
preferences, decision relevance, emotional weight, and irreversibility.
Anchor your 1-10 importance scale like this: a routine daily event (brushing
teeth) is 2; a significant but ordinary preference is 5; asking a crush out is
8; a life-changing decision is 10.
Return ONLY a JSON object:
{"importance": <integer 1-10>, "valence": <number -1..1, negative=unpleasant>,
 "arousal": <number 0..1, 1=highly arousing>,
 "reasons": ["task_impact"|"user_identity"|"decision"|"emotion"|"reversible"]}
Pick the reasons that actually apply (0 to 3 of them). No prose.`

const clamp = (x: number, low: number, high: number): number => Math.min(high, Math.max(low, x))

/** M5 rubric 判分（§3.5）。 */
export async function judgeImportance(
  llm: InternalLlm,
  text: string,
  signal?: AbortSignal,
): Promise<ImportanceVerdict> {
  return completeJson(llm, IMPORTANCE_SYSTEM, `Memory: ${text}`, (raw): ImportanceVerdict => {
    if (typeof raw !== 'object' || raw === null) throw new Error('M5 判分返回非对象')
    const record = raw as Record<string, unknown>
    // importance 必须是数值：null/undefined 经 Number() 会静默变成 0 再被夹取
    if (typeof record.importance !== 'number' || !Number.isFinite(record.importance)) {
      throw new Error('M5 判分 importance 非数值')
    }
    const importance = Math.round(clamp(record.importance, 1, 10))
    const reasons = Array.isArray(record.reasons)
      ? record.reasons.filter((r): r is (typeof IMPORTANCE_REASONS)[number] =>
          typeof r === 'string' && (IMPORTANCE_REASONS as readonly string[]).includes(r))
      : []
    return {
      importance,
      valence: clamp(Number(record.valence) || 0, -1, 1),
      arousal: clamp(Number(record.arousal) || 0, 0, 1),
      reasons: [...new Set(reasons)],
    }
  }, signal)
}

const CONTRADICTION_SYSTEM = `You judge how a NEW memory statement relates to similar EXISTING
memories (some may be distilled summary statements of older memories).
Return ONLY a JSON object:
{"index": <0-based index of the most-related existing memory>,
 "verdict": "contradicts" | "complements" | "duplicate" | "unrelated",
 "temporal": "overlapping" | "disjoint",
 "related": [<0-based indices of OTHER existing memories that are topically
              related to the new one — optional, may be empty or absent>]}
- "contradicts": the two statements cannot both hold — the old one is superseded.
- "temporal" (only for "contradicts"): "overlapping" when both make claims about
  the same time frame (present-tense claims included) — only then is the old
  memory superseded; "disjoint" when they describe genuinely different periods
  (e.g. a past state that has since legitimately changed) — then both stay valid.
- "complements": they add different facets of the same subject.
- "duplicate": the new statement adds nothing over the existing one.
- "unrelated": nothing above applies. "related" is independent of the verdict:
list there every other existing memory the new one is topically connected to
(same subject, project, person, or theme), even when the verdict is unrelated.
No prose.`

const VERDICTS: readonly SupersessionVerdict[] = ['contradicts', 'complements', 'duplicate', 'unrelated']

/**
 * 简化矛盾判定（§2.5 完整版）：一次调用判定全部近邻（记录 + 语义节点 gist），
 * 返回最强判定；contradicts 附带时间重叠判定（缺省/非法值按 overlapping，
 * 保持 P0 语义）；无近邻或全部无关返回 null。
 */
export async function judgeContradiction(
  llm: InternalLlm,
  candidate: string,
  neighbors: readonly { id: string; text: string }[],
  signal?: AbortSignal,
): Promise<ContradictionJudgment | null> {
  if (neighbors.length === 0) return null
  const listing = neighbors.map((n, i) => `${i}. ${n.text}`).join('\n')
  const verdict = await completeJson(
    llm,
    CONTRADICTION_SYSTEM,
    `NEW memory: ${candidate}\n\nEXISTING memories:\n${listing}`,
    (raw): ContradictionJudgment | null => {
      if (typeof raw !== 'object' || raw === null) return null
      const record = raw as Record<string, unknown>
      const index = record.index
      const verdict = record.verdict
      if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= neighbors.length) return null
      if (verdict !== 'contradicts' && verdict !== 'complements' && verdict !== 'duplicate' && verdict !== 'unrelated') {
        return null
      }
      if (verdict === 'unrelated') return null
      const temporal = record.temporal
      // P3-2 扩产出：related 索引列表（宽容解析：非法/越界/缺失按空；不含主判定目标）
      const related = Array.isArray(record.related)
        ? record.related
          .filter((i): i is number =>
            typeof i === 'number' && Number.isInteger(i) && i >= 0 && i < neighbors.length && i !== index)
          .map((i) => neighbors[i]!.id)
        : []
      return {
        verdict,
        againstId: neighbors[index]!.id,
        // 缺省/非法按 overlapping：P0 安全网语义不变（宁软失效不错过取代）
        temporal: verdict === 'contradicts' ? (temporal === 'disjoint' ? 'disjoint' : 'overlapping') : undefined,
        relatedIds: related,
      }
    },
    signal,
  )
  return verdict
}

// ---- epi 词法近似（§3.6 的 P0 子集） ----

const TIME_EXPRESSIONS: readonly RegExp[] = [
  /\d{4}[-/年.]\d{1,2}([-/月.]\d{1,2})?/u,
  /\d{1,2}:\d{2}/u,
  /(yesterday|today|tomorrow|last night|ago|monday|tuesday|wednesday|thursday|friday|saturday|sunday)/iu,
  /(昨天|今天|明天|前天|上周|下周|刚才|刚刚|之前|之后|上午|下午|晚上|点|月|日|号|年)/u,
]

const ENTITY_HINTS: readonly RegExp[] = [
  /\b[A-Z][a-zA-Z]+\b/u, // 英文专名启发（句首词由位置过滤兜底）
  /\d+/u, // 数字/日期成分
  /[\u4e00-\u9fff][a-zA-Z]|[a-zA-Z][\u4e00-\u9fff]/u, // 中英混排专名
]

function density(hits: number, tokens: number): number {
  return hits / Math.max(1, tokens)
}

/**
 * §3.6 词法特征：时间表达密度与实体密度（均按 token 数归一）。
 * epiLexical 与睡眠周期的完整 M6 公式共用。
 */
export function lexicalDensities(text: string): { time: number; entity: number } {
  const tokens = text.match(/[\p{L}\p{N}]+/gu) ?? []
  const timeHits = TIME_EXPRESSIONS.reduce((sum, re) => sum + (text.match(new RegExp(re.source, re.flags + 'g'))?.length ?? 0), 0)
  const entityHits = ENTITY_HINTS.reduce((sum, re) => sum + (text.match(new RegExp(re.source, re.flags + 'g'))?.length ?? 0), 0)
  return { time: density(timeHits, tokens.length), entity: density(entityHits, tokens.length) }
}

/**
 * epi 词法部分的有界近似 ∈ [0,1]：实体密度与时间表达密度经 sigmoid 合成。
 * 写入路径的 P0 子集；完整版（z 标准化 + G 探针）在睡眠周期 sleep-cycle.ts。
 */
export function epiLexical(text: string): number {
  const { time, entity } = lexicalDensities(text)
  return sigmoid(time + entity)
}

// ---- 睡眠周期判定（§2.3 / §3.4） ----

const GIST_SYSTEM = `You distill the shared semantic core from a cluster of related memories.
Write ONE self-contained statement that abstracts what these memories have in
common — the general fact, preference, or principle — dropping episodic
details (specific dates, times, places, session specifics). Write it in the
language the memories use. Return ONLY a JSON object:
{"gist": "<one or two sentences>"}`

/** gist 抽取（§2.3 第 2 步）：LLM 对簇做摘要抽象。 */
export async function extractGist(
  llm: InternalLlm,
  texts: readonly string[],
  signal?: AbortSignal,
): Promise<string> {
  const listing = texts.map((t) => `- ${t}`).join('\n')
  const parsed = await completeJson(llm, GIST_SYSTEM, `Memory cluster:\n${listing}`, (raw) => {
    if (typeof raw !== 'object' || raw === null) throw new Error('gist 抽取返回非对象')
    const gist = (raw as Record<string, unknown>).gist
    if (typeof gist !== 'string' || gist.trim() === '') throw new Error('gist 抽取返回空 gist')
    return gist.trim()
  }, signal)
  return parsed
}

const QA_GEN_SYSTEM = `You write question-answer pairs that check whether a summary preserves the
information of a memory cluster. Each question must be answerable from the
cluster content alone, and answers must be concise reference answers in the
memories' language. Return ONLY a JSON array of at most the requested number
of pairs, each {"question": "...", "answer": "..."}. No prose.`

/** QA 对生成（§2.3 第 3 步）：从簇内容生成参考问答对。 */
export async function generateQaPairs(
  llm: InternalLlm,
  texts: readonly string[],
  count: number,
  signal?: AbortSignal,
): Promise<readonly QaPair[]> {
  const listing = texts.map((t) => `- ${t}`).join('\n')
  const parsed = await completeJson(llm, QA_GEN_SYSTEM,
    `Memory cluster:\n${listing}\n\nReturn up to ${count} question-answer pairs as a JSON array.`,
    (raw) => {
      if (!Array.isArray(raw)) throw new Error('QA 生成返回非 JSON 数组')
      return raw
    }, signal)
  const pairs: QaPair[] = []
  for (const item of parsed) {
    if (typeof item !== 'object' || item === null) continue
    const { question, answer } = item as Record<string, unknown>
    if (typeof question !== 'string' || typeof answer !== 'string') continue
    if (question.trim() === '' || answer.trim() === '') continue
    pairs.push({ question: question.trim(), answer: answer.trim() })
    if (pairs.length >= count) break
  }
  return pairs
}

const GIST_ANSWER_SYSTEM = `You answer a question using ONLY the provided summary statement. Base your
answer strictly on the statement; if it does not contain the answer, reply
with exactly: unknown. Answer in the statement's language. Return ONLY a JSON
object: {"answer": "<short answer or unknown>"}`

/** 仅以 gist 作答（§2.3 第 3 步）：QA 闸门的被测方。 */
export async function answerFromGist(
  llm: InternalLlm,
  gist: string,
  question: string,
  signal?: AbortSignal,
): Promise<string> {
  return completeJson(llm, GIST_ANSWER_SYSTEM,
    `Summary statement: ${gist}\n\nQuestion: ${question}`,
    (raw) => {
      if (typeof raw !== 'object' || raw === null) throw new Error('gist 作答返回非对象')
      const answer = (raw as Record<string, unknown>).answer
      if (typeof answer !== 'string' || answer.trim() === '') throw new Error('gist 作答返回空答案')
      return answer.trim()
    }, signal)
}

const NODE_COMPAT_SYSTEM = `You judge whether a NEW distilled statement can be folded into an EXISTING
semantic knowledge node without contradiction. The node's gist may be
extended with the new facet, but the two must not state incompatible facts.
Return ONLY a JSON object: {"compatible": true} or {"compatible": false}`

/** M4 兼容性复核（§3.4）：con 超挂入阈值后，LLM 复核信息兼容才挂入。 */
export async function judgeNodeCompatibility(
  llm: InternalLlm,
  gist: string,
  nodeGist: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return completeJson(llm, NODE_COMPAT_SYSTEM,
    `EXISTING node gist: ${nodeGist}\n\nNEW statement: ${gist}`,
    (raw) => {
      if (typeof raw !== 'object' || raw === null) throw new Error('兼容性复核返回非对象')
      const compatible = (raw as Record<string, unknown>).compatible
      if (typeof compatible !== 'boolean') throw new Error('兼容性复核 compatible 非布尔')
      return compatible
    }, signal)
}

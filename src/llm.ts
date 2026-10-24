/**
 * 内部 LLM 调用（设计 §5.0）：M5 判分 / 矛盾判定 / 候选抽取一律经 ctx.llm
 * 的提供方无关流服务（compaction-basic 是先例），模型经 Config 选择。
 */
import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { InternalLlm } from './types.js'
import type { JudgeLlmConfig } from './config.js'

export class CordisLlm implements InternalLlm {
  private readonly ctx: Context
  private readonly config: JudgeLlmConfig

  constructor(ctx: Context, config: JudgeLlmConfig) {
    this.ctx = ctx
    this.config = config
  }

  async complete(system: string, user: string, signal?: AbortSignal): Promise<string> {
    const message = createUserMessage({
      content: [{ type: 'text', text: user }],
      source: { kind: 'user' },
    })
    const assembler = new BlockAssembler()
    for await (const chunk of this.ctx.llm.stream({
      provider: this.config.provider,
      model: this.config.model,
      messages: [message],
      system,
      temperature: this.config.temperature,
      maxTokens: this.config.maxTokens,
      ...(signal ? { signal } : {}),
    })) {
      assembler.push(chunk)
    }
    // dsh-llm 契约：适配器/提供方失败被规范化为终态 error/aborted finish 块，
    // 迭代本身不抛错——必须经 assembler.finish 读取，否则真实失败被误报为"空回复"。
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      const failure = finish.failure
      const status = failure.status !== undefined ? ` status=${failure.status}` : ''
      throw new Error(
        `ctx.llm 调用失败（provider=${this.config.provider} model=${this.config.model} code=${failure.code}${status}）：${failure.message}`,
      )
    }
    const text = assembler
      .blocks()
      .filter((block): block is Extract<typeof block, { type: 'text' }> => block.type === 'text')
      .map((block) => block.text)
      .join('')
      .trim()
    if (!text) throw new Error(`ctx.llm 空回复（provider=${this.config.provider} model=${this.config.model}）`)
    return text
  }
}

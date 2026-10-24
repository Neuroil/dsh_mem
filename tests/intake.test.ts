/**
 * 会话历史入口过滤（M3）单测：acceptSourceKinds 配置化 + form 守卫。
 */
import { describe, expect, it } from 'vitest'
import { isConversationMessage } from '../src/intake.js'
import type { Message } from '@deepseek-ai/dsh-llm'

const msg = (source: Record<string, unknown>, role: 'user' | 'assistant' = 'user'): Message =>
  ({ role, content: [], source }) as unknown as Message

describe('isConversationMessage（M3 入口过滤）', () => {
  const defaults = ['user']

  it('默认只认 kind=user 的用户消息', () => {
    expect(isConversationMessage(msg({ kind: 'user' }), defaults)).toBe(true)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x' }), defaults)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'qq' }), defaults)).toBe(false)
  })

  it('自定义 kind 加进接受列表即视为人类输入（IM 适配器接法）', () => {
    const accept = ['user', 'qq']
    expect(isConversationMessage(msg({ kind: 'qq' }), accept)).toBe(true)
    expect(isConversationMessage(msg({ kind: 'user' }), accept)).toBe(true)
    expect(isConversationMessage(msg({ kind: 'model' }), accept)).toBe(false)
  })

  it('带 form 的消息是插件上下文，恒不算对话（即使 kind 在接受列表中）', () => {
    const accept = ['user', 'plugin']
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'instructions' }), accept)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'catalog' }), accept)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'snapshot', sections: [] }), accept)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'notice', summary: 's' }), accept)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'relay' }), accept)).toBe(false)
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x', form: 'recall' }), accept)).toBe(false)
    // 无 form 的 plugin 消息按列表判定
    expect(isConversationMessage(msg({ kind: 'plugin', plugin: 'x' }), accept)).toBe(true)
  })

  it('kind=user 但带 form（理论上不会出现，防御性）也被 form 守卫排除', () => {
    expect(isConversationMessage(msg({ kind: 'user', form: 'notice', summary: 's' }), defaults)).toBe(false)
  })

  it('assistant 消息恒为对话另一半', () => {
    expect(isConversationMessage(msg({ kind: 'model' }, 'assistant'), defaults)).toBe(true)
  })
})

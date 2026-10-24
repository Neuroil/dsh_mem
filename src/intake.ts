/**
 * 会话历史入口过滤（M3）："什么算对话"不再硬编码。
 *
 * - 接受的 `source.kind` 由配置决定（`intake.session.acceptSourceKinds`，
 *   默认 ['user']）：IM 适配器（如 QQ 桥）若以自定义 kind 注入消息，把它加进
 *   配置即视为人类输入，无需改码。
 * - 语义守卫：带 `form` 的消息（instructions / catalog / snapshot / notice /
 *   relay / recall，见 @deepseek-ai/dsh-llm 的 MessageSource 契约）是插件注入
 *   的**上下文**而非对话，无论 kind 是否在接受列表中都一律排除——防止把系统
 *   提示词拼装、目录发布等内容当成用户说话写进记忆。
 */
import type { Message } from '@deepseek-ai/dsh-llm'

/** MessageSource 的最小结构（真联合满足；测试可传字面量）。 */
type SourceLike = { readonly kind: string; readonly form?: string }

/**
 * 一条消息是否属于"对话"（值得进入记忆写入管线）。
 * assistant 消息恒为对话另一半；user 侧按 acceptSourceKinds + form 守卫判定。
 */
export function isConversationMessage(message: Message, acceptSourceKinds: readonly string[]): boolean {
  if (message.role === 'assistant') return true
  const source = message.source as SourceLike
  if (source.form !== undefined) return false
  return acceptSourceKinds.includes(source.kind)
}

/**
 * 工作区作用域（M2）：记忆按"会话创建时的工作目录"物理隔离。
 *
 * scope = session.header.cwd（宿主已验证的绝对路径）的归一化哈希；每个 scope
 * 一个 KvUnit（unit 名 = `<storage.unit>_ws_<hash>`），无 cwd 的会话回退到
 * `storage.unit` 本名（向后兼容单库时代）。ScopeRegistry 管理惰性开启与
 * 统一关闭（HMR 卸载时全部收掉）。
 */
import type { Config } from './config.js'

/** FNV-1a 32 位哈希（与确定性嵌入同族；此处用于 scope 键，独立导出避免私有耦合）。 */
export function fnv1a(text: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash >>> 0
}

/**
 * 工作目录归一化：反斜杠→正斜杠、统一小写（Windows 盘符/大小写不敏感）、
 * 去掉尾部分隔符。归一后的同一路径得到同一 scope。
 */
export function normalizeWorkspacePath(path: string): string {
  return path.replaceAll('\\', '/').toLowerCase().replace(/\/+$/, '')
}

/** scope 键 = 归一路径的 FNV-1a 8 位十六进制（小写）。 */
export function workspaceScopeHash(cwd: string): string {
  return fnv1a(normalizeWorkspacePath(cwd)).toString(16).padStart(8, '0')
}

/**
 * scope 键 → unit 名：工作区 unit = `<storage.unit>_ws_<hash>`（满足宿主
 * UNIT_NAME_RE）；undefined（无 cwd / 关闭工作区隔离）→ `storage.unit` 本名。
 */
export function unitNameFor(scopeKey: string | undefined, config: Config): string {
  if (scopeKey === undefined) return config.storage.unit
  return `${config.storage.unit}_ws_${scopeKey}`
}

/** 会话侧的最小结构依赖（真 Session 满足；测试可传字面量）。 */
export type SessionLike = { readonly header?: { readonly cwd?: string } }

/** agent 侧的最小结构依赖（真 Agent.session 满足）。 */
export type AgentLike = { readonly session?: SessionLike } | undefined

function scopeKeyOf(cwd: string | undefined, config: Config): string | undefined {
  if (!config.storage.workspaceScope) return undefined
  return cwd === undefined || cwd === '' ? undefined : workspaceScopeHash(cwd)
}

/** 会话的 scope 键（写入路径路由）。 */
export function scopeKeyOfSession(session: SessionLike, config: Config): string | undefined {
  return scopeKeyOf(session.header?.cwd, config)
}

/** agent 的 scope 键（工具/注入路径路由，经 agent.session.header.cwd）。 */
export function scopeKeyOfAgent(agent: AgentLike, config: Config): string | undefined {
  return scopeKeyOf(agent?.session?.header?.cwd, config)
}

export interface ScopeRegistryOptions<T> {
  /** 开一个作用域（仅在首次命中该 scope 键时调用）。 */
  open: (scopeKey: string | undefined) => Promise<T>
  /** 关一个作用域（卸载时对所有已开/在途作用域调用）。 */
  close: (value: T) => void | Promise<void>
  /** open 失败的旁路报告（不吞错：get 仍 reject，由调用方决定降级）。 */
  onOpenError?: (scopeKey: string | undefined, err: unknown) => void
}

/**
 * 作用域注册表：scope 键 → 惰性单例（缓存 Promise，并发 get 共享同一次
 * open）；open 失败不缓存（下次事件重试）；closeAll 统一关闭（含在途 open
 * 完成后关闭）。closeAll 后注册表进入**终态**：get 一律拒绝、绝不重开——
 * 已卸载实例的迟到事件若重开刚关掉的 unit，会让新实例撞宿主 already-open
 * 保护（HMR 重载竞争），该工作区直到重启宿主都不可用。
 */
export class ScopeRegistry<T> {
  private readonly entries = new Map<string, Promise<T>>()
  private closed = false

  constructor(private readonly options: ScopeRegistryOptions<T>) {}

  private cacheKey(scopeKey: string | undefined): string {
    // scope 键恒为 8 位十六进制，'\0' 前缀保证回退槽永不与真实键碰撞
    return scopeKey === undefined ? '\0fallback' : scopeKey
  }

  get(scopeKey: string | undefined): Promise<T> {
    if (this.closed) {
      return Promise.reject(
        new Error('scope registry closed (plugin unloaded); refusing to reopen units'),
      )
    }
    const key = this.cacheKey(scopeKey)
    let entry = this.entries.get(key)
    if (entry === undefined) {
      entry = this.options.open(scopeKey).catch((err: unknown) => {
        this.entries.delete(key)
        this.options.onOpenError?.(scopeKey, err)
        throw err
      })
      this.entries.set(key, entry)
    }
    return entry
  }

  /** 卸载时调用：关闭全部已开（含在途 open 完成后）作用域。幂等。 */
  async closeAll(): Promise<void> {
    if (this.closed) return
    this.closed = true
    const entries = [...this.entries.values()]
    this.entries.clear()
    await Promise.allSettled(entries.map((entry) => entry.then((value) => this.options.close(value))))
  }

  get size(): number {
    return this.entries.size
  }
}

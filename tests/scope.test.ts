/**
 * 工作区作用域（M2）单测：路径归一化、scope 哈希、unit 命名（含宿主
 * UNIT_NAME_RE 合规）、workspaceScope=false/无 cwd 的回退语义，以及
 * ScopeRegistry 的惰性单例 / 并发共享 / 失败不缓存 / 统一关闭。
 */
import { describe, expect, it, vi } from 'vitest'
import {
  ScopeRegistry,
  normalizeWorkspacePath,
  scopeKeyOfAgent,
  scopeKeyOfSession,
  unitNameFor,
  workspaceScopeHash,
} from '../src/scope.js'
import { UNIT_NAME_RE, type Config } from '../src/config.js'

const config = (workspaceScope: boolean): Config => ({
  storage: { backend: 'json', unit: 'dsh_mem', version: 1, workspaceScope },
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
  emotion: { workspaces: [], alpha: 0.15, tauHours: 4, disclose: false },
  graph: { synonymCos: 0.8, maxDegree: 8 },
})

describe('normalizeWorkspacePath / workspaceScopeHash', () => {
  it('反斜杠→正斜杠、统一小写、去尾部分隔符', () => {
    expect(normalizeWorkspacePath('C:\\Users\\Neuroil\\Code\\dsh_mem')).toBe('c:/users/neuroil/code/dsh_mem')
    expect(normalizeWorkspacePath('C:/Users/Neuroil/Code/dsh_mem/')).toBe('c:/users/neuroil/code/dsh_mem')
    expect(normalizeWorkspacePath('D:\\Projects\\Bot')).toBe('d:/projects/bot')
  })

  it('同一工作区的不同写法得到同一 scope 哈希（Windows 大小写/分隔符不敏感）', () => {
    const a = workspaceScopeHash('C:\\Users\\Neuroil\\Code\\dsh_mem')
    const b = workspaceScopeHash('c:/users/neuroil/code/DSH_MEM/')
    const c = workspaceScopeHash('C:/Users/Neuroil/Code/dsh_mem')
    expect(a).toBe(b)
    expect(b).toBe(c)
  })

  it('不同工作区得到不同哈希；哈希为 8 位小写十六进制', () => {
    const a = workspaceScopeHash('C:/work/alpha')
    const b = workspaceScopeHash('C:/work/beta')
    expect(a).not.toBe(b)
    expect(a).toMatch(/^[0-9a-f]{8}$/)
    expect(b).toMatch(/^[0-9a-f]{8}$/)
  })

  it('哈希稳定（FNV-1a 固定向量）', () => {
    // fnv1a("") = 0x811c9dc5
    expect(workspaceScopeHash('')).toBe('811c9dc5')
  })
})

describe('unitNameFor（M2 unit 命名）', () => {
  it('工作区 unit = <unit>_ws_<hash>，满足宿主 UNIT_NAME_RE', () => {
    const name = unitNameFor(workspaceScopeHash('C:/work/alpha'), config(true))
    expect(name).toBe(`dsh_mem_ws_${workspaceScopeHash('C:/work/alpha')}`)
    expect(name).toMatch(UNIT_NAME_RE)
  })

  it('undefined（无 cwd / 关闭隔离）回退 storage.unit 本名（向后兼容）', () => {
    expect(unitNameFor(undefined, config(true))).toBe('dsh_mem')
    expect(unitNameFor(undefined, config(false))).toBe('dsh_mem')
  })
})

describe('scopeKeyOfSession / scopeKeyOfAgent', () => {
  it('按 cwd 哈希；无 cwd 返回 undefined', () => {
    const cfg = config(true)
    expect(scopeKeyOfSession({ header: { cwd: 'C:\\work\\alpha' } }, cfg)).toBe(workspaceScopeHash('C:/work/alpha'))
    expect(scopeKeyOfSession({ header: {} }, cfg)).toBeUndefined()
    expect(scopeKeyOfSession({}, cfg)).toBeUndefined()
  })

  it('agent 经 agent.session.header.cwd 解析', () => {
    const cfg = config(true)
    expect(scopeKeyOfAgent({ session: { header: { cwd: 'C:/work/alpha' } } }, cfg)).toBe(workspaceScopeHash('C:/work/alpha'))
    expect(scopeKeyOfAgent(undefined, cfg)).toBeUndefined()
    expect(scopeKeyOfAgent({ session: { header: {} } }, cfg)).toBeUndefined()
  })

  it('workspaceScope=false 恒为 undefined（全局单库旧行为）', () => {
    const cfg = config(false)
    expect(scopeKeyOfSession({ header: { cwd: 'C:/work/alpha' } }, cfg)).toBeUndefined()
    expect(scopeKeyOfAgent({ session: { header: { cwd: 'C:/work/alpha' } } }, cfg)).toBeUndefined()
  })
})

describe('ScopeRegistry', () => {
  it('惰性单例：同一 scope 键只 open 一次，get 返回同一实例', async () => {
    const open = vi.fn(async (key: string | undefined) => ({ key }))
    const close = vi.fn()
    const registry = new ScopeRegistry({ open, close })

    const a = await registry.get('ab12cd34')
    const b = await registry.get('ab12cd34')
    expect(a).toBe(b)
    expect(open).toHaveBeenCalledTimes(1)
    expect(registry.size).toBe(1)
  })

  it('并发 get 共享同一次 open（缓存 Promise）', async () => {
    let resolveOpen: ((v: { key: string | undefined }) => void) | undefined
    const open = vi.fn(() => new Promise<{ key: string | undefined }>((r) => { resolveOpen = r }))
    const registry = new ScopeRegistry({ open, close: () => {} })

    const p1 = registry.get('ab12cd34')
    const p2 = registry.get('ab12cd34')
    resolveOpen?.({ key: 'ab12cd34' })
    const [a, b] = await Promise.all([p1, p2])
    expect(a).toBe(b)
    expect(open).toHaveBeenCalledTimes(1)
  })

  it('undefined（回退槽）与真实 scope 键互不碰撞', async () => {
    const open = vi.fn(async (key: string | undefined) => ({ key }))
    const registry = new ScopeRegistry({ open, close: () => {} })

    const fallback = await registry.get(undefined)
    const scoped = await registry.get('ab12cd34')
    expect(fallback).not.toBe(scoped)
    expect(fallback.key).toBeUndefined()
    expect(scoped.key).toBe('ab12cd34')
    expect(open).toHaveBeenCalledTimes(2)
  })

  it('open 失败不缓存：下次 get 重试并成功；onOpenError 旁路上报', async () => {
    let fail = true
    const open = vi.fn(async (key: string | undefined) => {
      if (fail) throw new Error('boom')
      return { key }
    })
    const onOpenError = vi.fn()
    const registry = new ScopeRegistry({ open, close: () => {}, onOpenError })

    await expect(registry.get('ab12cd34')).rejects.toThrow('boom')
    expect(onOpenError).toHaveBeenCalledTimes(1)
    fail = false
    await expect(registry.get('ab12cd34')).resolves.toEqual({ key: 'ab12cd34' })
    expect(open).toHaveBeenCalledTimes(2)
  })

  it('closeAll 关闭全部已开作用域并清空缓存；幂等', async () => {
    const open = vi.fn(async (key: string | undefined) => ({ key }))
    const close = vi.fn()
    const registry = new ScopeRegistry({ open, close })

    await registry.get(undefined)
    await registry.get('ab12cd34')
    await registry.get('ef567890')
    await registry.closeAll()
    expect(close).toHaveBeenCalledTimes(3)
    expect(registry.size).toBe(0)
    await registry.closeAll()
    expect(close).toHaveBeenCalledTimes(3)
  })

  it('closeAll 后 get 一律拒绝：迟到事件绝不重开 unit（HMR 卸载防线）', async () => {
    const open = vi.fn(async (key: string | undefined) => ({ key }))
    const close = vi.fn()
    const registry = new ScopeRegistry({ open, close })

    await registry.get('ab12cd34')
    const closing = registry.closeAll()
    // closeAll 同步进入终态：进行中的 closeAll 期间与完成后的 get 都拒绝
    await expect(registry.get('ef567890')).rejects.toThrow(/closed/)
    await closing
    await expect(registry.get('ab12cd34')).rejects.toThrow(/closed/)
    await expect(registry.get(undefined)).rejects.toThrow(/closed/)
    expect(open).toHaveBeenCalledTimes(1)
    expect(close).toHaveBeenCalledTimes(1)
  })

  it('closeAll 等待在途 open 完成后再关闭（卸载不泄漏刚开的 unit）', async () => {
    let resolveOpen: ((v: { key: string | undefined }) => void) | undefined
    const open = vi.fn(() => new Promise<{ key: string | undefined }>((r) => { resolveOpen = r }))
    const close = vi.fn()
    const registry = new ScopeRegistry({ open, close })

    const pending = registry.get('ab12cd34')
    const closing = registry.closeAll()
    resolveOpen?.({ key: 'ab12cd34' })
    await pending
    await closing
    expect(close).toHaveBeenCalledTimes(1)
  })

  it('双工作区路由：两个不同 cwd 的会话进入不同 unit（物理隔离的装配前提）', async () => {
    const openedNames: Array<string | undefined> = []
    const open = vi.fn(async (key: string | undefined) => {
      const unitName = unitNameFor(key, config(true))
      openedNames.push(unitName)
      return { key, unitName }
    })
    const registry = new ScopeRegistry({ open, close: () => {} })
    const cfg = config(true)

    const s1 = scopeKeyOfSession({ header: { cwd: 'C:\\work\\alpha' } }, cfg)
    const s2 = scopeKeyOfSession({ header: { cwd: 'C:/work/beta' } }, cfg)
    const rt1 = await registry.get(s1)
    const rt2 = await registry.get(s2)

    expect(rt1).not.toBe(rt2)
    expect(rt1.unitName).toMatch(/^dsh_mem_ws_[0-9a-f]{8}$/)
    expect(rt2.unitName).toMatch(/^dsh_mem_ws_[0-9a-f]{8}$/)
    expect(rt1.unitName).not.toBe(rt2.unitName)
    expect(new Set(openedNames).size).toBe(2)
  })
})

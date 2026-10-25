/**
 * Runtime orchestration: wires storage, embedding, judges, the write path
 * and the retrieval pipeline into the two model-reachable channels (§5.2):
 *
 *  1. automatic injection — an agent-scoped `system-prompt/assemble`
 *     waterfall listener that runs the retrieval chain and appends the top
 *     memories as a prompt section for every step;
 *  2. on-demand recall — the `memory_search` tool.
 *
 * Write observation listens to persistent `session/event`s (user/message,
 * assistant/message, turn/end) and batches candidate extraction per turn
 * (§5.1). `agent/*` events are live-only and never persisted, so they are
 * used solely for the pending-query hint, never as a memory source.
 *
 * M2 工作区隔离：每个 scope（= session.header.cwd 的归一化哈希；无 cwd 回退
 * undefined = storage.unit 本名）一个 ScopedRuntime——自己的 store / 检索 /
 * 写入 / 睡眠 / 计分，惰性开启（ScopeRegistry 缓存 open Promise），卸载时
 * 统一关闭。写入按会话 scope 路由，注入与工具按 agent 会话 scope 路由；
 * 单个 scope 开启失败只影响该 scope 的那次事件，不拖垮其他工作区。HMR
 * 重载的卸载契约：unit 经 closeAll 统一关闭，agent 上的 assemble 监听器
 * 显式摘除、启动时对存量 agent 补挂——旧实例不得再碰 unit（否则新实例撞
 * already-open），新实例不得漏挂存量 agent。
 */
import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { AssistantMessage, Message, UserMessage } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
// 拉入 system-prompt 包的声明合并，使 'system-prompt/assemble' 事件可类型化
import type {} from '@deepseek-ai/dsh-system-prompt'
import { createEmbeddingProvider, createRerankerProvider } from './embedding.js'
import { blendMood, decayMood, describeMood, isEmotionEnabled } from './emotion.js'
import { graphStats } from './graph.js'
import { isConversationMessage } from './intake.js'
import type { TurnMessage } from './judges.js'
import { CordisLlm } from './llm.js'
import { MemoryStore, memoryUnitDescriptor } from './memory-store.js'
import { createVectorIndex } from './vector-store.js'
import { buildExpandedQuery, MemoryRetriever } from './retrieval.js'
import type { RetrieveOptions } from './retrieval.js'
import { ScoreKeeper } from './scoring.js'
import { SleepCycle } from './sleep-cycle.js'
import { ScopeRegistry, scopeKeyOfAgent, scopeKeyOfSession, unitNameFor } from './scope.js'
import type { Config } from './config.js'
import type { MoodState, RetrievalResult } from './types.js'
import { WritePipeline } from './write-path.js'
import { createMemoryClearTool, createMemorySearchTool, formatMemories } from './tool.js'

/** Extract visible text from a message's content blocks. */
export function messageText(message: Message): string {
  return message.content
    .filter((block): block is Extract<typeof block, { type: 'text' }> => block.type === 'text')
    .map((block) => block.text)
    .join('')
    .trim()
}

/** Render retrieval winners as a prompt-section block (design §5.2). */
export { formatMemories }

interface SessionState {
  /** Rolling conversation tail for query expansion (capped at messageWindow). */
  recent: TurnMessage[]
  /** This turn's not-yet-processed messages. */
  pending: TurnMessage[]
  watermark: number
  /** 会话的工作区 scope 键（首事件时由 session.header.cwd 解析，M2）。 */
  scopeKey?: string
  /** 会话工作区是否在情绪编码白名单内（首事件时解析一次，P3-1）。 */
  emotionEnabled: boolean
  lastExpandedQuery?: string
  lastResult?: RetrievalResult
}

/** 一个作用域（工作区或回退）的完整运行时：库 + 四条链路 + 睡眠门控。 */
interface ScopedRuntime {
  readonly scopeKey: string | undefined
  readonly store: MemoryStore
  readonly retriever: MemoryRetriever
  readonly write: WritePipeline
  readonly sleep: SleepCycle
  readonly scores: ScoreKeeper
  /** 自上次睡眠周期后新写入的记忆数（§5.3 turn/end 信号门控）。 */
  newSinceSleep: number
  sleeping: boolean
}

export class MemoryRuntime {
  private readonly ctx: Context
  private readonly config: Config
  private readonly log: (msg: string, err?: unknown) => void
  private readonly sessions = new Map<string, SessionState>()
  /** Pending human input per agent, for the assembly-time retrieval query. */
  private readonly pendingQuery = new Map<string, string>()
  private scopes!: ScopeRegistry<ScopedRuntime>
  private embedding!: ReturnType<typeof createEmbeddingProvider>
  private reranker!: ReturnType<typeof createRerankerProvider>
  private llm!: CordisLlm
  private started = false

  private constructor(ctx: Context, config: Config) {
    this.ctx = ctx
    this.config = config
    const logger = ctx.logger('dsh-mem')
    this.log = (msg, err) => (err ? logger.error(msg, err) : logger.info(msg))
  }

  /** Async construction: opens the storage unit and wires all listeners. */
  static async start(ctx: Context, config: Config): Promise<MemoryRuntime> {
    const runtime = new MemoryRuntime(ctx, config)
    await runtime.start()
    return runtime
  }

  private async start(): Promise<void> {
    // Persistence goes through ctx.storage — never a private database (§5.0).
    // 此处只验证后端可用（未注册/缺 kv facet 时启动失败保持响亮）；
    // unit 的实际 open 延迟到首个作用域命中（ScopeRegistry.open）。
    const backend = this.ctx.storage.backend.get(this.config.storage.backend)
    if (!backend.kv) {
      throw new Error(`storage backend '${this.config.storage.backend}' provides no kv facet; ` +
        `install dsh-storage-json or dsh-storage-sqlite and check the storage.backend config`)
    }

    // 三条内部链路共用同一 embedding/reranker/llm（无调用间状态，跨作用域共享）
    this.embedding = createEmbeddingProvider(this.config.embedding)
    this.reranker = createRerankerProvider(this.config.reranker, this.embedding)
    this.llm = new CordisLlm(this.ctx, this.config.llm)

    this.scopes = new ScopeRegistry<ScopedRuntime>({
      open: async (scopeKey) => this.openScope(scopeKey),
      close: (rt) => rt.store.close(),
      onOpenError: (scopeKey, err) =>
        this.log(`scope open failed (unit ${unitNameFor(scopeKey, this.config)})`, err),
    })
    // 清理函数必须**返回** closeAll 的 Promise：Cordis 卸载会 await disposer，
    // 这样旧实例的 unit 全部关闭后才加载新实例。原先 `void closeAll()` 让卸载
    // 提前返回，HMR 重载时新实例抢开同一 unit 会撞上后端 "already open"。
    this.ctx.effect(() => () => this.scopes.closeAll())

    this.observeSessions()
    this.observeAgents()

    if (this.config.tool.enabled) {
      this.ctx.tools.register(createMemorySearchTool({
        resolve: async (agent) => {
          const rt = await this.scopedFor(scopeKeyOfAgent(agent, this.config))
          return {
            retriever: rt.retriever,
            recordRetrieval: (result: RetrievalResult) => void rt.scores.recordRetrieval(result),
            mood: this.moodContextFor(agent?.session?.header?.cwd, rt),
          }
        },
        recentTurns: (agent) => this.recentTurnsFor(agent),
        config: this.config,
      }))
    }

    // 开发/测试工具（默认关闭）：按作用域硬删除全部长期记忆——全系统唯一
    // 违反「记录永不删除」（§2.5）的操作，须 config.tool.clear 显式开启。
    if (this.config.tool.clear) {
      this.ctx.tools.register(createMemoryClearTool({
        resolve: async (agent) => {
          const rt = await this.scopedFor(scopeKeyOfAgent(agent, this.config))
          return {
            clear: async () => {
              const counts = await rt.store.clearAll()
              // 睡眠触发计数一并复位：清空后旧计数没有意义，避免下一轮
              // turn/end 用残留计数立即触发一次空池睡眠周期。
              rt.newSinceSleep = 0
              return counts
            },
          }
        },
      }))
    }

    this.started = true
    this.log(`started: backend=${this.config.storage.backend}, workspaceScope=${this.config.storage.workspaceScope}, clearTool=${this.config.tool.clear}`)
  }

  /** 打开一个作用域：unit 名由 scope 键派生，四条链路各持自己的 store 实例。 */
  private async openScope(scopeKey: string | undefined): Promise<ScopedRuntime> {
    const unitName = unitNameFor(scopeKey, this.config)
    const backend = this.ctx.storage.backend.get(this.config.storage.backend)
    if (!backend.kv) {
      throw new Error(`storage backend '${this.config.storage.backend}' provides no kv facet`)
    }
    const unit = await backend.kv.open(memoryUnitDescriptor(unitName, this.config))
    let store: MemoryStore
    try {
      store = await MemoryStore.open(unit, createVectorIndex(), this.config)
    } catch (err) {
      // 开库已占用后端 unit 槽：快照加载失败必须就地关闭，否则 ScopeRegistry
      // 删掉失败的缓存项后每次重试都撞 "already open"，该工作区永久不可用。
      await unit.close().catch(() => {})
      throw err
    }
    const rt: ScopedRuntime = {
      scopeKey,
      store,
      retriever: new MemoryRetriever({
        embedding: this.embedding,
        reranker: this.reranker,
        lookup: store,
        config: this.config.retrieval,
      }),
      write: new WritePipeline({
        llm: this.llm,
        embedding: this.embedding,
        store,
        config: this.config,
        log: (msg) => this.log(msg),
      }),
      sleep: new SleepCycle({
        llm: this.llm,
        embedding: this.embedding,
        store,
        config: this.config,
        log: (msg) => this.log(msg),
      }),
      scores: new ScoreKeeper({
        llm: this.llm,
        store,
        config: this.config,
        log: (msg) => this.log(msg),
      }),
      newSinceSleep: 0,
      sleeping: false,
    }
    this.log(`scope opened: unit=${unitName} (${store.validMemories().length} valid memories)`)
    // P3-2 图密度观测：决定 P3-3 PPR 何时有意义（边数/平均度过低时联想收益有限）
    const stats = graphStats(store.validMemories(), store.validNodes())
    this.log(`graph: ${stats.edges} edges over ${stats.items} items (avg degree ${stats.avgDegree.toFixed(2)})`)
    return rt
  }

  private scopedFor(scopeKey: string | undefined): Promise<ScopedRuntime> {
    return this.scopes.get(scopeKey)
  }

  // ---- write path: session/event observation (§5.1) ----

  private observeSessions(): void {
    // 入口总开关（M3）：false = 不观察 dsh 会话事件，完全停写；
    // 检索通道（memory_search 与自动注入）不受影响。
    if (!this.config.intake.session.enabled) return
    this.ctx.on('session/event', (session: Session, event: SessionEvent) => {
      const state = this.stateFor(session)
      switch (event.type) {
        case 'user/message': {
          const message = event.data as UserMessage
          if (!isConversationMessage(message, this.config.intake.session.acceptSourceKinds)) break
          this.pushMessage(state, event, 'user', messageText(message))
          break
        }
        case 'assistant/message': {
          const message = (event.data as { message: AssistantMessage }).message
          this.pushMessage(state, event, 'assistant', messageText(message))
          break
        }
        case 'turn/end': {
          const batch = state.pending
          state.pending = []
          state.watermark = event.seq
          if (batch.length === 0) break
          const turn = (event.data as { turn: number }).turn
          const sessionId = session.id
          const scopeKey = state.scopeKey
          const emotionEnabled = state.emotionEnabled
          // Fire-and-forget: memory consolidation never blocks the turn loop.
          // scope 开启失败（或写入失败）只影响本工作区本轮，日志旁路。
          void (async () => {
            try {
              const rt = await this.scopedFor(scopeKey)
              const outcome = await rt.write.processTurn(sessionId, turn, batch)
              if (outcome.written.length > 0 || outcome.superseded.length > 0) {
                this.log(`turn ${turn}: wrote ${outcome.written.length}, superseded ${outcome.superseded.length}, dropped ${outcome.dropped}`)
              }
              // P3-1 情绪聚合：turn 级观测并入 mood（先按 now 衰减历史值再 EMA）。
              // 写入管线本身对 mood 无感知；非白名单工作区丢弃观测（读取侧 bypass）。
              if (emotionEnabled && outcome.turnMood !== undefined) {
                try {
                  const next = blendMood(rt.store.getMood(), outcome.turnMood, Date.now(), this.config.emotion.tauHours)
                  await rt.store.setMood(next)
                } catch (err) {
                  this.log('mood blend failed', err)
                }
              }
              // 睡眠周期触发计数（§5.3：turn/end 信号 + 新记忆量门控，按作用域）
              rt.newSinceSleep += outcome.written.length
              this.maybeSleep(rt)
            } catch (err) {
              this.log('write pipeline failed', err)
            }
          })()
          break
        }
      }
    })
  }

  private pushMessage(state: SessionState, event: SessionEvent, role: 'user' | 'assistant', text: string): void {
    if (!text) return
    const msg: TurnMessage = { role, text }
    state.pending.push(msg)
    state.recent.push(msg)
    const cap = this.config.write.messageWindow * 2
    if (state.recent.length > cap) state.recent.splice(0, state.recent.length - cap)
    state.watermark = event.seq
  }

  private stateFor(session: Session): SessionState {
    const id = String(session.id)
    let state = this.sessions.get(id)
    if (!state) {
      state = {
        recent: [],
        pending: [],
        watermark: 0,
        // scope 键与情绪白名单都在会话首事件时解析一次（cwd 是创建期元数据，会话期间不变）
        scopeKey: scopeKeyOfSession(session, this.config),
        emotionEnabled: isEmotionEnabled(session.header?.cwd, this.config),
      }
      this.sessions.set(id, state)
    }
    return state
  }

  /**
   * 情绪重排上下文（P3-1）：仅白名单工作区；mood 读取时按 τ 惰性衰减
   * （衰减是投影，不回写——下一次 turn 情绪并入时才前移锚点）。
   */
  private moodContextFor(cwd: string | undefined, rt: ScopedRuntime): { state: MoodState; alpha: number } | undefined {
    if (!isEmotionEnabled(cwd, this.config)) return undefined
    const mood = rt.store.getMood()
    if (mood === null) return undefined
    return { state: decayMood(mood, Date.now(), this.config.emotion.tauHours), alpha: this.config.emotion.alpha }
  }

  // ---- automatic channel: agent-scoped prompt assembly (§5.2) ----

  private observeAgents(): void {
    if (!this.config.section.enabled) return

    // The claimed input precedes assembly in the turn flow, so it is the
    // best available query at step 1 of a turn.
    this.ctx.on('agent/inbox/claimed', ({ agent, message }: { agent: Agent; message: UserMessage }) => {
      if (message.source.kind !== 'user') return
      const text = messageText(message)
      if (text) this.pendingQuery.set(String(agent.id), text)
    })

    // assemble 监听器必须定域在 agent.ctx（scope 过滤分发的契约），但 agent.ctx
    // 上的 effect 只随 agent 卸载、不随插件卸载：HMR 重载后旧实例的监听器仍
    // 活着，会经已 closeAll 的 ScopeRegistry 把刚关掉的 unit 重新打开，新实例
    // 再开即撞宿主 already-open 保护。因此显式登记每个 disposer：agent 死亡
    // 时清登记，插件卸载时全部摘除（本 effect 注册晚于 closeAll effect，fiber
    // 逆序反注册保证先摘监听器、后关 unit）。
    const assembleDisposers = new Map<string, () => void>()
    let unwinding = false
    const attachAssemble = (agent: Agent): void => {
      const id = String(agent.id)
      if (unwinding || assembleDisposers.has(id)) return
      const off = agent.ctx.on('system-prompt/assemble', async (assembly, context, next) => {
        const downstream = await next()
        try {
          const text = await this.renderAutoSection(agent, context.signal)
          if (text) downstream.sections.push({ name: 'dsh-mem', text })
        } catch (err) {
          this.log('auto section retrieval failed; leaving prompt unchanged', err)
        }
        return downstream
      })
      assembleDisposers.set(id, off)
    }

    this.ctx.on('agent/disposed', ({ agent }: { agent: Agent }) => {
      this.pendingQuery.delete(String(agent.id))
      // agent 卸载已连带移除其监听器，这里只清登记防止 map 无界增长
      assembleDisposers.delete(String(agent.id))
    })

    // 0.2.x 起 agent/created 为 serial 事件：监听器需返回 undefined | Promise<undefined>（awaited）
    this.ctx.on('agent/created', ({ agent }: { agent: Agent }): undefined => {
      attachAssemble(agent)
      return undefined
    })
    // 重载（HMR）后已存活的 agent 不会再收到 agent/created：启动时对存量
    // agent 补挂，重载后自动注入才不中断。
    for (const agent of this.ctx.agents.list()) attachAssemble(agent)

    this.ctx.effect(() => () => {
      unwinding = true
      for (const off of assembleDisposers.values()) off()
      assembleDisposers.clear()
    })
  }

  private recentTurnsFor(agent: Agent | undefined): readonly string[] {
    if (!agent) return []
    const state = this.sessions.get(String(agent.id))
    return state ? state.recent.slice(-this.config.retrieval.contextTurns).map((m) => m.text) : []
  }

  private async renderAutoSection(agent: Agent, signal?: AbortSignal): Promise<string> {
    const agentId = String(agent.id)
    const state = this.sessions.get(agentId)
    const query = this.pendingQuery.get(agentId)
      ?? (state ? [...state.recent].reverse().find((m) => m.role === 'user')?.text : undefined)
    if (!query) return ''

    // scope 取自 agent 的会话（首个装配可能先于任何 session/event，此时
    // sessions 里还没有 state——scopeKeyOfAgent 直接读 session.header.cwd）
    const cwd = agent.session?.header?.cwd
    const rt = await this.scopedFor(scopeKeyOfAgent(agent, this.config))
    const mood = this.moodContextFor(cwd, rt)
    const recentTurns = (state ? state.recent : []).slice(-this.config.retrieval.contextTurns).map((m) => m.text)
    const expanded = buildExpandedQuery(query, recentTurns, this.config.retrieval.contextTurns)
    // Skip re-running the chain when the expanded query is unchanged within a
    // turn (steps after tool calls re-assemble the prompt).
    let body: string
    if (!state || state.lastExpandedQuery !== expanded || !state.lastResult) {
      const result = await rt.retriever.retrieve(query, {
        recentTurns,
        signal,
        topK: this.config.section.maxMemories,
        ...(mood !== undefined ? { mood } : {}),
      } satisfies RetrieveOptions)
      void rt.scores.recordRetrieval(result)
      if (state) {
        state.lastResult = result
        state.lastExpandedQuery = expanded
      }
      body = formatMemories(result, this.config.section.maxMemories, this.config.section.header)
    } else {
      void rt.scores.recordRetrieval(state.lastResult)
      body = formatMemories(state.lastResult, this.config.section.maxMemories, this.config.section.header)
    }
    // disclose（P3-1）：仅白名单工作区 + 开关开启 + 已有 mood 时附一行当前心境
    if (body !== '' && this.config.emotion.disclose && mood !== undefined) {
      body += `\n${describeMood(mood.state)}`
    }
    return body
  }

  // ---- offline consolidation cycle ("sleep", §2.3/§5.3) ----

  /**
   * 睡眠周期触发门控（§5.3，按作用域）：自上次周期后新写入 ≥
   * sleep.minNewMemories 时 fire-and-forget 执行一轮；周期内不重入（新写入
   * 计数保留，周期结束后立即补判）。聚类池不足 clusterMinSize 时不触发也
   * 不清零——等记忆继续累积。
   */
  private maybeSleep(rt: ScopedRuntime): void {
    if (!this.started || !this.config.sleep.enabled || rt.sleeping) return
    if (rt.newSinceSleep < this.config.sleep.minNewMemories) return
    const pool = rt.store
      .validMemories()
      .filter((record) => record.flags.compressed_into === undefined)
    if (pool.length < this.config.sleep.clusterMinSize) return
    rt.newSinceSleep = 0
    rt.sleeping = true
    void rt.sleep.run()
      .then((outcome) => {
        const migrated = outcome.clusters.filter((c) => c.route === 'new_node' || c.route === 'attached').length
        const gated = outcome.clusters.filter((c) => c.route === 'qa_failed' || c.route === 'epi_failed').length
        this.log(
          `sleep cycle: ${outcome.clusters.length} clusters (${migrated} migrated, ${gated} gated), ` +
          `refreshed ${outcome.refreshedRecords} records / ${outcome.refreshedNodes.length} nodes, ` +
          `decayed ${outcome.decayedRecords} records / ${outcome.decayedNodes} nodes`,
        )
      })
      .catch((err) => this.log('sleep cycle failed', err))
      .finally(() => {
        rt.sleeping = false
        // 周期运行期间积累的新写入立即补判一次
        this.maybeSleep(rt)
      })
  }
}

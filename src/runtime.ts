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
 */
import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { AssistantMessage, Message, UserMessage } from '@deepseek-ai/dsh-llm'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
// 拉入 system-prompt 包的声明合并，使 'system-prompt/assemble' 事件可类型化
import type {} from '@deepseek-ai/dsh-system-prompt'
import { createEmbeddingProvider, createRerankerProvider } from './embedding.js'
import type { TurnMessage } from './judges.js'
import { CordisLlm } from './llm.js'
import { MemoryStore, memoryUnitDescriptor } from './memory-store.js'
import { createVectorIndex } from './vector-store.js'
import { buildExpandedQuery, MemoryRetriever } from './retrieval.js'
import { ScoreKeeper } from './scoring.js'
import { SleepCycle } from './sleep-cycle.js'
import type { Config } from './config.js'
import type { RetrievalResult } from './types.js'
import { WritePipeline } from './write-path.js'
import { createMemorySearchTool, formatMemories } from './tool.js'

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

/** Only genuine human prompts become memories; injected/plugin contexts do not. */
function isHumanPrompt(message: Message): boolean {
  return message.role === 'assistant' || message.source.kind === 'user'
}

interface SessionState {
  /** Rolling conversation tail for query expansion (capped at messageWindow). */
  recent: TurnMessage[]
  /** This turn's not-yet-processed messages. */
  pending: TurnMessage[]
  watermark: number
  lastExpandedQuery?: string
  lastResult?: RetrievalResult
}

export class MemoryRuntime {
  private readonly ctx: Context
  private readonly config: Config
  private readonly log: (msg: string, err?: unknown) => void
  private readonly sessions = new Map<string, SessionState>()
  /** Pending human input per agent, for the assembly-time retrieval query. */
  private readonly pendingQuery = new Map<string, string>()
  private store!: MemoryStore
  private retriever!: MemoryRetriever
  private write!: WritePipeline
  private sleep!: SleepCycle
  private scores!: ScoreKeeper
  private started = false
  /** 自上次睡眠周期后新写入的记忆数（§5.3 turn/end 信号门控）。 */
  private newSinceSleep = 0
  private sleeping = false

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
    const backend = this.ctx.storage.backend.get(this.config.storage.backend)
    if (!backend.kv) {
      throw new Error(`storage backend '${this.config.storage.backend}' provides no kv facet; ` +
        `install dsh-storage-json or dsh-storage-sqlite and check the storage.backend config`)
    }
    const unit = await backend.kv.open(memoryUnitDescriptor(this.config))
    this.ctx.effect(() => () => {
      void unit.close().catch(() => {})
    })

    this.store = await MemoryStore.open(unit, createVectorIndex(), this.config)

    const embedding = createEmbeddingProvider(this.config.embedding)
    const reranker = createRerankerProvider(this.config.reranker, embedding)
    this.retriever = new MemoryRetriever({
      embedding,
      reranker,
      lookup: this.store,
      config: this.config.retrieval,
    })
    // 三条内部链路共用同一 ctx.llm 路由（CordisLlm 无调用间状态）
    const llm = new CordisLlm(this.ctx, this.config.llm)
    this.write = new WritePipeline({
      llm,
      embedding,
      store: this.store,
      config: this.config,
      log: (msg) => this.log(msg),
    })
    this.sleep = new SleepCycle({
      llm,
      embedding,
      store: this.store,
      config: this.config,
      log: (msg) => this.log(msg),
    })
    this.scores = new ScoreKeeper({
      llm,
      store: this.store,
      config: this.config,
      log: (msg) => this.log(msg),
    })

    this.observeSessions()
    this.observeAgents()

    if (this.config.tool.enabled) {
      this.ctx.tools.register(createMemorySearchTool({
        retriever: this.retriever,
        recentTurns: (agent) => this.recentTurnsFor(agent),
        onRetrieved: (result) => void this.scores.recordRetrieval(result),
        config: this.config,
      }))
    }

    this.started = true
    this.log(`started: ${this.store.validMemories().length} valid memories in store`)
  }

  // ---- write path: session/event observation (§5.1) ----

  private observeSessions(): void {
    this.ctx.on('session/event', (session, event: SessionEvent) => {
      const state = this.stateFor(session.id)
      switch (event.type) {
        case 'user/message': {
          const message = event.data as UserMessage
          if (!isHumanPrompt(message)) break
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
          // Fire-and-forget: memory consolidation never blocks the turn loop.
          void this.write.processTurn(sessionId, turn, batch)
            .then((outcome) => {
              if (outcome.written.length > 0 || outcome.superseded.length > 0) {
                this.log(`turn ${turn}: wrote ${outcome.written.length}, superseded ${outcome.superseded.length}, dropped ${outcome.dropped}`)
              }
              // 睡眠周期触发计数（§5.3：turn/end 信号 + 新记忆量门控）
              this.newSinceSleep += outcome.written.length
              this.maybeSleep()
            })
            .catch((err) => this.log('write pipeline failed', err))
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

  private stateFor(sessionId: string): SessionState {
    let state = this.sessions.get(sessionId)
    if (!state) {
      state = { recent: [], pending: [], watermark: 0 }
      this.sessions.set(sessionId, state)
    }
    return state
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
    this.ctx.on('agent/disposed', ({ agent }: { agent: Agent }) => {
      this.pendingQuery.delete(String(agent.id))
    })

    this.ctx.on('agent/created', ({ agent }: { agent: Agent }) => {
      // Agent-scoped registration: contributions unwind with the agent
      // (architecture doc: scope registrations to one agent via agent.ctx).
      agent.ctx.on('system-prompt/assemble', async (assembly, context, next) => {
        const downstream = await next()
        try {
          const text = await this.renderAutoSection(String(agent.id), context.signal)
          if (text) downstream.sections.push({ name: 'dsh-mem', text })
        } catch (err) {
          this.log('auto section retrieval failed; leaving prompt unchanged', err)
        }
        return downstream
      })
    })
  }

  private recentTurnsFor(agent: Agent | undefined): readonly string[] {
    if (!agent) return []
    const state = this.sessions.get(String(agent.id))
    return state ? state.recent.slice(-this.config.retrieval.contextTurns).map((m) => m.text) : []
  }

  private async renderAutoSection(agentId: string, signal?: AbortSignal): Promise<string> {
    const state = this.sessions.get(agentId)
    const query = this.pendingQuery.get(agentId)
      ?? (state ? [...state.recent].reverse().find((m) => m.role === 'user')?.text : undefined)
    if (!query || !state) return ''

    const recentTurns = state.recent.slice(-this.config.retrieval.contextTurns).map((m) => m.text)
    const expanded = buildExpandedQuery(query, recentTurns, this.config.retrieval.contextTurns)
    // Skip re-running the chain when the expanded query is unchanged within a
    // turn (steps after tool calls re-assemble the prompt).
    if (state.lastExpandedQuery !== expanded || !state.lastResult) {
      state.lastResult = await this.retriever.retrieve(query, {
        recentTurns,
        signal,
        topK: this.config.section.maxMemories,
      })
      state.lastExpandedQuery = expanded
      void this.scores.recordRetrieval(state.lastResult)
    }
    return formatMemories(state.lastResult, this.config.section.maxMemories, this.config.section.header)
  }

  // ---- offline consolidation cycle ("sleep", §2.3/§5.3) ----

  /**
   * 睡眠周期触发门控（§5.3）：自上次周期后新写入 ≥ sleep.minNewMemories 时
   * fire-and-forget 执行一轮；周期内不重入（新写入计数保留，周期结束后立即
   * 补判）。聚类池不足 clusterMinSize 时不触发也不清零——等记忆继续累积。
   */
  private maybeSleep(): void {
    if (!this.started || !this.config.sleep.enabled || this.sleeping) return
    if (this.newSinceSleep < this.config.sleep.minNewMemories) return
    const pool = this.store
      .validMemories()
      .filter((record) => record.flags.compressed_into === undefined)
    if (pool.length < this.config.sleep.clusterMinSize) return
    this.newSinceSleep = 0
    this.sleeping = true
    void this.sleep.run()
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
        this.sleeping = false
        // 周期运行期间积累的新写入立即补判一次
        this.maybeSleep()
      })
  }
}

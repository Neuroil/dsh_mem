/**
 * dsh-mem — semantic-driven long-term memory for DeepSeek Harness.
 *
 * P0 scope (design §6): plugin skeleton, data model on ctx.storage, the
 * write path (M1 embedding / M5 rubric scoring / simplified contradiction)
 * and the retrieval chain (M1 two-stage + M2 CSLS + M3 MMR) exposed through
 * both channels: automatic prompt-section injection and the memory_search
 * tool.
 *
 * Plugin contract (§5.0): named exports only — a default export would lose
 * the inject metadata. Every registration below is an effect and unwinds in
 * reverse order on unload; config changes reload the plugin via HMR.
 */
import type { Context } from '@deepseek-ai/cordis'
import { Config, type Config as ConfigType } from './config.js'
import { MemoryRuntime } from './runtime.js'

export { Config }
export type { ConfigType }

export const name = 'dsh-mem'

export const inject = ['storage', 'llm', 'tools', 'systemPrompt', 'sessions']

export function apply(ctx: Context, config: ConfigType): void {
  const logger = ctx.logger('dsh-mem')
  MemoryRuntime.start(ctx, config).catch((err: unknown) => {
    // Startup failures are loud but do not kill the host process.
    logger.error('failed to start: %o', err)
  })
}

export { MemoryRuntime }

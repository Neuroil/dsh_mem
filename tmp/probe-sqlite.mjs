/**
 * 一次性探针（M1 验证）：确认 sqlite 后端在当前宿主上可注册、kv.open 可开、
 * putRecord/loadAll 可读写。手法沿用 .agent 调试笔记第五节（宿主 console 被压，
 * 用文件副作用观测）。
 */
import { appendFileSync } from 'node:fs'

export const name = 'probe-sqlite'
export const inject = ['storage']

const OUT = 'C:/Users/Neuroil/Code/dsh_mem/tmp/probe-result.txt'
const write = (line) => appendFileSync(OUT, line + '\n')

export async function apply(ctx) {
  write(`probe loaded at ${new Date().toISOString()}`)
  try {
    // 后端注册行与本探针都 inject storage，先后次序无保证——轮询等待后端出现
    // （registry.get 未注册时抛 StorageError，不是返回 undefined）
    let backend = undefined
    for (let i = 0; i < 50; i++) {
      try {
        backend = ctx.storage.backend.get('sqlite')
      } catch {
        backend = undefined
      }
      if (backend?.kv) break
      await new Promise((r) => setTimeout(r, 100))
    }
    if (!backend?.kv) throw new Error("backend 'sqlite' not registered after 5s (kv facet missing)")

    const unit = await backend.kv.open({
      name: 'probe_sqlite',
      version: 1,
      tables: ['t'],
      hasGlobal: false,
    })
    await unit.putRecord('t', 'k1', { hello: 'sqlite', arr: [1, 2, 3], nested: { ok: true } })
    const snap = await unit.loadAll()
    await unit.close()

    const row = snap.tables?.t?.k1
    if (row?.hello !== 'sqlite') throw new Error(`roundtrip mismatch: ${JSON.stringify(row)}`)
    write(`OK: registered, opened, wrote, read back: ${JSON.stringify(row)}`)
  } catch (err) {
    write(`FAIL: ${String(err?.stack ?? err)}`)
  }
}

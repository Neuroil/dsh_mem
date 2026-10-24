/**
 * 顺序执行全部 benchmark 并汇总（保证计时互不抢 CPU）。
 * 各脚本单独可跑（npm run insert / retrieval / write / sleep / space），
 * 结果 JSON 落在 results/；本入口只做编排与环境摘要。
 */
import { spawnSync } from 'node:child_process'
import { readFileSync, existsSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

const BENCHES = ['bench-insert.mjs', 'bench-retrieval.mjs', 'bench-write.mjs', 'bench-sleep.mjs', 'bench-space.mjs']

const t0 = performance.now()
const outcomes = []
for (const bench of BENCHES) {
  console.log(`\n########## ${bench} ##########`)
  const res = spawnSync(process.execPath, [join(import.meta.dirname, bench)], {
    stdio: 'inherit',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  })
  outcomes.push({ bench, status: res.status, signal: res.signal })
  if (res.status !== 0) {
    console.error(`${bench} 失败（exit ${res.status}${res.signal ? ' / ' + res.signal : ''}），中止聚合`)
    process.exitCode = 1
    break
  }
}

// 汇总快照：把各 results/*.json 拼成一份带环境头的结果集
const summary = {
  meta: {
    date: new Date().toISOString(),
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    cpu: `${os.cpus()[0]?.model ?? '?'} × ${os.cpus().length}`,
    totalRuntimeSec: Math.round((performance.now() - t0) / 1000),
  },
  outcomes,
  benches: {},
}
for (const bench of BENCHES) {
  const name = bench.replace('bench-', '').replace('.mjs', '')
  const path = join(import.meta.dirname, 'results', `${name}.json`)
  if (existsSync(path)) summary.benches[name] = JSON.parse(readFileSync(path, 'utf8'))
}
writeFileSync(join(import.meta.dirname, 'results', 'all.json'), JSON.stringify(summary, null, 2))
console.log(`\n全部完成：${outcomes.filter((o) => o.status === 0).length}/${BENCHES.length} 项成功，总耗时 ${summary.meta.totalRuntimeSec}s；聚合结果 results/all.json`)

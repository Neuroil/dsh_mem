# TODO

> 2026-10-24 文档同步检查 + SQLite 交互逻辑审查的产出。
> 检查基线：05354e7（工作树干净）、`npm test` 202/202 全绿、e2e 产物 `tmp/e2e-write/store.sqlite` 实测核验。
> 结论：SQLite 交互逻辑本身无阻断问题；以下为文档同步遗留与记录在案的小边界。

## 高优先级

- [x] **README「方式 A」仍指向 `src/index.ts`**（`README.md:136`，即调试笔记 R3 遗留项）——**已修（2026-10-24）**
  改为指向 `lib/index.js`，并在该节加了显式警告（必须先 `npm run build`；宿主 loader 不转译 TS，指 `src/index.ts` 会静默加载失败）。

- [x] **升级 dsh-\* 依赖并重新验证**（即调试笔记 R4 建议项）——**已修（2026-10-24）**
  全部 `@deepseek-ai/dsh-*` 升到精确 `0.2.1-rc.2`（与宿主对齐；rc.3 会被 peer 闸拒故不取）；移除了零引用的 `dsh-code-runtime`（registry 无 0.2.x 线）与 `dsh-invariants`（其他 dsh 包的传递依赖，需要时仍会被装上）。升级后唯一 API 漂移：`agent/created` 变为 serial 事件、监听器需返回 `undefined | Promise<undefined>`——`src/runtime.ts:293` 监听器已改 async。`npm run typecheck && npm test && npm run build` 全绿（202/202）。

## 中优先级

- [ ] **启动失败可见性**（即调试笔记 R2 建议项）
  `src/index.ts:27-30` 的 catch 只调 `logger.error`，而宿主 console 不输出插件日志，启动失败完全静默——这正是当初「插件装了但像没装」排障困难的根源。在 catch 里补一条粗粒度可见信号，如 `console.error('[dsh-mem] failed to start:', err)`。

- [ ] **全景图 §17 标题过期**（小瑕疵）
  `docs/architecture/插件工作原理全景图.md:589` 标题仍写「M1/M2/M3 后」，但表内已含 M4 收尾的行（写入路径端到端）。改为「M1–M4 后」之类的表述。

## 已知边界（非错误，记录在案，暂不处理）

以下三项在 SQLite 交互逻辑审查中确认为可接受边界；若后续使用场景扩大再升级为任务：

- **嵌入维度不匹配静默截断**：`cosine` 对两向量取 `min(a.length, b.length)`（`src/metrics.ts:16`）。换嵌入模型后，存量旧维度记录不会报错，只会产生无意义相似度。README 已提示切换时对齐 `embedding.dimensions`；如需防护，可在 `MemoryStore.open()` 加载时校验记录维度与当前配置不一致并警告。
- **并发读改写窗口**：`updateRecord`（`src/memory-store.ts:229`）是内存镜像上的 read-modify-write，无按键串行化。同一工作区多会话并发写同一 id 时后持久化者胜——镜像最终与介质一致、不脱节，但可能丢一次中间更新。当前触发面极窄（如 r_m 刷新与软失效恰好同时命中同一记录）。
- **`ScopeRegistry.closeAll` 后 `get()` 仍可懒重开**：`src/scope.ts:89` 不检查 `closed` 标志。插件卸载后监听器已随 ctx 回收，实际路径打不到；追求严格可在 `get()` 入口加检查。
- **npm audit 17 项告警（1 moderate / 14 high / 2 critical）**：集中在 vitest/tinypool（dev-only，不随包发布）与 @deepseek-ai 包内嵌的 zod 旧副本（<4.6.6，由上游版本范围控制，插件侧只能 override 强扭）。上游发版后重装即消，暂不处理。
- **F-10 睡眠触发计数在周期运行前清零**（全景图 F 表）：周期整体抛错时本轮累计不回补，需再攒 `minNewMemories` 条新写入才重试——自愈型低危；修法是 catch 分支回补或把清零移到成功后。

## 待决定

- [ ] `@deepseek-ai/dsh-storage-domain` devDep 去留：实现仅用 `ctx.storage`（类型来自 `@deepseek-ai/dsh-storage`），`ctx.storageDomain` seam 未使用、src/tests 零引用（设计文档 §4.1/§5.0 原计划两者并用，演进注 ④ 已注记分歧）。移除或保留需定一句口径。
- [ ] `@deepseek-ai/dsh-message-feedback` **已发布**（2026-10-24 npm 实查 0.0.1-rc.2）——README/全景图原先「message-feedback 包类型未发布到 npm / 不在依赖集」的前提已部分失效。「被采纳」确认制接线重新可行：评估把该包纳入依赖集、核对 `feedback/committed` payload 类型与宿主 `ctx.messageFeedback` 是否一致；若接线，S 增益从「未被纠正默认即时入账」切为确认制（README §3a、全景图 §10.1）。
- [ ] `config.ts` 的 llm schema 默认值（`deepseek` / `deepseek-chat`）是否改为宿主实际注册名（`deepseek-official` / `deepseek-v4-flash`）：`cordis.patch.yml` 与 README 示例已改为实际名，schema 默认仍是中性占位——裸配置在 dsh-llm-deepseek 宿主上会 NO_ADAPTER（写入管线空转、无报错弹窗）。

## 后续里程碑候选（明确不做，按需启动）

归档自 README「明确不在 P2」与计划文件「明确不做」；均非使用阻断项，启动前先在全景图立 F 项：

- [ ] embedding int8 + base64 量化（换真实嵌入端点、库变大后做）；
- [ ] 语义图 PPR 联想检索（HippoRAG 式 multi-hop；SemanticNode 已预留 `links` 图结构）；
- [ ] spacing 加权检索、sur/valence 情绪一致性检索（P3，设计 §6/§10）；
- [ ] LoCoMo 式活宿主端到端回放（验收长程问答提升，需真实 LLM + 长宿主）；
- [ ] messageFeedback「被采纳」确认制（前置评估见上方待决定）。
- [ ] `.agent/debug/` 下三个未跟踪文件的去留：`dsh_mem.backup.json`（介质备份）、`inspect-sqlite.mjs`、`repro-retrieval.mjs`（调试脚本）。排查笔记 `unit命名不规范.md` 已入库；这三个要么提交、要么加进 .gitignore。

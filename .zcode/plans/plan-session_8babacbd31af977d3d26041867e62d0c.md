# dsh-mem 重构：sqlite 后端 + 按工作区隔离 + 入口可配置

（计划全文将按你的要求写入根目录 `plan-10-24_21-55.md` 供其他 agent 同步）

## 核心设计决定

- **scope = 会话创建时的工作目录** `session.header.cwd`（宿主验证过的绝对路径）；每 scope 一个 KvUnit：`dsh_mem_ws_<8位fnv1a-hex>`（路径归一化：小写、反斜杠→正斜杠），物理隔离；无 cwd 会话回退 `config.storage.unit` 本名（向后兼容）。
- 旧 `dsh_mem.json` **不迁移**（deterministic 嵌入向量无语义价值），原地保留。
- 写入/睡眠/评分链路本就是窄接口（`MemoryStoreLike`/`SleepStoreLike`），judges/sleep-cycle/scoring/write-path 零改动；睡眠周期与计分按 scope 各自实例与门控。
- 入口配置：`intake.session.acceptSourceKinds`（默认 `['user']`）+ form 守卫（带 `form` 的消息是插件上下文，不算对话）。

## M1 — sqlite 后端（配置为主）
`dsh plugin --profile web add @deepseek-ai/dsh-storage-sqlite`（npm 预构建，后端名 `sqlite`，基于 node:sqlite，Node 24.14.1 ✓）→ `--dump-config`/探针验证 → patch 改 `storage.backend: sqlite` → 跑对话验证介质。回滚 = 改回 json（纯配置）。

## M2 — 工作区隔离（主要代码）
新建 `src/scope.ts`（路径归一/哈希/unitNameFor/cwdOfSession/cwdOfAgent）；`src/runtime.ts` 重构为 `Map<scopeKey, ScopedRuntime>` 惰性开启 + `ctx.effect` 统一 close，写入按 `session.header.cwd` 路由，注入/工具按 agent 会话 cwd 解析，单 scope 开启失败不拖垮全局；`config.ts` 增 `storage.workspaceScope`（默认 true）；`memoryUnitDescriptor` 参数化；`tool.ts` deps 改按 agent 解析 scope。测试：哈希稳定性/RE 合规/双工作区互不可见/无 cwd 回退。

## M3 — 入口可配置
config 新段 `intake.session.{enabled, acceptSourceKinds}`；isHumanPrompt 配置化 + form 守卫；过滤逻辑提纯为独立可测函数。测试：自定义 kind 接受/拒绝、form 拒绝、enabled=false 停写不停检索。

## M4 — 验证与文档
typecheck + vitest + build（宿主加载 lib/，必须 build）；真机双工作区验证；同步 `docs/architecture/插件工作原理全景图.md`、README 配置速览、设计文档 §5.0/§5.1 附注；版本 0.2.0；git 按里程碑分次提交。

## 风险
sqlite 包 0.0.1-rc.1 对宿主 0.1.2-rc.1 兼容性无官方说明 → 探针先行，失败回退 json（M2/M3 不依赖后端）；多进程写争用无重试 → 单进程使用可接受；agent→cwd 访问器实现时对照 dsh-agent 类型定稿（Session 侧已确认 `session.header.cwd`）。

## 不做（后续候选）
embedding 量化、全局偏好层、QQ 分群 scope、外部历史导入。
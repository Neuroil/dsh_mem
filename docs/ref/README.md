# dsh（DeepSeek Harness）插件开发参考资料

> 目标：为「聊天 bot 记忆管理插件」项目沉淀 dsh 插件开发的**契约与方法**。
> 以下文件均为官方文档中文版原文，直接抓取自
> [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) 仓库 `docs/` 目录（master 分支，抓取于 2026-10-24），
> 保持了官方目录结构以便溯源。

## dsh 是什么

DeepSeek Harness（`dsh`）是 DeepSeek 开源的 agent harness，核心架构是 **"一切皆插件"（everything is a plugin）**：
基于 Cordis 插件框架（vendored），没有特权内核——模型适配器、工具注册表、会话日志、甚至 agent 主循环本身都是插件。
产品启动时由分层配置（`cordis.yml` patch 层）组装出一棵插件树。

- 官方仓库：https://github.com/deepseek-ai/deepseek-harness
- 文档站：https://deepseek-harness.github.io/deepseek-harness/
- 安装运行：`npx @deepseek-ai/dsh web`（Web UI 在 http://127.0.0.1:3080）
- 注意：developer preview 阶段，官方明确警告会有 breaking changes。

---

## 文件索引

### 插件开发主路径（必读，按顺序）

| 文件 | 内容 |
|---|---|
| [docs/user/develop/basic/index.zh.md](docs/user/develop/basic/index.zh.md) | **第一个插件**：插件 = 导出 `apply(ctx)` 的模块；三种插件形态（函数/对象/Service 类）；`inject` 声明依赖；`ctx.effect()` 自动清理；`--patch` 本地加载 |
| [docs/user/develop/basic/tool.zh.md](docs/user/develop/basic/tool.zh.md) | **开发一个工具**：`defineTool` DSL 最小示例；schema 自动流入系统提示词 |
| [docs/user/develop/basic/config.zh.md](docs/user/develop/basic/config.zh.md) | **插件配置**：Schemastery `Schema.object` 定义 Config；默认值；"无硬编码可调参数"约定；配置错误要响亮；配置变更触发 HMR |
| [docs/user/develop/basic/publish.zh.md](docs/user/develop/basic/publish.zh.md) | **打包与安装**：组合包（bundle）vs profile 两种 manifest；`dsh plugin add`；加载顺序四层；patch 按 id 整行覆盖不深度合并；git 安装的 `prepare` 脚本与 pnpm≥10 `allowBuilds` 授权坑 |
| [docs/user/develop/framework/index.zh.md](docs/user/develop/framework/index.zh.md) | **插件与生命周期**：Fiber 状态机（PENDING→LOADING→ACTIVE→…）；依赖驱动的加载/自动卸载重载；自动清理机制清单；`ctx.plugin()` 嵌套；dispose 语义；HMR |
| [docs/user/develop/framework/service.zh.md](docs/user/develop/framework/service.zh.md) | **服务与依赖**：Service 基类与 `ctx.<key>` 挂载；`declare module` 类型声明合并；必需依赖（inject）vs 可选依赖（`ctx.get()`）；服务消失时自动 dispose/重载；服务隔离（isolate） |
| [docs/user/develop/framework/events.zh.md](docs/user/develop/framework/events.zh.md) | **事件系统**：`emit`/`bail`/`serial`/`waterfall` 四种分发模式；waterfall 必须调 `next()`；类型化事件（声明合并）；Cordis 事件 vs 持久化会话事件（`session/event`） |
| [docs/user/develop/practice/index.zh.md](docs/user/develop/practice/index.zh.md) | **能力三角色**：Service Definition / Provider / Consumer 分包模式；以 Bash（dsh-shell / dsh-bash-local / dsh-tool-bash）为例；"不要预防性拆分" |
| [docs/user/develop/practice/llm-adapter.zh.md](docs/user/develop/practice/llm-adapter.zh.md) | 实现 LLM 适配器（`ctx.llm.registerAdapter`） |
| [docs/user/develop/practice/dynamic-cordis.zh.md](docs/user/develop/practice/dynamic-cordis.zh.md) | 运行时动态组装 Cordis（短篇） |

### 深度参考（工具/扩展点契约的真源）

| 文件 | 内容 |
|---|---|
| [docs/cookbook/adding-a-tool.zh.md](docs/cookbook/adding-a-tool.zh.md) | **工具编写参考（权威）**：`execute()` 全部约定——参数自动校验、规范 JSON 返回值、异常=isError、遵守 `exec.signal`、`presentationMeta`、`exec.agent.inject()` 异步通知；后台任务 `ctx.jobs.start()`；执行策略五扩展点；PTC mode；UI 卡片（presentCall/presentResult，generic/terminal/diff/read/search/web）及"展示器必须是纯函数"等硬性规则 |
| [docs/cookbook/extension-cookbook.zh.md](docs/cookbook/extension-cookbook.zh.md) | **扩展插件形态实操手册**：权限门禁钩子示例（`tools/pre-execute` 返回 `{kind:'deny'}`）；UI 插件；外部协议驱动（ACP）；**功能→机制映射表**（每个产品功能对应哪个扩展点；其中"记忆"一行 = **section 提供方 + 工具**） |
| [docs/reference/capability-seams.zh.md](docs/reference/capability-seams.zh.md) | 能力接缝完整参考（~68KB，全量 seam 目录） |
| [docs/reference/event-producer-consumer.zh.md](docs/reference/event-producer-consumer.zh.md) | 事件生产者/消费者模式详解 |
| [docs/reference/tool-execution-pipeline.zh.md](docs/reference/tool-execution-pipeline.zh.md) | 工具执行管线（从 tool/call 到 tool/result 的完整流程） |
| [docs/reference/agent-lifecycle.zh.md](docs/reference/agent-lifecycle.zh.md) | Agent 生命周期（turn/step 结构） |
| [docs/reference/architecture.zh.md](docs/reference/architecture.zh.md) | 整体架构与微内核声明 |
| [docs/reference/cordis-primer.zh.md](docs/reference/cordis-primer.zh.md) | Cordis 概念精简版（plugin/context/inject/events/可逆注册） |
| [docs/reference/glossary.zh.md](docs/reference/glossary.zh.md) | 术语表 |

### Cordis 框架教程（7 章，无需 API key 可跑）

目录：[docs/cordis-tutorial/](docs/cordis-tutorial/) —— 索引见 [index.zh.md](docs/cordis-tutorial/index.zh.md)

1. [第一个插件](docs/cordis-tutorial/01-first-plugin.zh.md)：插件是函数，由 loader 挂载
2. [生命周期与 effect](docs/cordis-tutorial/02-lifecycle-and-effects.zh.md)：注册随 fiber 卸载自动撤销
3. [服务](docs/cordis-tutorial/03-services.zh.md)：`ctx` 上公开能力 + `inject`
4. [事件](docs/cordis-tutorial/04-events.zh.md)：类型化事件、waterfall 短路
5. [配置](docs/cordis-tutorial/05-config.zh.md)：`cordis.yml` 校验配置、错误响亮
6. [组合与 HMR](docs/cordis-tutorial/06-composition-and-hmr.zh.md)：配置文件即插件树、热重载、诊断永不加载的插件
7. [进入 harness](docs/cordis-tutorial/07-into-the-harness.zh.md)：在真实 harness 服务上注册模型可调用工具

### Cordis 核心 API 参考

目录：[docs/cordis-api/](docs/cordis-api/) —— [context](docs/cordis-api/context.zh.md) · [events](docs/cordis-api/events.zh.md) · [fiber](docs/cordis-api/fiber.zh.md) · [registry](docs/cordis-api/registry.zh.md) · [service](docs/cordis-api/service.zh.md)

---

## 契约速览（提炼版）

### 1. 插件本体契约

```ts
import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'

export const name = 'my-plugin'          // 必需（函数形态）
export const inject = ['tools']          // 可选：必需依赖，就绪后才跑 apply
export interface Config { /* ... */ }
export const Config: Schema<Config> = Schema.object({ /* 默认值写在 schema */ })

export function apply(ctx: Context, config: Config) {
  // 在这里注册能力；导出 Config 时签名为 (ctx, config)，否则 (ctx)
}
```

- **三种形态**：函数（最常用）/ 对象（`export default { name, inject, apply }`）/ Service 子类（`export default class extends Service`，向其他插件提供 `ctx.<key>` 服务）。
- **函数形态必须具名导出** `name`/`inject`/`Config`/`apply`，不能用 default export（会丢 inject 元数据）。
- **必需依赖用 `inject`，可选依赖用 `ctx.get(name)`**（不要用属性代理）。
- **一切注册皆 effect**：`ctx.on`、`ctx.tools.register`、`ctx.effect(() => cleanup)` 在插件卸载时自动逆序清理；手动资源必须放进 `ctx.effect()` 返回 disposer。
- Fiber 状态机：`PENDING → LOADING → ACTIVE → UNLOADING → DISPOSED`（`apply` 抛异常 → FAILED，进程按设计终止）。
- 依赖的服务消失 → 插件自动卸载，服务恢复 → 自动重载（HMR 同理）。

### 2. 工具契约（`defineTool`，来自 `@deepseek-ai/dsh-tools`）

```ts
ctx.tools.register(defineTool({
  name: 'read_file',
  description: '...',                    // 模型可见
  parameters: {                          // 自动推导 + 校验 args 类型
    path: { type: 'string', required: true, description: 'Absolute path' },
    limit: { type: 'number' },           // 缺省即可选
  },
  output: {
    schema: { type: 'string' },          // execute 返回值的规范 schema
    render: (_args, value) => [{ type: 'text', text: value }],  // 规范值 → 模型可见内容
    // presentationMeta?: (args, value) => 可回放 JSON（供 UI 卡片，可选）
  },
  // presentCall / presentResult?: UI 卡片渲染意图（可选，必须是纯函数）
  async execute(args, exec) {
    // args 已按 schema 校验；exec.signal 必须遵守（取消）
    return 规范JSON值                    // 抛异常 = isError
  },
}))
```

`execute()` 硬性规则：
- 参数已校验，`args` 视为只读；`exec`（callId/name/arguments/agent/token/signal）全程不可变。
- 只返回**一个规范 JSON 值**（对象 schema 必须显式 `additionalProperties: true|false`）；不要返回内容块、不要让调用方解析自然语言。
- 抛异常或返回无效值 = `isError`；领域性"不理想结果"（如非零退出码）写进规范值。
- 异步通知用 `exec.agent.inject({ content, source: { kind: 'my-plugin' } })`（追加持久化上下文，下次请求可见）。
- 长任务：producer 配置 `run_in_background` + `ctx.jobs.start({ kind, label, owner: exec.agent, run })`，成功返回 `{ kind: 'background', jobId }`。
- 注册后定义只读；热替换 = dispose 旧 effect 再注册新的；同名只能注册一次。

### 3. 事件契约

四种分发模式：`emit`（广播）/ `bail`（短路取第一个真值）/ `serial`（顺序等异步）/ `waterfall`（流水线，**必须调 `next()`**，不调即拦截）。

关键区分：
- **Cordis 运行时事件**：`tools/*`、`agent/*` 等_live 协调，不落盘。
- **持久化会话事件**：`turn/*`、`step/*`、`user/message`、`assistant/*`、`tool/call`、`tool/result`、`compaction/*`、`todo/write`——追加进会话日志、重启存活、模型可见；要观察它们需监听 `session/event` 并检查 `event.type`（事件信封 `{ seq, time, type, data }`，`session.events` 可遍历）。

### 4. 扩展点速查（waterfall 默认，标注除外）

| 目的 | 机制 |
|---|---|
| 允许/拒绝/询问某次工具调用 | `tools/pre-execute` 返回 `{kind:'allow'\|'deny'\|'ask', reason}` |
| 不可推翻的单调拒绝 | `ctx.tools.guard()` |
| 包裹分发（超时/重试/指标；仅 exec.signal 可替换） | `tools/execute` |
| 改写结果/展示/附加模型可见上下文 | `tools/post-execute` |
| 观测不可变最终结果（审计） | `tools/result` |
| 改写模型请求配置 | `agent/request`（返回替换后的 LlmCallConfig） |
| 拦截/改写进入 step 的消息 | `agent/pre-step` |
| 决定是否结束/继续 turn | `agent/turn-stopping`（serial，无 next） |
| 系统提示词装配 | `ctx.systemPrompt.section()`（排序 + 作用域）；`system-prompt/assemble` 整体变换 |
| 工具可见集过滤 | `ctx.tools.restrict()`（展示/查找/执行三者对齐） |

官方"功能→机制映射"明确：**记忆类能力 = 系统提示词 section 提供方 + 工具**（见 extension-cookbook.zh.md 末表）。

### 5. 配置与加载契约

- Config 用 Schemastery：`Schema.string().default(...)`、`Schema.union([...])` 等；加载时校验，非法即失败且报错明确；不要导出普通对象当 Config。
- 生效配置 = 空根之上按序叠加四层：① profile 的 bundles 列表各组合包 patch（按安装顺序）→ ② profile 自己的 `cordis.patch.yml` → ③ `$DSH_HOME/cordis.patch.yml`（机器级）→ ④ 各 `--patch <path>`（argv 顺序）。后层按行 id **整行覆盖**（不深度合并，改一个键也要重述整行 config）。
- `!!js` 表达式只允许出现在插件 `config` 和条目 `disabled` 字段里。
- 本地开发：`dsh web --patch ./overlay.yml`（插件路径必须**绝对路径**）；`--dump-config` 检查最终组合。

### 6. 打包分发契约

- 组合包 `package.json` 带 `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }`；patch 里插件行用包名引用；`engines.dsh` SemVer 声明兼容范围。
- 安装：`dsh plugin --profile <name> add <本地路径 | github:org/repo | npm 包名>`（内部转发 pnpm；`remove` 同步移除层）。
- git 安装拉源码不跑 build：作者要提供自包含 `prepare` 脚本；用户要在 profile 的 `pnpm-workspace.yaml` 加 `allowBuilds: { 包名: true }`（= 允许该包安装时在本机执行代码，仅对可信源授权，建议锁定 commit）。免授权替代：发 npm（预构建）或 `pnpm pack` tarball。

### 7. 能力三角色（仅当需要可替换提供方时）

Service Definition（定义 `ctx.<key>` 接口与 Request/Result 类型）→ Service Provider（实现）→ Consumer（通常是模型可见工具）。Provider 与 Consumer 互不依赖、都只依赖 Definition。简单工具插件不需要拆分。

---

## 社区经验补充（二手来源，供参考）

- [DeepSeek Harness (dsh) 插件开发教程（dev.to，中文）](https://dev.to/henry_lin_3ac6363747f45b4/deepseek-harness-dsh-cha-jian-kai-fa-jiao-cheng-4h6j) —— 十章开发者教程，与官方文档互为印证；含 in-repo 包结构、测试门禁（行为测试、HMR-safety 测试、快照测试）与排障表。
- [官方 Discussion #961：给 dsh 写你的第一个插件](https://github.com/deepseek-ai/deepseek-harness/discussions/961) —— 实战教程；补充了：免构建纯 ESM 插件骨架、`--patch` 下裸包名解析需 symlink node_modules 的坑、读当前会话的 `exec.agent.session.events` 事件信封表、运行时技能注册 `ctx.get('skills').register({ name, description, content, source:'runtime', provider })`。
- 仓库内待查的延伸文档（需要时可再抓）：`docs/subsystems/core.zh.md`（各服务的生成式 cordis-surface API 清单——内置服务方法的真源）、`docs/user/guide/providers.zh.md`（全部插件配置目录）、`packages/core/tools/README.zh.md`（扩展点的输入/顺序/返回值/失败行为）、`packages/jobs/jobs/README.zh.md`（后台任务运行时）。

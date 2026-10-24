# 问题摘要：dsh-mem 在 dsh 0.1.2-rc.1 上的加载失败与排查记录

> 记录时间：2026-10-24
> 结论：插件**已成功运行**，但根因是一个会让所有按默认配置安装的人都踩中的 bug；本文给出根因、证据、诊断手法与建议修法。

---

## 一、环境

| 项 | 值 |
|---|---|
| 宿主 | `dsh` 0.1.2-rc.1（`npx @deepseek-ai/dsh web`，Web GUI 127.0.0.1:3080） |
| profile | `web`（bundles: `@deepseek-ai/dsh-base` + `@deepseek-ai/dsh-web-app`，`patchReload: live`） |
| 插件 | dsh-mem 0.1.0，`lib/` 为预构建产物，依赖 `@deepseek-ai/dsh-* ^0.0.1-rc.1` |
| 安装方式 | 用户 patch 层 `~/.dsh/profiles/web/cordis.patch.yml` 的 `insert` 条目，指向 `lib/index.js` 绝对路径 |
| 平台 | Windows 10/11，Node 24.14.1 |

---

## 二、现象

1. 按 README 装完、重启 `dsh web` 后，终端**完全没有** `[dsh-mem] started:` 之类的日志；
2. GUI 里没有 `memory_search` 工具卡片，模型侧也调不到该工具；
3. `~/.dsh/storages/` 下没有任何 `dsh_mem` / `dsh-mem` 介质文件；
4. 反复修改配置、重启均无变化，且**没有任何报错**。

---

## 三、根因

### R1（阻断性）—— storage unit 名含连字符，被宿主直接拒绝

**事实**：

- 插件默认 `storage.unit = 'dsh-mem'`（`src/config.ts:155`），并且其 schema 校验为 `/^[a-z][a-z0-9_-]*$/`（**允许**连字符）；
- 宿主 `@deepseek-ai/dsh-storage@0.1.2-rc.1` 的校验是：
  ```js
  const UNIT_NAME_RE = /^[a-z][a-z0-9_]*$/;   // 只允许小写字母、数字、下划线
  ```
- `dsh-storage-json` 的 `kv.open()` → `validateDescriptor()` 会抛
  `StorageError('malformed-medium', "invalid unit name 'dsh-mem'")`。

**调用链**：

```
ctx.llm / storage 就绪 → index.ts apply()
  → MemoryRuntime.start(ctx, config)
      → ctx.storage.backend.get('json').kv.open({ name: 'dsh-mem', ... })
          → validateDescriptor → 抛 invalid unit name 'dsh-mem'
  → 被 apply() 里的 .catch(err => logger.error('failed to start: %o', err)) 吞掉
  → 后续 ctx.tools.register(createMemorySearchTool(...)) 根本没执行
```

因为**工具注册代码在 `kv.open()` 之后**（`src/runtime.ts:99-147`），一旦 open 失败，工具、注入 section、写入监听全都不会建立，表现为"插件似乎不存在"。

**修复（配置层，已生效）**：在 patch 里显式覆盖

```yaml
config:
  storage:
    backend: json
    unit: dsh_mem      # ← 关键
```

**建议源码修**（这份包本身的问题，需改代码）：

1. `src/config.ts` 默认值改为 `unit: 'dsh_mem'`；
2. schema 收紧为 `Schema.string().pattern(/^[a-z][a-z0-9_]*$/)`，与宿主规则一致；
3. 或者更稳妥：在 `memoryUnitDescriptor()` 里做 `unit.replace(/-/g, '_')` 归一化，同时对描述符做一次本地校验，让错误在**加载期**（schema/启动期）而不是运行期抛出；
4. README 的"配置速览"表补一行说明：unit 名不允许连字符。

---

### R2 —— 启动失败完全不可见（诊断成本的真正来源）

- 本宿主 console **不输出 info 级日志**。实测：headless 一次性运行成功时，整段输出只有模型回答本身，没有任何插件日志；
- 叠加 R1 的 `.catch(...)`，故障是**静默**的。所以"终端没有 `[dsh-mem] started`"这件事，**既不能证明加载成功，也不能证明失败**。

**结论（重要）**：判断插件是否加载，不能看日志，要看**可观测副作用**：`memory_search` 工具是否存在，或 `~/.dsh/storages/dsh_mem.json` 是否出现。

**建议**：

- `apply()` 的 catch 里除 `logger.error` 外，再落一条粗粒度可见信号（如 `console.error('[dsh-mem] failed to start:', err)`），因为 logger 在本宿主不可见；
- README 排障表增加一行："看不到任何 `[dsh-mem]` 日志 → 先确认宿主是否输出 info 日志；判定加载请查工具是否存在"。

---

### R3 —— README「方式 A」用 `src/index.ts` 在本宿主不可行

- 宿主 loader（`cordis-plugin-loader`）对 `.ts` 的处理只是把相对 specifier 的扩展名改写为 `.js`，**没有 TS 转译**；
- 而 `src/index.ts` 内部导入写的是 `./config.js`、`./runtime.js` —— `src/` 下并不存在这些 `.js` 文件，只有 `lib/` 有；
- 实测直接 `import` `lib/index.js`（file:// URL 形式）可正常加载，因此**文档应改为指向 `lib/index.js`**（或要求先 `npm run build`）。

---

### R4（潜在，本次未触发）—— 依赖版本错位

插件依赖 `@deepseek-ai/dsh-* ^0.0.1-rc.1`（实际装到 0.0.1-rc.1~rc.5），宿主为 0.1.2-rc.1。本次逐项核验的跨版本契约**全部兼容**：

| 契约 | 结论 |
|---|---|
| `ctx.storage.backend.get(name): StorageBackend` + `backend.kv.open(descriptor)` | ✅ 形状一致 |
| `KvUnitDescriptor { name, version, tables, hasGlobal, layout? }` | ✅ 插件未传 `layout`，默认 `single`，合法 |
| `ctx.tools.register(definition)` | ✅ 仅按 `definition.name` 建索引，无品牌校验 |
| `ctx.llm.stream(options)` 的 `GenerateOptions` / `StreamChunk` | ✅ 与 0.0.1-rc.5 逐字段一致 |
| 事件 `agent/created` / `agent/disposed` / `agent/inbox/claimed` / `system-prompt/assemble` | ✅ 均存在，签名一致 |
| 持久事件 `user/message`（data 即消息）、`assistant/message`（data 为 `{ message }`）、`turn/end` | ✅ 与插件解析方式一致 |

**建议**：仍应把依赖升到 `^0.1.2-rc.1` 后重新 `typecheck` + `build`，防止宿主继续演进后被动失效。

---

## 四、本次验证结果（证据）

修复 unit 名后，以下全部实测通过：

| 项 | 验证方法 | 结果 |
|---|---|---|
| 模块在宿主进程内可 import | 探针插件内动态 `import('file:///…/lib/index.js')` | ✅ 导出 `Config/MemoryRuntime/apply/inject/name` |
| Config schema 接受配置 | 探针内调 `Config(cfg)` | ✅ 默认值正确补齐 |
| 5 个 inject 可解析 | 与插件同 `inject` 的探针 | ✅ 加载成功（不卡 PENDING） |
| patch 插入 + Windows 绝对路径 | A/B 双探针 | ✅ 宿主会 `isAbsolute → pathToFileURL` |
| 存储 unit 可打开 | 改成 `dsh_mem` 后 | ✅ |
| `memory_search` 注册并可执行 | 实际调用一次 | ✅ 空库返回 `No relevant memories found.` |
| 判分/抽取的 LLM 通道 | 直接调 `CordisLlm.complete()` | ✅ 返回 `PONG` |
| 写入路径落盘 | 一轮对话后查介质文件 | ✅ `~/.dsh/storages/dsh_mem.json`（约 118 KB） |
| 抽取质量 | 反序列化介质 | ✅ 8 条中文 episodic，`imp` 4–6、`S=imp`、`tier=warm`、`dim=1024`、`semantic` 表 0 条（睡眠周期尚未触发） |

**当前生效配置**（`~/.dsh/profiles/web/cordis.patch.yml`）：

```yaml
- insert:
    - id: dsh-mem
      name: 'C:/Users/Neuroil/Code/dsh_mem/lib/index.js'
      config:
        storage:
          backend: json
          unit: dsh_mem
        embedding:
          mode: deterministic
        reranker:
          mode: deterministic
        llm:
          provider: deepseek-official
          model: deepseek-v4-flash
```

> 说明：`llm.provider` 必须是宿主已注册的路由名。本宿主注册的是 **`deepseek-official`**（`@deepseek-ai/dsh-llm-deepseek` 的 `PROVIDER` 常量），模型目录为 `deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v4-flash-vision-exp`；插件默认的 `deepseek` / `deepseek-chat` 均不可用。
> `embedding` / `reranker` 目前是 `deterministic` 离线替身（词袋哈希），只验证管线连通，**没有语义能力**，需替换为真实端点。

---

## 五、可复用的诊断手法（本次踩出来的）

1. **先验组合，再验加载**：`dsh --profile web --dump-config` 会打印合成后的插件树并标注来源层（`# == <patch 文件路径>`）。若目标条目在树里，问题就不在 patch 解析。
2. **用 headless 一次性进程复现启动**：`dsh --profile headless --patch <file> '...'` 会打印真实启动期错误，且不占用 3080、不影响正在运行的 GUI。它需要凭据：可从 `~/.dsh/.credentials.yaml` 的 `refs` 取出 `DEEPSEEK_API_KEY` 注入 `$env:DEEPSEEK_API_KEY`。
3. **探针插件写文件，绕开被压掉的 console**：宿主里 `ctx.logger` 不可见，但插件内 `appendFileSync` 到临时文件可见。探针要声明与被测插件**相同的 `inject`**，否则会得到误导性的 `cannot get property "X" without inject`（这是 cordis 的合法约束，不是 bug）。
4. **判定"插件是否已加载"最省事的办法**：让探针尝试再注册一个同名工具，抛
   `tool "memory_search" is already registered` 即说明真插件已注册 → 已加载。
5. **判定"写入链是否通"**：查 `~/.dsh/storages/<unit>.json`；`single` 布局的 unit **只在首次 `putRecord` 时才创建文件**，所以启动后文件不存在属于正常，不能据此判断失败。
6. **热重载在本宿主是生效的**：修改 profile patch 文件后条目会被重组合（本次最终就是靠它上线，无需重启）。早期"热重载失效"的判断是错的——当时依据是缺失的日志，而日志本就不显示。

---

## 六、遗留与后续

- [ ] 源码修 R1（unit 默认值 / schema 正则 / 描述符归一化），这是唯一阻断性问题；
- [ ] 源码或文档修 R3（README 指向 `lib/index.js`，或明确要求先 build）；
- [ ] 依赖升到 `^0.1.2-rc.1` 并重新 `npm run typecheck && npm test && npm run build`；
- [ ] 真实嵌入/重排端点（SiliconFlow `bge-m3` + `bge-reranker-v2-m3`，或任意 OpenAI 兼容 `/embeddings` + Cohere/Jina 式 `/rerank`）；切换后需把 `embedding.dimensions` 与实际模型对齐；
- [ ] 睡眠周期（P1）尚无观测：需要 ≥ `sleep.minNewMemories`(10) 条新写入且聚类池不少于 `clusterMinSize`(3)；当前 deterministic 嵌入下同义句余弦仅 ~0.27–0.35，需先调低 `sleep.clusterCos` 或换真实嵌入后再观察 `semantic` 表；
- [ ] 已知副作用清理：排查期间一次 headless 测试让模型自动生成了工作区 `AGENTS.md`（dsh 的持久偏好机制，非本插件行为），已删除。

---

## 七、回滚方式

把 `~/.dsh/profiles/web/cordis.patch.yml` 恢复为：

```yaml
[]
```

宿主会卸载该条目；记忆介质 `~/.dsh/storages/dsh_mem.json` 不会被删除，可保留或手动清理。

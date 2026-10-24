# dsh-mem — 语义驱动记忆插件（DeepSeek Harness）

> 核心立场：**遗忘与巩固由语义而非时间决定。时间戳只作为元数据保存，永不作为衰减/删除的驱动。**
> 设计文档：`docs/design/语义驱动记忆系统设计.md`（定稿 v1.0）。本包实现其 **P0 + P1 + P2** 里程碑。

一句话：一个 Cordis 插件，监听会话日志，把值得记的内容（经 M5 rubric 判分）连同嵌入向量写入持久记忆库；每次模型请求前按语义检索（余弦召回 + CSLS + 精排 + MMR）注入最相关的记忆；矛盾的新记忆把旧记忆**或语义节点**软失效（不删除，且只对时间上重叠的矛盾生效）；每次记忆进入最终上下文都按线索质量巩固其分数 S、被竞争压制的按小权重衰减、睡眠周期对全库施加慢兜底衰减——S 决定 hot/warm/cold 分层而非删除；离线"睡眠"周期把冗余的情景簇蒸馏成语义 gist 节点（聚类 → gist → QA 闸门 → M4 路由 → M6 评估 → 迁移），原始记忆压缩归档；`memory_search` 支持 `at` 时间旅行参数，可检索任意历史时点的记忆状态。时间从不参与排序或删除决策（唯一例外是慢兜底衰减，且只影响分层、永不删除）。

## 产出结果

### 1. 自动注入的记忆 section（每步系统提示词，库非空且命中时）

```
Relevant long-term memories (semantic retrieval; recency was not a factor):
1. 用户偏好用 PostgreSQL 作为生产数据库  [id=9f2ac1… score=0.812 tier=warm]
2. 部署延迟通常由配置复杂性导致  [id=41b7de… score=0.774 tier=warm]
```

评分排序是纯语义的：十年前相关的原则性记忆可以排在上一分钟的闲聊之前。同一轮内展开查询不变时复用检索结果，不重复调用嵌入/重排端点。

### 2. `memory_search` 工具（模型主动召回）

| 参数 | 类型 | 说明 |
|---|---|---|
| `query` | string，必填 | 自然语言检索请求 |
| `limit` | integer，可选 | 返回条数上限（默认 `tool.maxResults`=5；< 1 返回规范 error 值） |
| `at` | string，可选 | 时间旅行时点（ISO 8601 或 epoch ms，§2.5 完整版）：检索该时点的记忆状态——被取代的记忆在该时点之前重新可见；非法值返回规范 error 值而非抛异常 |

返回单个规范 JSON 值（模型可见文本由 `render` 从该值投影）。`layer` 区分原始记忆（`episodic`）与睡眠周期蒸馏出的 gist（`semantic`）；传了合法 `at` 时输出带 `at` 回显：

```json
{
  "query": "which database should we use?",
  "expandedQuery": "which database…\n---\n最近对话轮摘要",
  "memories": [
    { "id": "9f2ac1…", "text": "用户偏好用 PostgreSQL…", "score": 0.812, "cue": 0.65, "tier": "warm", "layer": "episodic", "createdAt": 1761273600000 }
  ],
  "suppressedCount": 2
}
```

### 3. 持久记忆库（ctx.storage，不自建数据库）

- KV unit（默认名 `dsh_mem`，`config.storage.unit` 可改；unit 名只允许小写字母/数字/下划线，**不允许连字符**——宿主规则 `/^[a-z][a-z0-9_]*$/`），两张表：`episodic`（MemoryRecord）与 `semantic`（SemanticNode，睡眠周期填充）。
- 记录含 M5 判分（imp/valence/arousal/reasons）、邻域统计（r_m/red）、巩固分数 S、分层 tier（hot/warm/cold）、使用计数（used_count/suppressed_count/last_used_at/last_cue_quality/last_decay_at）——计数自 P2 起被 S 动态消费；语义节点带同构的可选 counters。
- **软失效**：被取代的记录或语义节点只设 `invalid_at` / `superseded_by` 并移出检索索引，永不删除。矛盾判定带**时间重叠闸**（§2.5 完整版）：只对同一时间框架的矛盾软失效，"过去用 MySQL"与"现在用 PostgreSQL"这类不同时期的矛盾两条都保留。
- **压缩归档**（P1）：被语义节点吸收的记录设 `flags.compressed_into` / `qa_passed` 与 `time_meta.compressed_at`（归档时点，时间旅行消费）并退出检索池；QA/epi 闸门拒收的簇标 `flags.incompressible`，全簇已标时需新成员加入才重试。
- 落盘位置由所选 storage 后端拥有（json / sqlite 各自决定介质），unit 格式版本见 `config.storage.version`。

### 3a. 巩固分数 S 的完整动态（P2，设计 §2.1）

```
初始：    S ← imp                                  （写入时）
成功使用：S ← S + α·(1 + cue)·g(arousal)           α = score.useGain = 0.5
                                                  g = 0.5 + 0.5·arousal（§3.5 乘法门；语义节点无 arousal，门恒 1）
被抑制：  S ← S − ε                                ε = score.suppressPenalty = 0.05（每查询至多一次）
兜底衰减：S ← S·e^(−Δt/τ)                          τ = score.decayTauDays = 180 天（半衰期 ≈ 125 天）
```

- **成功使用 = 进入最终上下文**（两条检索通道的 top 输出，scoring.ts 记账）。设计所列"被采纳"信号（ctx.messageFeedback）不在当前依赖集内（message-feedback 包类型未发布到 npm，P1 对 ctx.schedule 的同款取舍）：按"未被纠正"默认成立即时入账，接入后可改为确认制。
- **兜底衰减是工程保险**：只影响分层、永不删除。以 `counters.last_decay_at` 为锚**惰性求值**（事件路径先补衰减再加减增量，睡眠周期第 7 步全库批处理），与逐日衰减数学等价；只作用于仍在检索宇宙的记录/节点。
- **分层**：S ≥ 8 hot、2–8 warm、< 2 cold；cold 需更强语义线索（CSLS cue 再 +0.1）。tier 随 S 动态跃迁；**事件驱动的跃迁触发 M5 重判**（新判分按 `score.rejudgeEma` EMA 混入），衰减引起的跃迁不重判（衰减不构成重要性的新证据，防边界穿越烧调用）。
- 语义节点同构：使用/抑制事件、兜底衰减都作用（P1 只有挂入增益 attachGain）。

### 3b. 时间旅行（P2，设计 §2.5 完整版）

- 任意时点 t 的记忆状态 = `valid_at ≤ t < invalid_at` 且当时未压缩（`compressed_at` 未设或晚于 t）的记录 + 当时有效的语义节点（`MemoryStore.retrievableAt(t)`）。
- `memory_search` 的 `at` 参数走该视图；被取代的记忆在其失效前的时点重新可检索。近似边界：tier / r_m / 节点 gist 取当前落盘快照（历史 S 与邻域不重建）；P1 遗留的无 `compressed_at` 归档记录保守视为历来已压缩（无法定期则不复活）。

### 4. 睡眠巩固周期（P1，设计 §2.3/§5.3）

turn/end 信号门控触发（自上次周期新写入 ≥ `sleep.minNewMemories`，默认 10；fire-and-forget 不阻塞会话，运行中不重入）：

```
1. 聚类     pairwise 余弦 ≥ clusterCos 的有效未压缩记忆 → 连通分量；
           簇大小 ≥ clusterMinSize 且簇内平均 pairwise 余弦 ≥ clusterCos（链式传导被平均值拦截）
2. gist    LLM 对簇摘要抽象（"用户多次提到部署延迟与配置复杂性相关"）
3. QA 闸门  从簇生成 QA 对、仅以 gist 作答，平均 token 级 F1 ≥ qaF1Threshold(0.8) 才放行；
           拒收 → 保留原簇并标记"不可压缩"（防错误 gist 固化传播）
4. M4 路由  con = gist 对语义网络峰值余弦的 z 分数（基线 = 最近 conWindow 条记忆的分布）：
           con > +0.5σ 且 LLM 兼容复核通过 → 挂入已有节点（member_refs 追加、S += attachGain）；
           con < −1σ 且簇均 imp ≥ 7 → 建新节点；其余留情景层等下轮；空网络无条件建首节点（引导）
5. M6 评估  gist 的完整 epi = σ(z(实体密度)+z(时间密度)−z(探针相关度))，低于 epiThreshold 才迁移；
           epi 下降量 = "细节遗忘量"的量化
6. 迁移     成员 compressed_into 指向节点、记录 compressed_at 归档时点，退出检索池
7. 刷新     全库 r_m/red 与节点 r_m 重算（§3.2/§3.3"夜间刷新"；无簇时也执行）；
           兼施 S 兜底衰减（§2.1，P2）：留层记录与有效节点 S·e^(−Δt/τ)，衰减跃迁不触发 M5 重判
```

### 5. 仓库工件

| 路径 | 内容 |
|---|---|
| `src/` | 插件源码（模块地图见[开发](#开发)） |
| `lib/` | `npm run build` 产物（ESM + .d.ts） |
| `tests/` | 180 个离线单测（vitest，无需网络与 API key） |
| `scripts/eval-ndcg.mjs` | §7.1 检索四档消融脚手架 |
| `scripts/eval-sleep.mjs` | P1 睡眠周期脚手架（QA 保留率、压缩后召回存续） |
| `scripts/eval-semtime.mjs` | P2 语义 vs 时间消融脚手架（§7.4 关键消融 + 时间旅行） |
| `data/` | 示例语料与查询集（格式见[评测](#评测ndcg-四档消融)） |
| `docs/` | 设计文档（定稿）、`docs/architecture/插件工作原理全景图.md`（内部细节唯一引用点）、dsh 官方文档镜像 `docs/ref/`、立项认知机制对话 `1.md` |
| `cordis.patch.yml` | 组合包 patch（`dsh plugin add` 时使用） |

### 6. 当前验证状态

| 门禁 | 结果 |
|---|---|
| `npm run typecheck`（strict + noUncheckedIndexedAccess） | 0 错误 |
| `npm test` | 180/180 通过 |
| `npm run build` → `lib/` | 通过；`node` 导入 `lib/index.js` 与 Config 默认值校验通过 |
| 非法配置响亮失败 | Schema 加载时抛 `ValidationError`（如 `score.useGain: "not-a-number"`） |
| eval-ndcg 冒烟（deterministic 替身） | 四档管线全部跑通（数值无语义意义） |
| eval-sleep 冒烟（echo LLM + deterministic 替身） | 聚类→gist→QA→迁移→统计刷新→S 衰减全链路跑通（数值无语义意义） |
| eval-semtime 冒烟（deterministic 替身） | semantic / time / ga-product 三臂 + 时间旅行复活计数跑通（数值无语义意义） |

**尚未验证**（需外部资源，见验收目标）：真实嵌入/重排端点下的 nDCG 与语义 vs 时间消融数字、M5 判分与人工标注的 Spearman ρ > 0.7、真实 LLM 下 QA 闸门保留率 ≥ 0.8 与长程问答提升、messageFeedback"被采纳"确认制接线（message-feedback 包类型进入依赖集后），以及在真实 dsh 宿主中的端到端会话运行（本包的宿主对接均已对照 npm 上 `@deepseek-ai/*` 真实类型定义编写并通过类型检查，但未在活的 harness 进程里跑过）。

## 使用方法

### 前置条件

- Node ≥ 20；dsh 运行环境（`npx @deepseek-ai/dsh web`）。
- profile 里有 **storage 后端**（dsh-base 自带持久化层；纯自定义组合需自装 `@deepseek-ai/dsh-storage-json` 或 `dsh-storage-sqlite`）与 **llm 适配器**（如 `dsh-llm-deepseek` + 凭据）。
- 嵌入/重排端点的 API key（默认指向 SiliconFlow 的 bge-m3 / bge-reranker-v2-m3）。没有 key 可先用 `deterministic` 模式冒烟（离线替身，验证管线不验证语义）。

### 方式 A：本地源码加载（开发）

创建 `overlay.yml`（**插件路径必须是绝对路径**）：

```yaml
- insert:
    - id: dsh-mem
      name: 'C:/Users/you/Code/dsh_mem/src/index.ts'
      config:
        embedding:
          mode: http            # 无 key 冒烟改为 deterministic
          apiKey: !!js process.env.MEM_EMBEDDING_API_KEY ?? ''
        reranker:
          apiKey: !!js process.env.MEM_RERANKER_API_KEY ?? ''
        llm:
          provider: deepseek    # 必须与已注册的 llm 适配器路由一致
          model: deepseek-chat
```

```sh
npx @deepseek-ai/dsh web --patch ./overlay.yml   # --dump-config 可检查最终组合
```

配置修改经 HMR 热重载生效（改 `config` 即卸载重载本插件，注册自动清理重建）。

### 方式 B：安装为组合包

```sh
dsh plugin --profile web add C:/Users/you/Code/dsh_mem
```

包身份来自 `package.json` 的 `dsh.bundle.patch` → `cordis.patch.yml`；`remove` 同步移除。git 安装的授权坑（pnpm ≥10 `allowBuilds`）见 docs/ref/README。

### 必要配置最小集

其余全部有 schema 默认值，通常只需：

1. `embedding.apiKey` + `reranker.apiKey`（或两者 `mode: deterministic`）；
2. `llm.provider` / `llm.model`：M5 判分与矛盾判定的调用路由，**必须是 ctx.llm 上已注册适配器的 provider 名**，模型随之有可用凭据。

### 确认插件在跑

- 启动日志：`[dsh-mem] started: 0 valid memories in store`；
- 每轮 `turn/end` 后（有写入时）：`turn N: wrote X, superseded Y, dropped Z`；
- 睡眠周期触发后：`sleep cycle: C clusters (M migrated, G gated), refreshed R records / N nodes, decayed R'/N'`；
- 聊天 UI 出现 `memory_search` 工具卡片；库非空且命中时，`memory_search` 结果与自动记忆 section 出现在上下文（语义节点带 `gist` 标记）；
- storage 后端介质上出现 `dsh_mem` unit。

## 评测：nDCG 四档消融（设计 §7.1）

数据格式（JSONL）：

```
corpus.jsonl   每行 {"id":"m01","text":"..."}          # id 可省略，自动编号
queries.jsonl  每行 {"query":"...","relevant":["m01","m09"]}
               # 或分级相关度 {"query":"...","relevant":{"m03":2,"m09":1}}
```

```sh
# deterministic 冒烟（离线，验证管线）
npm run eval:ndcg -- --corpus data/corpus.jsonl --queries data/queries.jsonl

# 真实端点（有语义意义的数字）
npm run eval:ndcg -- --corpus data/corpus.jsonl --queries data/queries.jsonl \
  --embedding-mode http --embedding-api-key $KEY \
  --reranker-mode http --reranker-api-key $KEY
```

deterministic 冒烟输出（数值仅证明四档可执行，**没有**语义意义）：

```
corpus=12 queries=8 embedding=deterministic reranker=deterministic
rung	nDCG@10	recall@50(L0 pool)
base	0.6124	1.0000
csls	0.6691	1.0000
rerank	0.6124	1.0000
full	0.5389	1.0000
```

验收目标：真实数据下 CSLS 档随库增大与 base 档拉开差距（hubness 恶化）；M5 判分另需人工标注集（Spearman ρ > 0.7），判分 prompt 已带 Generative Agents 锚定示例。

## 评测：睡眠周期（P1，QA 保留率与压缩存续）

```sh
# deterministic 冒烟（离线 echo LLM 替身，验证管线连通）
npm run build
npm run eval:sleep -- --corpus data/corpus.jsonl --queries data/queries.jsonl \
  --cluster-min-size 2 --cluster-cos 0.25
```

deterministic 嵌入是词袋哈希，同义句对余弦仅 ~0.27–0.35，冒烟需降低聚类阈值（如上）。输出（数值仅证明管线可执行，**没有**语义意义）：

```
corpus=12 queries=8 embedding=deterministic clusterCos=0.25 clusterMinSize=2
clusters=1 migrated=1 gated=0 compressed=2 nodes=1
  new_node   members=2 qaF1=0.941 gistEpi=0.745 memberEpi=0.533 node=e09d… :: GIST: Deployment delays…
QA-F1 of migrated clusters (retention, echo stand-in): 0.9412
mean epi drop of migrated clusters (detail forgetting):  -0.2112
nDCG@10 before=0.5389 after(remapped)=0.5474
```

- **QA-F1（保留率）**：迁移簇的平均 QA 闸门 F1——真实 LLM 下应 ≥ 0.8（闸门本身即按该阈值放行）；
- **epi 下降量**：成员均 epi − gist epi，即"细节遗忘量"（echo 替身的 gist 是成员原文回显，故为负；真实 gist 应为正）;
- **nDCG after(remapped)**：压缩后按"相关成员命中经其所属节点计数"重映射相关度再测——被压缩信息若仍能经语义节点召回，该值不显著低于 before（长程问答存续的代理指标）。

可调项：`--cluster-cos`、`--cluster-min-size`、`--qa-f1-threshold`、`--epi-threshold`；真实端点参数与 eval-ndcg 相同（`--embedding-mode http --embedding-api-key …`）。**真实 LLM 下的睡眠周期**（gist 抽取 / QA 闸门 / M4 兼容复核）需在宿主中以 `llm.provider` 指向已注册适配器运行，本脚本离线替身不覆盖该路径的语义质量。

## 评测：语义 vs 时间消融（P2，设计 §7.4 关键消融）

直接检验本项目核心假设——语义驱动的遗忘/检索优于 Generative-Agents 式时间衰减。数据格式：

```
corpus-times.jsonl   每行 {"id","text","imp"?,"time"?,"validAt"?,"invalidAt"?}
                     time/validAt/invalidAt = ISO 8601 或 epoch ms；
                     缺省 time 按行序在过去 30 天线性铺开（仅让时间 baseline 可执行）
queries-times.jsonl  每行 {"query","relevant":[id...],"at"?}
                     at = 时间旅行时点：semantic 臂检索该时点重建的记忆状态
```

```sh
npm run build
npm run eval:semtime -- --corpus data/corpus-times.jsonl --queries data/queries-times.jsonl
```

三臂：`semantic`（本插件全管线，含软失效与有效性模型）、`time`（GA recency 单因子 0.995^h）、`ga-product`（GA 完整乘积 imp × 0.995^h × cos）。两个 baseline 对**原始语料**排名——Generative Agents 没有取代/有效性模型，该差距本身是被测对象的一部分。deterministic 冒烟输出（数值仅证明三臂可执行，**没有**语义意义）：

```
corpus=8 queries=5 embedding=deterministic evalNow=2026-10-20T00:00:00.000Z recency=0.995^h
semantic  	nDCG@10=0.7328
time      	nDCG@10=0.4125
ga-product	nDCG@10=0.5752
time-anchored queries: 1 (resurfaced relevant ids via time travel: 1)
```

- `resurfaced`：时间旅行查询中，"当前视图已不可见（被软失效）但该时点有效"的相关记忆数——支柱五完整版的直接演示（示例数据里 2026-06 的 MySQL 记忆在 6 月时点重新可检索，而 8 月切换到 PostgreSQL 之后它被取代）；
- 验收目标：真实端点与真实标注下 semantic 显著优于两个时间 baseline（尤其"十年前的原则 > 一分钟前的闲聊"类查询）；LoCoMo 式端到端回放需活宿主 + 真实 LLM，见上文"尚未验证"。

## 开发

```sh
npm install
npm run typecheck                          # tsc --noEmit
npm test                                   # vitest run（全部 180 个）
npx vitest run tests/sleep-cycle.test.ts   # 单文件
npm run build                              # 产出 lib/
```

模块地图（全部相对导入带 `.js` 扩展，中文注释标注设计章节号）：

```
src/
  index.ts        插件入口（具名导出 name/inject/Config/apply；inject: storage,llm,tools,systemPrompt,sessions）
  config.ts       全超参 Config schema（Schemastery，默认值在 schema，无硬编码可调参数）
  types.ts        数据模型与共享接口（MemoryRecord/SemanticNode/ScoreCounters/RetrievableItem/Provider 接口）
  memory-store.ts ctx.storage KV 仓库 + 记录/节点软失效 + 检索宇宙与时间旅行视图 + 邻域统计（r_m/red）
  metrics.ts      cosine / CSLS / MMR / z 标准化 / sigmoid / QA-F1 纯函数
  vector-store.ts 精确 kNN 索引（P0 规模；接口留出 ANN 余地）
  retrieval.ts    四级检索管线 + 消融档（base/csls/rerank/full；L0 池 = 未压缩记录 + 语义节点；at 时间旅行选项）
  embedding.ts    嵌入/重排客户端（OpenAI 兼容 /embeddings、/rerank；deterministic 离线替身）
  llm.ts          内部 LLM 调用（ctx.llm 流服务 + BlockAssembler；适配器失败按终态 error/aborted 块识别）
  judges.ts       候选抽取 / M5 判分 / 矛盾判定（含 temporal 重叠）/ epi 词法 / gist 抽取 / QA 生成与作答 / M4 兼容复核（JSON 健壮解析 + 重试一次）
  scoring.ts      支柱一 S 完整动态（§2.1）：使用/抑制事件、arousal 乘法门、惰性兜底衰减、tier 跃迁 + M5 EMA 重判（ScoreKeeper）
  write-path.ts   写入管线（§5.1 七步 + §2.5 完整版：时间重叠闸 + 语义节点软失效；单候选失败不中断整轮）
  sleep-cycle.ts  睡眠巩固周期（§2.3 七步 + 第 7 步 S 兜底衰减；迁移带 compressed_at 归档时点）
  runtime.ts      编排：会话观察、agent 定域自动注入、ScoreKeeper 记账接线、睡眠周期触发
  tool.ts         memory_search 工具（含 at 时间旅行参数）+ section 文本渲染
```

实现选择说明：自动注入落在 `system-prompt/assemble` waterfall（agent 定域注册）而非 `ctx.systemPrompt.section()`——后者 text provider 是同步签名，承载不了异步检索链路；前者为官方文档同章节列出的扩展点。检索查询取自 `agent/inbox/claimed` 的待处理输入 ⊕ 最近对话尾部。

## 配置速览（默认值见 `src/config.ts`）

| 键 | 说明 |
|---|---|
| `storage.backend` / `unit` / `version` | KV 后端名（json/sqlite）、unit 名（**不含连字符**，规则同宿主 `/^[a-z][a-z0-9_]*$/`，连字符会被归一为下划线）、格式版本 |
| `embedding.*` | mode(http/deterministic)、endpoint、model、apiKey、dimensions(1024)、batchSize |
| `reranker.*` | 同上（无 dimensions） |
| `llm.provider` / `model` / `temperature`(0) / `maxTokens` | 内部判分调用（经 ctx.llm） |
| `write.messageWindow`(10) / `maxCandidates`(8) / `minImportance`(3) / `contradictionTopK`(5) / `contradictionMinCos`(0.8) | 写入路径 |
| `retrieval.l0TopK`(100) / `cslsTopK`(20) / `rerankTopK`(8) / `mmrTopK`(5) / `mmrLambda`(0.6) / `cslsK`(10) / `contextTurns`(3) / `coldCueBoost`(0.1) / `cueThreshold`(0.3) | 检索链路 |
| `tier.hotThreshold`(8) / `coldThreshold`(2) | 分层（S ≥ 8 hot，S < 2 cold） |
| `score.useGain`(0.5) / `suppressPenalty`(0.05) / `decayTauDays`(180) / `rejudgeEma`(0.5) | 支柱一 S 动态（§2.1）：使用增益 α / 抑制惩罚 ε / 兜底衰减 τ（天）/ 跃迁重判 EMA 权重 |
| `sleep.enabled` / `minNewMemories`(10) | 睡眠周期开关与触发阈值（新写入计数） |
| `sleep.clusterCos`(0.75) / `clusterMinSize`(3) | 聚类边阈值（兼簇内平均 pairwise 下限）与最小簇大小 |
| `sleep.qaPairCount`(3) / `qaF1Threshold`(0.8) | QA 闸门：每簇问答对数与平均 F1 阈值 |
| `sleep.epiThreshold`(0.5) | gist 的 M6 epi 迁移阈值（低于才迁语义层） |
| `sleep.conWindow`(1000) / `conAttachSigma`(0.5) / `conNewNodeSigma`(−1) / `conNewNodeMinImp`(7) / `attachGain`(0.5) | M4 路由：z 基线窗口与三档路由阈值、挂入增益 |
| `section.enabled` / `maxMemories`(5) / `header` | 自动注入通道 |
| `tool.enabled` / `maxResults`(5) | memory_search 工具 |

## 排障

| 现象 | 原因与处理 |
|---|---|
| 启动报 `storage backend '…' provides no kv facet` | profile 没有 storage 后端或 `storage.backend` 名不对：装 `dsh-storage-json`/`dsh-storage-sqlite` 并核对注册名 |
| 插件停在 PENDING 不加载 | `inject` 的服务（storage/llm/tools/systemPrompt/sessions）缺提供方；查 `--dump-config` 与启动日志 |
| 打开 unit 报 `version-mismatch` | `config.storage.version` 与介质上既有版本不一致；P1 仅向记录新增可选字段（旧记录缺字段按 undefined 容忍），保持 version=1 即可 |
| 判分调用失败（NO_ADAPTER / 空回复 / error finish） | `llm.provider` 未注册适配器或凭据失效；llm.ts 会抛出携带失败码的错误，见 dsh 日志 |
| 嵌入/重排 401、超时 | 核对 apiKey、endpoint、网络；临时可切 `deterministic` 模式隔离问题 |
| 睡眠周期从不触发 | `sleep.enabled` 为 false、新写入未达 `sleep.minNewMemories`、或聚类池小于 `clusterMinSize`；看启动后是否有 `sleep cycle:` 日志 |
| 睡眠周期全部 gated（qa_failed） | QA 闸门在拒绝：gist 质量不足（换更强判分模型）或 `qaF1Threshold` 过严；被拒簇已标 incompressible，需新成员加入才重试 |
| 压缩后检索不到原细节 | 设计行为：成员内容由节点 gist 承载（§2.3"记住要点、忘掉细节"）；回溯经 `flags.compressed_into` 索引原记录，或用 `memory_search` 的 `at` 参数在归档前的时点检索 |
| 记忆还在但排不上名 | tier 可能已沉到 cold（S 动态：长期未使用 + 兜底衰减）：cold 项需 CSLS cue ≥ `cueThreshold + coldCueBoost` 才保留；调 `tier.coldThreshold` / `score.decayTauDays`，或等使用事件把它推回 warm |
| `memory_search` 返回 `invalid 'at'` | `at` 需为 ISO 8601（以 4 位年份开头）或正整数 epoch ms；领域级错误写进规范值，不影响工具通道 |

## 明确不在 P2（后续里程碑）

messageFeedback"被采纳"确认制（S 增益目前按"未被纠正"默认即时入账，§5.0 所列 seam 的类型不在依赖集内）、LoCoMo 式活宿主端到端回放；语义图 PPR 联想检索、spacing 加权、sur、valence 情绪一致性检索（P3，设计 §6/§10）。

已知接受的取舍（P2）：跨写链并发 `updateRecord` 的丢失更新窗口延续 P1 声明（最坏一次 flag/计数/S 增量，可被下一周期自愈）；写入路径与睡眠周期可并发写同一 unit 的不同记录（KvUnit 单调用原子、整链不串行是宿主契约给调用方的自由度）；兜底衰减与全库统计刷新同门控（`sleep.enabled=false` 时离线维护整体停摆，事件路径的惰性衰减追补仍然生效）。

# dsh-mem — 语义驱动记忆插件（DeepSeek Harness）

> 核心立场：**遗忘与巩固由语义而非时间决定。时间戳只作为元数据保存，永不作为衰减/删除的驱动。**
> 设计文档：`docs/design/语义驱动记忆系统设计.md`（定稿 v1.0）。本包实现其 **P0 + P1 + P2** 里程碑，另交付 **M1/M2/M3**（接入宿主侧 sqlite 存储后端——后端本体是独立包 `dsh-storage-sqlite`，本包不含任何后端、按工作区隔离记忆、会话历史入口可配置化）。

一句话：一个 Cordis 插件，监听会话日志，把值得记的内容（经 M5 rubric 判分）连同嵌入向量写入持久记忆库——**每个工作区（会话创建目录）一个独立记忆库（M2）**；每次模型请求前按语义检索（余弦召回 + CSLS + 精排 + MMR）注入最相关的记忆；矛盾的新记忆把旧记忆**或语义节点**软失效（不删除，且只对时间上重叠的矛盾生效）；每次记忆进入最终上下文都按线索质量巩固其分数 S、被竞争压制的按小权重衰减、睡眠周期对全库施加慢兜底衰减——S 决定 hot/warm/cold 分层而非删除；离线"睡眠"周期把冗余的情景簇蒸馏成语义 gist 节点（聚类 → gist → QA 闸门 → M4 路由 → M6 评估 → 迁移），原始记忆压缩归档；`memory_search` 支持 `at` 时间旅行参数，可检索任意历史时点的记忆状态。时间从不参与排序或删除决策（唯一例外是慢兜底衰减，且只影响分层、永不删除）。

## 使用方法

### 前置条件（依赖清单）

逐项核对，**全部满足**再进入后面的步骤；每项标注获取方式与缺失后果，整表可直接交给 Agent 执行核对：

| # | 依赖 | 具体要求 | 获取方式 | 缺失后果 |
|---|---|---|---|---|
| 1 | Node.js | ≥ 20 | nodejs.org 或系统包管理器 | 宿主与插件均无法运行 |
| 2 | dsh 宿主 | `npx @deepseek-ai/dsh web` 可启动并进入聊天 UI | npm（DeepSeek 官方包） | 无运行环境 |
| 3 | 存储后端 | 二选一：json（dsh-base 自带，零安装）；或 **sqlite（推荐）** `@deepseek-ai/dsh-storage-sqlite`，版本与宿主**逐位一致**（宿主 0.2.1-rc.2 就装 0.2.1-rc.2；装 0.2.1-rc.3 会被 peer 闸拒） | `dsh plugin --profile web add @deepseek-ai/dsh-storage-sqlite@<宿主版本>`（宿主版本用 `dsh --version` 查） | 插件启动报 `provides no kv facet`，完全不加载 |
| 4 | LLM 适配器 + 凭据 | 一个已在 ctx.llm **注册的 provider 名**与有凭据的模型：`dsh-llm-deepseek` 注册名为 `deepseek-official`，模型 `deepseek-v4-flash` / `deepseek-v4-pro`（本插件 schema 默认的 `deepseek` / `deepseek-chat` **不在注册表**，必须覆盖） | 适配器随 dsh base bundle；凭据入宿主凭据文件 `~/.dsh/.credentials.yaml`（refs 指向 `DEEPSEEK_API_KEY`） | 判分/矛盾判定/抽取全部失败（NO_ADAPTER），写入管线空转 |
| 5 | 嵌入端点 | **推荐本地 llama.cpp**（零 key、离线、免费；部署步骤见下文「[推荐：本地部署 llama.cpp](#推荐本地部署-llamacpp-替代云端-api嵌入--重排)」一节，**含启动脚本——server 须先于宿主跑起来**）——插件默认模型 `BAAI/bge-m3`（1024 维，多语言）有现成 GGUF。云端替代：SiliconFlow key（siliconflow.cn 控制台申请） | 下节 / siliconflow.cn | 只能以 `deterministic` 冒烟（验证管线、无语义） |
| 6 | 重排端点 | 同 #5：本地 llama.cpp 开第二个 server 进程；云端替代：SiliconFlow key（默认模型 `BAAI/bge-reranker-v2-m3`；云端方案可与 #5 共用同一把） | 同 #5 | 检索链路在精排步失败（见[排障](#排障)） |
| 7 | 本插件本体 | 方式 A：本仓库构建出的 `lib/`；方式 B：本包目录经 `dsh plugin add` | 下两节 | — |

嵌入/重排的 `endpoint` / `model` / `dimensions` **出厂默认已对准 SiliconFlow**（`https://api.siliconflow.cn/v1/embeddings` + `BAAI/bge-m3`(1024)；`https://api.siliconflow.cn/v1/rerank` + `BAAI/bge-reranker-v2-m3`），云端方案真实语义只需补两把 key；换任何 OpenAI 兼容 `/embeddings` 或 `/rerank` 服务——**含本地 llama.cpp（推荐，见下节）**——时改 `endpoint` / `model` / `apiKey` 三个键、并把 `dimensions` 对齐实际模型即可。

### 方式 A：本地源码加载（开发）

> **必须先 `npm run build` 生成 `lib/`，插件路径指向 `lib/index.js`**。宿主 loader 不转译 TS，直接指 `src/index.ts` 会**静默加载失败**（无任何报错，工具与存储都不出现）。

创建 `overlay.yml`（**插件路径必须是绝对路径**）：

```yaml
- insert:
    - id: dsh-mem
      name: '/Path/To/dsh_mem/lib/index.js'
      config:
        embedding:
          mode: http            # 无 key 冒烟改为 deterministic
          apiKey: !!js process.env.MEM_EMBEDDING_API_KEY ?? ''
        reranker:
          apiKey: !!js process.env.MEM_RERANKER_API_KEY ?? ''
        llm:
          provider: deepseek-official   # dsh-llm-deepseek 的注册名；必须与宿主已注册路由一致
          model: deepseek-v4-flash
```

```sh
npx @deepseek-ai/dsh web --patch ./overlay.yml   # --dump-config 可检查最终组合
```

配置修改经 HMR 热重载生效（改 `config` 即卸载重载本插件，注册自动清理重建）。

### 方式 B：安装为组合包

```sh
dsh plugin --profile web add /Path/To/dsh_mem
```

包身份来自 `package.json` 的 `dsh.bundle.patch` → `cordis.patch.yml`；`remove` 同步移除。git 安装的授权坑（pnpm ≥10 `allowBuilds`）见 docs/ref/README。

### 必要配置最小集

其余全部有 schema 默认值，通常只需：

1. `embedding.apiKey` + `reranker.apiKey`（或两者 `mode: deterministic`）；
2. `llm.provider` / `llm.model`：M5 判分与矛盾判定的调用路由，**必须是 ctx.llm 上已注册适配器的 provider 名**，模型随之有可用凭据；
3. 切 sqlite 后端（可选但推荐）：profile patch 挂 `@deepseek-ai/dsh-storage-sqlite` 行（config 带 `path`，如 `!!js dshHomePath('storages/dsh_mem.sqlite')`）+ 本插件 `storage.backend: sqlite`。

**记忆隔离与入口（M2/M3）默认即所需行为**：`storage.workspaceScope: true` = 每工作区一库（QQ bot 等以适配器接入的会话自动归属其运行目录的工作区）；`intake.session.acceptSourceKinds: ['user']` = 只有用户消息算人类输入——IM 适配器若以自定义 `source.kind` 注入消息，把该 kind 加进列表即可（带 `form` 的插件上下文恒不算对话）；`intake.session.enabled: false` 可完全停写（检索通道不受影响）。

### 推荐：本地部署 llama.cpp 替代云端 API（嵌入 + 重排）

嵌入与重排不必走云端：插件默认的两个模型——`BAAI/bge-m3`（嵌入，1024 维，中英多语言）与 `BAAI/bge-reranker-v2-m3`（重排）——都有现成 GGUF 量化版，可用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 在本机跑成 HTTP 服务。**推荐理由：零 API key、零按量计费、记忆数据不出本机**；纯 CPU 即可胜任（有 NVIDIA GPU 时换 CUDA 构建，自动加速）。llama.cpp 没有 GUI（不像 Ollama / LM Studio 一键起服务），下面把安装、模型下载、启动、验证、接入写成可整段交给 Agent 逐步执行的步骤；坚持用云端 SiliconFlow 的跳过本节，直接走下一节的步骤（只需两把 key）。

**先记住一件事：llama.cpp 与 dsh 宿主毫无关系**——`dsh web` 的启动流程既不会拉起也不会等待这两个 server；它们没在跑，插件的每轮检索与写入都会报 fetch failed。所以第 6 步会把启动固化成**脚本**：每次使用插件前先跑脚本、探活通过，再启动宿主；长期使用建议注册成开机自启。

**第 1 步 · 安装 llama.cpp**（任选其一）：

```sh
# Windows（winget，或从 GitHub Releases 下载预编译包解压，免安装）
winget install llama.cpp
#   Releases：https://github.com/ggml-org/llama.cpp/releases/latest
#   有 NVIDIA GPU 选 llama-<版本>-bin-win-cuda-12.4-x64.zip；纯 CPU 选 …-bin-win-avx2-x64.zip

# macOS（Homebrew）
brew install llama.cpp

# Linux（源码构建，需 cmake ≥ 3.14 与编译工具链）
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --config Release   # 产物 build/bin/llama-server
```

完成判据：`llama-server --version` 打印版本号。下文命令假设它在 PATH 里；解压安装的把 `llama-server` 换成解压目录里的可执行文件路径即可。

**第 2 步 · 下载两个 GGUF 模型**（与出厂默认的 SiliconFlow 模型同款，Q8_0 量化精度损失可忽略）：

```sh
mkdir -p models && cd models
# 嵌入：llama.cpp 官方组织的 bge-m3 转换（Q8_0 约 0.6 GB）
huggingface-cli download ggml-org/bge-m3-Q8_0-GGUF --local-dir .
# 重排：常用的 bge-reranker-v2-m3 转换（Q8_0 约 0.6 GB）
huggingface-cli download gpustack/bge-reranker-v2-m3-GGUF --local-dir .
```

- 国内网络不畅：先 `export HF_ENDPOINT=https://hf-mirror.com`（Windows PowerShell：`$env:HF_ENDPOINT='https://hf-mirror.com'`）再执行，或改用 ModelScope 等镜像站下载同款文件；
- 完成判据：`models/` 下有 `bge-m3-Q8_0.gguf` 与 `bge-reranker-v2-m3-Q8_0.gguf`（具体文件名以仓库页为准）。

**第 3 步 · 启动嵌入 server**（一个 llama-server 只能加载一个模型、一种池化，嵌入与重排要开**两个进程、两个端口**）：

```sh
llama-server -m models/bge-m3-Q8_0.gguf --embeddings --pooling cls -c 8192 --port 8080
```

- `--embeddings` 进入嵌入模式；`--pooling cls` 是 bge-m3 密向量的池化方式（新版会从 GGUF 元数据推断，显式写最稳）；`-c 8192` 是 bge-m3 原生上下文长度；CUDA 构建默认自动把全部层放上 GPU，无需额外参数；
- 完成判据（另开一个终端执行）：

```sh
curl http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":["部署延迟与配置复杂性相关"]}'
# → {"data":[{"embedding":[…共 1024 个数…],"index":0,…}]}
```

**第 4 步 · 启动重排 server**（第二个进程）：

```sh
llama-server -m models/bge-reranker-v2-m3-Q8_0.gguf --reranking --port 8081
```

- `--reranking` 进入重排模式并开放 Cohere/Jina 式 `/v1/rerank` 端点；llama.cpp 直接返回模型原始输出（logit 量纲），本插件检测到越出 [0,1] 的分数会自动做 sigmoid 归一，无需额外配置；
- 完成判据：

```sh
curl http://127.0.0.1:8081/v1/rerank -H "Content-Type: application/json" \
  -d '{"model":"reranker","query":"用什么数据库","documents":["生产数据库是 PostgreSQL","今天天气不错"]}'
# → {"results":[{"index":0,"relevance_score":…},{"index":1,"relevance_score":…}]}
#   相关文档（第 0 条）的分数应显著更高
```

**第 5 步 · 接入本插件**：把出厂默认的 SiliconFlow 端点换成两个本地 server。方式 A 改 `overlay.yml` 的 `config`；方式 B 在 profile patch 的 dsh-mem 条目 `config` 里改（覆盖会整体替换 config，须完整重述）：

```yaml
        embedding:
          mode: http
          endpoint: 'http://127.0.0.1:8080/v1/embeddings'
          model: 'bge-m3'          # llama.cpp 不校验 model 字段，占位即可
          dimensions: 1024         # bge-m3 密向量维度
          apiKey: ''               # 本地服务无需鉴权，留空
        reranker:
          mode: http
          endpoint: 'http://127.0.0.1:8081/v1/rerank'
          model: 'reranker'
          apiKey: ''
```

- 从 `deterministic` 切来：**必须先清库**——旧记录的嵌入是词袋哈希，换真实嵌入后是噪声（换嵌入模型属不兼容变更，全景图 B-5）：先停宿主进程（防 SQLITE_BUSY），删 `~/.dsh/storages/dsh_mem.sqlite`；
- 从 SiliconFlow 的 bge-m3 切来：**也建议清库**——云端 f16 与本地 Q8_0 的向量分布有微小系统差，新旧混检会轻微劣化；新库从零积累没有迁移成本；
- 内部判分 LLM（`llm.provider` / `llm.model`）走宿主 ctx.llm 适配器（如 `deepseek-official`），与嵌入/重排端点无关，不用改；
- 完成判据：重启 `npx @deepseek-ai/dsh web` 后聊天 UI 出现 `memory_search` 工具卡片，对模型说「记住：我的生产数据库是 PostgreSQL」后库文件增长（同下一节第 4 步的三项确认）。

**第 6 步 · 写成启动脚本并常驻（重要）**：把第 3/4 步的两条命令固化成脚本——**每次使用插件前先跑它**。llama.cpp 与宿主无关，宿主不会替你拉起，忘了启动 = 每轮检索/写入报 fetch failed；交给 Agent 执行时更要明确交代这一步，脚本不在宿主启动链路里，Agent 不会自动想起它。

Windows（存为 `start-llamacpp.ps1`，路径按实际改；跑一次拉起两个窗口）：

```powershell
$llama = 'llama-server'          # 解压安装写 llama-server.exe 的绝对路径
$dir   = 'C:\path\to\models'     # 第 2 步两个 GGUF 所在目录

Start-Process $llama "-m `"$dir\bge-m3-Q8_0.gguf`" --embeddings --pooling cls -c 8192 --port 8080"
Start-Process $llama "-m `"$dir\bge-reranker-v2-m3-Q8_0.gguf`" --reranking --port 8081"
```

macOS / Linux（存为 `start-llamacpp.sh`，`chmod +x` 后运行；路径按实际改）：

```sh
#!/usr/bin/env sh
LLAMA=llama-server          # 解压安装写绝对路径
DIR=models                  # 第 2 步两个 GGUF 所在目录（相对运行目录）

nohup "$LLAMA" -m "$DIR/bge-m3-Q8_0.gguf" --embeddings --pooling cls -c 8192 --port 8080 >llamacpp-embed.log 2>&1 &
nohup "$LLAMA" -m "$DIR/bge-reranker-v2-m3-Q8_0.gguf" --reranking --port 8081 >llamacpp-rerank.log 2>&1 &
```

跑完用公开探活端点确认两个 server 就绪（`/health` 无需鉴权；`{"status":"ok"}` = 可用，`loading model` = 还在加载）：

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8081/health
```

- 推荐顺序固定为：**跑启动脚本 → 探活通过 → `npx @deepseek-ai/dsh web`**；长期使用把脚本注册成系统自启（Windows「任务计划程序」、Linux systemd、macOS launchd），开机即就绪；
- 宿主运行中 server 掉线不致命：期间检索与写入报 fetch 错误，重新跑脚本、探活通过后自动恢复，宿主无需重启。

### 从冒烟到可用：切换真实端点（可整段交给 Agent 逐步执行）

`deterministic` 冒烟只证明管线连通，**记不住语义**。以下把「切换真实端点 + 验证 + 使用约束」写成可执行步骤；全部改动只落在两个地方：profile patch 文件 `~/.dsh/profiles/web/cordis.patch.yml` 和（可选）一个库文件。每步带完成判据，Agent 可逐步自验。

**第 1 步 · 存储后端**：执行 `dsh --version` 记下宿主版本（下称 `<host>`）；patch 尚无 sqlite 后端时执行 `dsh plugin --profile web add @deepseek-ai/dsh-storage-sqlite@<host>`（**逐位一致**），并在 patch 顶部 `- insert:` 列表加挂载条目：

```yaml
    - id: storage-sqlite
      name: '@deepseek-ai/dsh-storage-sqlite'
      config:
        path: !!js dshHomePath('storages/dsh_mem.sqlite')   # 库文件位置，可改
```

完成判据：重启后日志无 `provides no kv facet`。

**第 2 步 · 插件条目与 key**：patch 的同一个 `- insert:` 列表加（或改成）：

```yaml
    - id: dsh-mem
      name: '<本仓库绝对路径>/lib/index.js'
      # 方式 B（dsh plugin add）安装时不必手写本条目：包内 cordis.patch.yml 自动生效，
      # 只需按 id 覆盖 config（覆盖会整体替换 config，须完整重述）。
      config:
        storage:
          backend: sqlite
          unit: dsh_mem
        embedding:
          mode: http                            # deterministic → http
          apiKey: '<SiliconFlow key>'           # 或 !!js process.env.MEM_EMBEDDING_API_KEY ?? ''
        reranker:
          apiKey: '<同一把 SiliconFlow key>'     # 或 !!js process.env.MEM_RERANKER_API_KEY ?? ''
        llm:
          provider: deepseek-official           # dsh-llm-deepseek 的注册名，必须与宿主一致
          model: deepseek-v4-flash
```

`endpoint` / `model` / `dimensions` 不必写（出厂默认即 SiliconFlow `bge-m3`·1024 / `bge-reranker-v2-m3`）。走本地 llama.cpp 的不用这两把 key：把 `embedding` / `reranker` 两段换成「推荐：本地部署 llama.cpp」第 5 步的端点配置即可，并记得那两个 server 与宿主无关——**先跑该节第 6 步的启动脚本、探活通过，再起宿主**。完成判据：重启 `npx @deepseek-ai/dsh web` 后聊天 UI 出现 `memory_search` 工具卡片。

**第 3 步 · 清理 deterministic 时代的旧库**（此前用 deterministic 跑过对话才需要）：旧记录的嵌入是词袋哈希，换真实嵌入后是噪声（换嵌入模型属不兼容变更，全景图 B-5）。**先停宿主进程**（防 SQLITE_BUSY），删除 `~/.dsh/storages/dsh_mem.sqlite`。没有免清理的迁移：把 `storage.version` 改成 `2` 只会让旧介质开库报 `version-mismatch`（那是强制清理的信号），最终仍要删库。

**第 4 步 · 重启并验证**：`npx @deepseek-ai/dsh web`，然后依次确认——

1. 聊天 UI 出现 `memory_search` 工具卡片（宿主 console 压掉插件日志，**以卡片和库文件为准**，日志仅作旁证）；
2. 对模型说「记住：我的生产数据库是 PostgreSQL」，稍候看 `~/.dsh/storages/dsh_mem.sqlite` 体积增长（或日志 `turn N: wrote 1, …`）；
3. 新开一轮问「我用什么数据库」——回答应来自记忆（自动注入的 section 或 `memory_search` 调用），而不是模型现编。

**第 5 步 · 日常使用三条约束**（使用习惯，不是配置）：

1. **一个 profile 一个进程**：web GUI 与 headless 并行写同一 sqlite 库会 `SQLITE_BUSY`（后端无 busy-wait/重试）；
2. **写入靠长驻宿主**：headless 一次性进程回复完就退，写入管线（多次 LLM 判分，秒级）来不及落盘（全景图 F-13）——headless 只适合查询，不适合喂记忆；
3. **睡眠周期是慢工序**：攒满 10 条新写入才在后台跑一次（聚类→蒸馏→迁移），不跑不影响日常读写；其聚类/矛盾阈值按真实嵌入设计，`deterministic` 下基本不触发属正常。

### 确认插件在跑

- 启动日志：`[dsh-mem] started: backend=…, workspaceScope=…, clearTool=…`；首个会话事件后：`scope opened: unit=… (N valid memories)`；
- 每轮 `turn/end` 后（有写入时）：`turn N: wrote X, superseded Y, dropped Z`；
- 睡眠周期触发后：`sleep cycle: C clusters (M migrated, G gated), refreshed R records / N nodes, decayed R'/N'`；
- 聊天 UI 出现 `memory_search` 工具卡片（开启 `tool.clear` 后另有 `memory_clear` 卡片）；库非空且命中时，`memory_search` 结果与自动记忆 section 出现在上下文（语义节点带 `gist` 标记）；
- storage 介质上出现 unit：json 后端每 unit 一个 `~/.dsh/storages/<unit>.json`；sqlite 后端全部 unit 在同一 `<path>.sqlite` 内（表名 `u_<unit>_episodic` / `u_<unit>_semantic`）。

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

### 2a. `memory_clear` 工具（开发/测试用，默认关闭）

`config.tool.clear: true` 开启后，聊天 UI 出现 `memory_clear` 卡片：一次性**硬删除当前工作区记忆库的双表全部内容**（含被取代与已归档的记忆），**时间旅行（`at`）也不可回溯**——全系统唯一违反「记录永不删除」的操作，仅供测试迭代快速清库；生产 profile 保持默认关闭，跨工作区全清仍以删介质文件为准。

| 参数 | 类型 | 说明 |
|---|---|---|
| `confirm` | boolean，必填 | 必须严格为 `true`：缺失由宿主参数校验拒绝；`false` 返回规范 error 值且不触碰存储 |

返回单个规范 JSON 值（删除计数）：

```json
{ "episodic": 3, "semantic": 2, "total": 5 }
```

### 3. 持久记忆库（经宿主 ctx.storage 持久化，介质由存储后端决定）

本包**不自带数据库/存储引擎**，也不直接读写文件：一切持久化经宿主的存储系统 `ctx.storage`（接口由 `@deepseek-ai/dsh-storage` 定义）。插件代码只见 KV unit 抽象——`storage.backend.get(name).kv.open(descriptor)` 打开一个 unit（描述符：两张逻辑表、无全局槽、带格式版本），之后以 `putRecord` / `loadAll` / `close` 读写；实际介质与落盘位置由 profile 里安装的**存储后端**决定（json 后端 = 每 unit 一个 JSON 文件；sqlite 后端 = 一个真实 SQLite 库文件）。

- **按工作区隔离（M2）**：记忆库的 scope = 会话创建时的工作目录（`session.header.cwd`，宿主验证过的绝对路径）。每个工作区一个 KV unit（`<unit>_ws_<8位哈希>`，路径归一化后 FNV-1a），写入/检索/睡眠/计分全套按作用域独立——**物理隔离，跨工作区不可能串库**。无 cwd 的会话（如部分 headless 场景）回退到 `config.storage.unit` 本名；`storage.workspaceScope: false` 可恢复全局单库旧行为。
- **存储后端（M1）**：后端是宿主侧独立安装的插件包，本包不含任何后端。生产推荐 `sqlite`（`dsh-storage-sqlite`，一个库文件承载全部 unit，消灭 json 后端 pretty-print 的体积膨胀；安装需与宿主版本精确对齐，见[前置条件](#前置条件)）；json 后端（dsh-base 自带）保留为回退与测试用途。
- KV unit（前缀默认 `dsh_mem`，`config.storage.unit` 可改；unit 名只允许小写字母/数字/下划线，**不允许连字符**——宿主规则 `/^[a-z][a-z0-9_]*$/`），两张逻辑表：`episodic`（MemoryRecord）与 `semantic`（SemanticNode，睡眠周期填充）——json 后端下是文件里的两个键，sqlite 后端下是两张 key/value 表 `u_<unit>_episodic` / `u_<unit>_semantic`（见[确认插件在跑](#确认插件在跑)）。
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

- **成功使用 = 进入最终上下文**（两条检索通道的 top 输出，scoring.ts 记账）。设计所列"被采纳"信号（ctx.messageFeedback）不在当前依赖集内（专用包 `@deepseek-ai/dsh-message-feedback` 已发布但尚未纳入依赖集，P1 对 ctx.schedule 的同款取舍）：按"未被纠正"默认成立即时入账，接入后可改为确认制。
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
| `tests/` | 243 个离线单测（vitest，无需网络与 API key） |
| `scripts/eval-ndcg.mjs` | §7.1 检索四档消融脚手架 |
| `scripts/eval-sleep.mjs` | P1 睡眠周期脚手架（QA 保留率、压缩后召回存续） |
| `scripts/eval-semtime.mjs` | P2 语义 vs 时间消融脚手架（§7.4 关键消融 + 时间旅行） |
| `benchmark/` | 性能与空间基准套件（插入/检索/写入/睡眠/空间五项，对 `lib/` 构建产物计时，含真实 sqlite 后端）——实测数字、四堵规模墙与优化优先级见 `benchmark/README.md` |
| `data/` | 示例语料与查询集（格式见[评测](#评测ndcg-四档消融)） |
| `docs/` | 设计文档（定稿）、`docs/architecture/插件工作原理全景图.md`（内部细节唯一引用点）、dsh 官方文档镜像 `docs/ref/`、立项认知机制对话 `1.md` |
| `cordis.patch.yml` | 组合包 patch（`dsh plugin add` 时使用） |

### 6. 当前验证状态

| 门禁 | 结果 |
|---|---|
| `npm run typecheck`（strict + noUncheckedIndexedAccess） | 0 错误 |
| `npm test` | 211/211 通过 |
| `npm run build` → `lib/` | 通过；`node` 导入 `lib/index.js` 与 Config 默认值校验通过 |
| 非法配置响亮失败 | Schema 加载时抛 `ValidationError`（如 `score.useGain: "not-a-number"`） |
| sqlite 后端探针（M1） | headless one-shot 实测：后端注册 → kv.open → putRecord/loadAll 往返（嵌套 JSON 完好） |
| 双工作区路由（M2） | 从两个不同 cwd 各跑一轮 headless 对话：开出两个哈希与 scope.ts 预期**逐字符一致**的工作区 unit；回退 unit 存在且空 |
| eval-ndcg 冒烟（deterministic 替身） | 四档管线全部跑通（数值无语义意义） |
| eval-sleep 冒烟（echo LLM + deterministic 替身） | 聚类→gist→QA→迁移→统计刷新→S 衰减全链路跑通（数值无语义意义） |
| eval-semtime 冒烟（deterministic 替身） | semantic / time / ga-product 三臂 + 时间旅行复活计数跑通（数值无语义意义） |
| 性能与空间基准（benchmark/，真实 sqlite 后端） | 五项全绿：put O(1)（sqlite ~3k ops/s）/ 检索全管线 O(N)（2 万条 93ms、QPS 11）/ 每轮写入 O(N·c)（1 万条库 c=4 约 0.42s）/ 睡眠聚类 O(N²)（≈21–25 万对/s）/ sqlite 体积 = 紧凑 JSON×1.29、嵌入占逻辑负载 83.5%（int8+base64 量化假设 −41%）；规模墙与优化建议见 `benchmark/README.md`（全景图 F-14） |

**尚未验证**（需外部资源，见验收目标）：真实嵌入/重排端点下的 nDCG 与语义 vs 时间消融数字、M5 判分与人工标注的 Spearman ρ > 0.7、真实 LLM 下 QA 闸门保留率 ≥ 0.8 与长程问答提升、messageFeedback"被采纳"确认制接线（message-feedback 包类型进入依赖集后），以及长驻宿主中跨工作区的完整对话验证（M2 的路由已在 headless 实测；headless 一次性进程的写入退出竞态见全景图 F-13）。


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
npm test                                   # vitest run（全部 211 个）
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
  tool.ts         memory_search 工具（含 at 时间旅行参数）+ memory_clear（开发/测试硬删除，默认关）+ section 文本渲染
```

实现选择说明：自动注入落在 `system-prompt/assemble` waterfall（agent 定域注册）而非 `ctx.systemPrompt.section()`——后者 text provider 是同步签名，承载不了异步检索链路；前者为官方文档同章节列出的扩展点。检索查询取自 `agent/inbox/claimed` 的待处理输入 ⊕ 最近对话尾部。

## 配置速览（默认值见 `src/config.ts`）

| 键 | 说明 |
|---|---|
| `storage.backend` / `unit` / `version` / `workspaceScope`(true) | KV 后端名（json/sqlite）、unit 名前缀与无 cwd 回退名（**不含连字符**，规则同宿主 `/^[a-z][a-z0-9_]*$/`，连字符会被归一为下划线）、格式版本、按工作区一区一库开关（false = 全局单库旧行为；工作区 unit 名 = `<unit>_ws_<路径哈希>`） |
| `intake.session.enabled`(true) / `acceptSourceKinds`(['user']) | 会话历史入口：停写开关（检索不受影响）与"什么算人类输入"的 source.kind 白名单（IM 适配器自定义 kind 加进列表即可；带 `form` 的插件上下文恒不算对话） |
| `emotion.workspaces`(**[]**) / `alpha`(0.15≤0.2) / `tauHours`(4) / `disclose`(false) | 按工作区的情绪编码（P3-1）：白名单（归一化路径精确匹配，空=全部 bypass 恒等）、情绪重排权重（只进语义筛选后的排序键，cold 门控/L0 召回纯语义）、mood 衰减常数、心境披露开关 |
| `graph.synonymCos`(0.8) / `maxDegree`(8) | 语义图 related 边（P3-2）：同义边余弦阈值与每记录度数上限（聚类建边 + 矛盾调用扩产出 + 同义边三件套） |
| `embedding.*` | mode(http/deterministic)、endpoint、model、apiKey、dimensions(1024)、batchSize——endpoint 出厂默认 SiliconFlow；本地 llama.cpp 为 `http://127.0.0.1:8080/v1/embeddings`（见「推荐：本地部署 llama.cpp」） |
| `reranker.*` | 同上（无 dimensions；本地 llama.cpp 为 `http://127.0.0.1:8081/v1/rerank`） |
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
| `tool.enabled` / `maxResults`(5) / `clear`(false) | memory_search 工具；`clear` 开启 `memory_clear`（按工作区硬删除全部长期记忆的开发/测试工具，唯一违反「永不删除」的操作，仅供测试迭代） |

## 排障

| 现象 | 原因与处理 |
|---|---|
| 启动报 `storage backend '…' provides no kv facet` | profile 没有 storage 后端或 `storage.backend` 名不对：装 `dsh-storage-json`/`dsh-storage-sqlite` 并核对注册名 |
| `dsh plugin add` 装 sqlite 后端被拒（incompatible / peer 闸） | 版本须与宿主**精确一致**（宿主 `dsh --version` 是 0.2.1-rc.2 就装 `@deepseek-ai/dsh-storage-sqlite@0.2.1-rc.2`）；报错信息里会给出 `allow-version` 豁免命令，但存储层不建议豁免版本差 |
| sqlite 写入报 SQLITE_BUSY | 同一库文件被两个进程同时写（如 web GUI 与 headless --patch 并行）：后端无 busy-wait/重试，保持"一个 profile 一个进程" |
| 插件停在 PENDING 不加载 | `inject` 的服务（storage/llm/tools/systemPrompt/sessions/agents）缺提供方；查 `--dump-config` 与启动日志 |
| 改配置热重载后某工作区持续报 `already-open`、记忆读写全失败 | 旧版本 HMR 卸载竞争 bug：旧实例残留在存量 agent 上的 `system-prompt/assemble` 监听器把刚关掉的 unit 又打开，新实例再开即被宿主拒（重载次数越多越容易触发）。已修复——卸载时显式摘除存量监听器、启动时补挂、ScopeRegistry 关闭后不再重开。升级后不会再触发；若升级前已处于损坏状态（unit 已被旧实例的 store 占住），重启宿主一次释放即恢复 |
| 打开 unit 报 `version-mismatch` | `config.storage.version` 与介质上既有版本不一致；P1 仅向记录新增可选字段（旧记录缺字段按 undefined 容忍），保持 version=1 即可 |
| 判分调用失败（NO_ADAPTER / 空回复 / error finish） | `llm.provider` 未注册适配器或凭据失效；llm.ts 会抛出携带失败码的错误，见 dsh 日志 |
| 嵌入/重排 401、超时 | 核对 apiKey、endpoint、网络；临时可切 `deterministic` 模式隔离问题 |
| 本地 llama.cpp 报 `fetch failed` / ECONNREFUSED | 最常见原因：server 根本没启动——llama.cpp 与宿主无关，宿主不会拉起它。跑部署节第 6 步的启动脚本，再 `curl http://127.0.0.1:<端口>/health` 探活（`{"status":"ok"}` 才可用）；其次查端口写错。server 拉起后宿主自动恢复，无需重启 |
| 本地 rerank 排名异常 / 分数全挤在一起 | llama.cpp 版本过旧的 rerank 已知缺陷，升级到最新 release；本插件会把越出 [0,1] 的原始 logit 自动 sigmoid 归一，正常无需干预 |
| llama-server 启动即退出 / 嵌入质量异常 | 模式旗标用错：嵌入用 `--embeddings --pooling cls`，重排用 `--reranking`；并确认 GGUF 下载完整（中断会产生半个文件，重下） |
| 睡眠周期从不触发 | `sleep.enabled` 为 false、新写入未达 `sleep.minNewMemories`、或聚类池小于 `clusterMinSize`；看启动后是否有 `sleep cycle:` 日志 |
| 睡眠周期全部 gated（qa_failed） | QA 闸门在拒绝：gist 质量不足（换更强判分模型）或 `qaF1Threshold` 过严；被拒簇已标 incompressible，需新成员加入才重试 |
| 压缩后检索不到原细节 | 设计行为：成员内容由节点 gist 承载（§2.3"记住要点、忘掉细节"）；回溯经 `flags.compressed_into` 索引原记录，或用 `memory_search` 的 `at` 参数在归档前的时点检索 |
| 记忆还在但排不上名 | tier 可能已沉到 cold（S 动态：长期未使用 + 兜底衰减）：cold 项需 CSLS cue ≥ `cueThreshold + coldCueBoost` 才保留；调 `tier.coldThreshold` / `score.decayTauDays`，或等使用事件把它推回 warm |
| `memory_search` 返回 `invalid 'at'` | `at` 需为 ISO 8601（以 4 位年份开头）或正整数 epoch ms；领域级错误写进规范值，不影响工具通道 |
| 情绪重排不生效 / mood 不更新 | 工作区不在 `emotion.workspaces` 白名单（默认空 = 全部 bypass）；核对路径写法（大小写/分隔符不敏感但须精确匹配，非前缀）；mood τ=4h 衰减，隔日归零属正常；消融脚本 `npm run eval:emotion` 可离线验证 |
| 聊天 UI 没有 `memory_clear` 工具卡片 | 默认关闭（破坏性工具）：config 加 `tool: { clear: true }`（HMR 即生效），启动日志 `clearTool=true` 旁证；它只清**当前工作区**的库，换嵌入模型/版本迁移等全清场景仍以删介质文件为准 |

## 明确不在 P2（后续里程碑）

messageFeedback"被采纳"确认制（S 增益目前按"未被纠正"默认即时入账，§5.0 所列 seam 的类型不在依赖集内）、LoCoMo 式活宿主端到端回放；语义图 PPR 联想检索、spacing 加权、sur、valence 情绪一致性检索（P3，设计 §6/§10）。

已知接受的取舍（P2）：跨写链并发 `updateRecord` 的丢失更新窗口延续 P1 声明（最坏一次 flag/计数/S 增量，可被下一周期自愈）；写入路径与睡眠周期可并发写同一 unit 的不同记录（KvUnit 单调用原子、整链不串行是宿主契约给调用方的自由度）；兜底衰减与全库统计刷新同门控（`sleep.enabled=false` 时离线维护整体停摆，事件路径的惰性衰减追补仍然生效）。另：`memory_clear`（`tool.clear` 开启时）是全库唯一硬删除路径（「永不删除」的唯一例外），按作用域清库、时间旅行不可回溯，仅供测试迭代（全景图 §9.4）。

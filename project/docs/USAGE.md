# 使用说明

面向第一次使用本项目的开发者：从启动服务、配置模型，到导入资料、检索和问答。
环境搭建的完整步骤见 [../../已有资源/文档/07-环境搭建与依赖.md](../../已有资源/文档/07-环境搭建与依赖.md)，
架构与设计见 [DESIGN.md](DESIGN.md)，后续任务见 [ROADMAP.md](ROADMAP.md)。

## 1. 启动

```bash
conda activate rag
cd project

# 安装（首次）
pip install -e ".[dev,milvus]"
cp config.example.json config.json

# Milvus（混合检索依赖；没有它时会回退关键词索引）
docker start milvus-standalone
curl -s http://localhost:9091/healthz
```

PDF/扫描件由 `connector_pdf` 独立插件处理，需要额外的 docparse 环境并启动 Ollama；
docparse CLI 启动前会检查 Ollama 可达，原生 PDF/DOCX 也需服务在跑（但不会实际调用 OCR）。
不处理 PDF 时可以先跳过。Markdown 和 HTML 插件不需要额外服务。
只用关键词索引的离线模式：

```bash
RAG_INDEX_BACKEND=keyword rag-cli search "关键词"
```

## 2. 两种使用方式

### 交互式（推荐日常使用）

```bash
rag-cli
```

```text
RAG Multidocs CLI
config=config.json  index=milvus  chat=未配置  credentials=.rag/credentials.json
直接输入问题开始；/help 查看命令，输入 exit 或 Ctrl-D 退出。

rag> 检索器如何训练
rag> /search 表格
rag> /exit
```

- 直接输入文字：检索 + 生成答案（配置了 chat 时），答案流式输出并带 `[n]` 引用；
- 以 `/` 开头的是命令；
- 常见命令可以省略 `/`，例如 `setup`、`help`、`exit`；
- `exit` / `quit` / `q` / `/exit` / Ctrl-D 都可以退出。

### 单条命令（脚本/CI）

```bash
rag-cli ingest --source txt --path samples/notes.txt
rag-cli search "检索器如何训练"
rag-cli ask "检索器如何训练" --stream
rag-cli -p "检索器如何训练"          # 一次性问答
```

## 3. 第一次配置模型

```bash
rag-cli setup
```

向导分两步：

1. **Chat 模型**：负责生成答案，可选。选 `0` 或直接回车可关闭，只做检索；
2. **Embedding 模型**：负责入库和查询时的向量化，默认
   `local / BAAI/bge-small-zh-v1.5 / dim=512`。

需要 API Key 的 provider（DashScope、OpenAI、DeepSeek）会在选择后提示隐藏输入；
留空表示稍后设置。配置结束后会打印当前选择，例如：

```text
当前选择：chat=deepseek/deepseek-chat  embedding=local/BAAI/bge-small-zh-v1.5 dim=512
```

也可以分步配置：

```bash
rag-cli models list
rag-cli models use --kind chat --provider deepseek --model deepseek-chat
rag-cli models use --kind embedding --provider dashscope --model text-embedding-v4 --dim 1024
rag-cli keys set --provider deepseek
```

## 4. chat 和 embedding 的区别

| 类型 | 作用 | 何时使用 | 换模型的影响 |
| --- | --- | --- | --- |
| chat | 把检索到的证据交给 LLM 生成答案 | `ask`、交互式提问 | 下次提问生效，无需重建索引 |
| embedding | 入库切块向量化 + 查询向量化 | `ingest`、`search`、`ask` | 维度可能变化，必须新建 Milvus 集合并重新导入 |

当前选择的 embedding 可以用 `/config` 或 `rag-cli models list` 查看（带 `*`）。

**换 embedding 的注意事项**：不同 provider/模型/维度的向量不能混用。
切换后请修改 `config.json` 的 `milvus_collection`（例如 `rag_chunks_dashscope`），
再重新 `rag-cli ingest`；旧集合可以保留，切回去时改回集合名即可。

自动按模型/维度命名集合仍在路线图中（P1-3），当前需要手动改集合名。

## 5. 导入与检索

```bash
# TXT
rag-cli ingest --source txt --path samples/notes.txt

# Markdown 目录（独立 Connector）
rag-cli ingest --source connector.markdown --path samples/markdown

# HTML 目录（独立 Connector）
rag-cli ingest --source connector.html --path samples/html

# PDF（独立 Connector，需要 docparse 环境）
rag-cli ingest --source pdf --path /path/to/file.pdf

# 检索
rag-cli search "表格" --top-k 3

# 文档、连接器、索引状态
rag-cli docs list
rag-cli connectors list
rag-cli stats
```

交互模式下的等价命令：

```text
rag> /ingest txt samples/notes.txt
rag> /search 表格
rag> /docs
rag> /stats
rag> /connectors
```

## 6. 本地 API Key 管理

API Key 不写入 `config.json`。默认保存在项目内的
`.rag/credentials.json`：

- 文件权限 `0600`，目录 `0700`；
- `.rag/` 已在 `.gitignore` 中，不会被提交；
- 文件内是**明文密钥**，只建议用于个人/课程环境；
- 已设置的 shell 环境变量优先级更高。

```bash
rag-cli setup                         # 向导里隐藏输入
rag-cli keys set --provider deepseek  # 单独保存/修改
rag-cli keys list                     # 脱敏显示，标注来源 env/file
rag-cli keys unset --provider deepseek
rag-cli keys path                     # 凭证文件位置
```

`rag-cli` 启动时会把文件里的 Key 加载到**当前进程**的环境变量，不会修改父 shell。
需要让当前 shell 也生效时：

```bash
eval "$(rag-cli keys export)"   # 会输出明文，只在确有必要时使用
```

自定义凭证文件位置：

```bash
export RAG_CREDENTIALS=/path/to/credentials.json
# 或在 config.json 中设置 "credentials_file"
```

支持的环境变量：`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`；
`local`/`hashing`/Ollama 不需要 Key。

## 7. 交互命令速查

| 命令 | 作用 |
| --- | --- |
| 直接输入问题 | 检索 + 生成答案 |
| `/search <问题>` | 只检索，不调用 LLM |
| `/ask <问题>` | 检索并生成答案 |
| `/ingest <source> <path> [--mode auto\|page\|region]` | 导入文档 |
| `/docs` | 文档列表 |
| `/stats` | 索引统计 |
| `/connectors` | 连接器及启用状态 |
| `/models` | 所有 provider、当前选择、Key 状态 |
| `/model <chat\|embedding> <provider> [模型名] [--dim N]` | 选择模型；chat 传 `none` 关闭问答 |
| `/keys [list\|set <provider>\|unset <provider>\|path]` | 管理本地 Key |
| `/setup` | 两步配置 chat/embedding |
| `/config` | 当前配置（不显示 Key） |
| `/clear` | 清屏 |
| `/help` | 帮助 |
| `/exit`、`/quit`、`/q` | 退出 |

## 8. 常见问题

**第一次提问停顿几秒**
首次会加载 Milvus 索引和本地嵌入模型，终端会提示
`正在加载索引与嵌入模型（首次可能需要几秒）...`；同一会话内会复用。

**方向键显示 `^[[A`**
说明终端 `TERM=dumb`。REPL 会在 TTY 下自动切换到 `xterm-256color`；
如果仍不正常，启动前执行 `export TERM=xterm-256color`。

**提示未配置 chat provider**
说明当前只配置了检索。运行 `/setup` 选择 chat 模型并填 Key，
或继续只用 `/search`；也可以 `rag-cli models use --kind chat --provider none`
显式关闭生成式问答。

**Key 删除了但 `keys list` 仍显示**
如果当前 shell 设置了同名环境变量，环境变量优先。`unset` 只删除本地文件中的 Key；
用 `unset DEEPSEEK_API_KEY` 或新开一个没有该变量的 shell。

**换 embedding 后检索报维度错误**
向量维度与 Milvus 集合不一致。改 `milvus_collection` 为新集合名并重新 `ingest`。

**Milvus 不可用**
会打印 `[WARN]` 并自动回退到本地 BM25 关键词索引；也可以用
`RAG_INDEX_BACKEND=keyword` 强制离线模式。

## 9. 验证

```bash
cd project
pytest -q -m "not integration"          # 离线单元测试
ruff check src tests connectors
MILVUS_URI=http://localhost:19530 pytest -q -m integration
```

当前基线：离线单元测试 57 passed，Milvus 集成测试 1 passed（需要可用的 Milvus 服务）。
`samples/campus-corpus/` 含 TXT/Markdown/HTML/PDF/PNG/DOCX 多格式样例。

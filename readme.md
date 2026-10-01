# Doc Translator

一个网页版的文档翻译工具，支持 EPUB、MOBI / AZW3、PDF、Markdown 四类格式，可以输出双语对照或仅译文。

## 启动

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

启动后打开 http://127.0.0.1:8765 。

## 翻译服务

在页面顶部的「翻译服务」里管理。同一种类型可以添加多个（比如不同的 Key、不同的模型、走中转地址），每个服务可以单独开关、设为默认、测试。

| 类型 | 接口 | 默认地址 |
| --- | --- | --- |
| 谷歌翻译 | 免费网页接口，无需 Key，量大可能被限流 | 内置 |
| OpenAI | Chat Completions | `https://api.openai.com/v1` |
| Claude | Anthropic Messages | `https://api.anthropic.com/v1` |
| Gemini | generateContent | `https://generativelanguage.googleapis.com/v1beta` |
| Grok | OpenAI 兼容 | `https://api.x.ai/v1` |
| OpenAI 兼容接口 | Chat Completions | 自己填，DeepSeek、通义、Kimi、OpenRouter、Ollama 等 |

每个服务可以配置 API Key、名称、API 地址、模型（下拉预置 / 自定义输入 / 从接口获取列表），高级设置里有并发数、每批段落数和字符数、temperature、System Prompt 和 Prompt。

### 提示词

默认提示词和分段协议取自沉浸式翻译 1.33.3（`app/prompts.py`）：

- 简体中文、繁体中文、英语、日语、韩语、俄语、法语、西班牙语、葡萄牙语有专门的母语提示词，其他语言用通用英文模板。
- 多段批量翻译用 `[[p0]]`、`[[p1]]`… 标记分段，以 `[[source_end]]` 结尾，模型按标记返回；漏掉的段落会单独补翻。
- 书名 / 文档标题会作为上下文填进 `{{title_prompt}}`。
- 自定义时可用 `{{to}}`（目标语言）、`{{text}}`（原文）、`{{title_prompt}}`，分段协议会自动附加。

配置保存在 `data/services.json`（权限 600），API Key 不会返回给前端。

译文缓存在 `data/cache.sqlite3`。同一段文字再翻译时直接用缓存，任务中途失败后重新提交也能接着翻。

## 各格式的处理方式

| 格式 | 做法 | 输出 |
| --- | --- | --- |
| Markdown | 按块切分，跳过代码块和 front matter，保留标题、列表、引用、表格的结构 | `.md` |
| EPUB | 按 OPF spine 找出正文，翻译最内层的块级元素，目录（nav / ncx）也一起翻译，其他资源原样打包 | `.epub` |
| MOBI / AZW3 | 先解包。KF8 格式直接取里面的 EPUB，老 MOBI7 先转成 XHTML 再拼成 EPUB，之后按 EPUB 流程翻译 | `.epub`，装了 Calibre 时还会多输出一份原格式 |
| PDF | 提取文本块，擦掉原文后在原位置写入译文（字号自动缩放），公式块跳过 | 仅译文：原版式；双语：原文和译文左右并排 |

## 目录

```
app/
  main.py          FastAPI 服务：上传、任务、进度、下载
  runner.py        去重、缓存、切批、并发
  translators.py   翻译引擎（Google / OpenAI / Claude / Gemini / Grok / 兼容接口）
  services.py      翻译服务配置的增删改查
  formats/         各格式的解析和回写
static/            前端页面
tests/             测试（pytest）
```

## 已知限制

- 扫描版 PDF 需要 OCR，暂不支持。PDF 里复杂的多栏排版、表格内的文字可能会错位。
- 段落里的行内格式（加粗、链接）翻译后会丢掉，只保留纯文本，图片会保留。
- 任务状态保存在内存里，服务重启后进行中的任务会丢失，但已经翻译的段落还在缓存里。
- 服务没有做鉴权，只适合在本机使用（默认绑定 127.0.0.1）。

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

### 翻译记录

每次翻译都会留下一条记录（「记录」页，翻译页底部也显示最近 3 条），刷新页面、换一本书、重启服务都不会丢。

- 记录和文件（原文件、译文、预览）只在两种情况下删除：在列表里手动删除，或者超过保留时间。
- 保留时间在「记录」页顶部设置：7 天、30 天（默认）、90 天、1 年、永久保留或自定义天数，保存在 `data/settings.json` 的 `retention_days`（0 表示永久）。服务启动时和之后每小时检查一次。
- 服务重启时还没翻完的任务标记为「已中断」，点「继续翻译」接着翻，已翻好的段落直接用缓存。
- 失败的任务可以「重试」。

### 阅读器（边翻边预览）

Markdown、EPUB、MOBI / AZW3 提交后留在翻译页，任务出现在「最近翻译」里；点「查看进度」或「阅读」进入全屏阅读器，翻好一段显示一段（PDF 暂不支持）。

- 宽屏左右对照、窄屏上下排列，可切换「对照 / 译文 / 原文」；正文用衬线字体，标题、列表、引用各有样式。
- 左侧大纲来自文档的章节标题，圆点表示该章进度（灰未开始、蓝翻译中、绿完成），标题翻好后显示译文；窄屏收进抽屉。
- 默认跟随翻译进度；自己滚动或点大纲后暂停，底部出现「跟随翻译进度」按钮。跳到哪里，后台就优先翻译那里之后的内容。
- 地址栏是 `#read/<任务 id>`，刷新后还在原来的书；Esc 返回。
- 预览只是推送已完成的段落，不会重建输出文件，翻译速度不受影响。

### CC Switch

如果本机装了 [CC Switch](https://github.com/farion1231/cc-switch)，「翻译服务」页会多一组「CC Switch 当前生效」，直接用它里面 Claude、Codex（ChatGPT）、Gemini、Grok 当前选中的供应商：

- 只读打开 `~/.cc-switch/cc-switch.db`（可用环境变量 `CC_SWITCH_DB` 指定），每次翻译时现读，Key 不会复制到本项目。
- 在 CC Switch 里切换供应商后，点列表里的「刷新」即可生效。
- 地址和 Key 以 CC Switch 为准；模型、并发、提示词等可以在这里单独调整。
- Codex 和 Grok Build 配置了 `wire_api` / `api_backend = "responses"` 时走 OpenAI Responses API；Claude 用 `ANTHROPIC_AUTH_TOKEN` 时按 Bearer 认证。
- 官方账号登录（OAuth）的配置没有 API Key，会显示为不可用。

### 语言

源语言和目标语言都可以选，语言表（128 种，含文言文、粤语、东北话等）取自沉浸式翻译。

- 源语言默认「自动检测」：抽样检测文档语言，用于提示词里的 `{{from}}`，并匹配 `wyw2zh-CN` 这类按源语言生效的提示词。
- 已经是目标语言的段落不翻（比如英文书里引用的中文，翻成中文时跳过）。简繁互译不算同一种语言。
- 检测到文档语言和目标语言一致时，任务页会提示，对应插件的 sameLangCheck。
- 谷歌翻译按插件的代码映射转换语言（zh-HK→zh-TW、fil→tl 等），不支持的语言（如文言文）会提示换用 AI 服务。
- 译文元素带 `lang` 属性；阿拉伯语、希伯来语等从右到左的语言加 `dir="rtl"`，PDF 里右对齐。

### 提示词

默认提示词和分段协议取自沉浸式翻译 1.33.3（`app/prompts.py`）：

- 按插件 langOverrides 的 `<from>2<to>` 规则选提示词：简体中文、繁体中文、英语、日语、韩语、俄语、法语、西班牙语、葡萄牙语、文言文、东北话有专门的提示词，其他语言用通用英文模板。
- 多段批量翻译用 `[[p0]]`、`[[p1]]`… 标记分段，以 `[[source_end]]` 结尾，模型按标记返回；漏掉的段落会单独补翻。
- 书名 / 文档标题会作为上下文填进 `{{title_prompt}}`。
- 自定义时可用 `{{to}}`（目标语言）、`{{from}}`（源语言）、`{{text}}`（原文）、`{{title_prompt}}`，分段协议会自动附加。

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
  main.py          FastAPI 服务：上传、任务、预览、下载、设置
  history.py       翻译记录（data/jobs/<id>/job.json）和保留时间清理
  settings.py      应用设置（data/settings.json）
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
- 服务没有做鉴权，只适合在本机使用（默认绑定 127.0.0.1）。

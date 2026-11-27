# 网页翻译设计方案

按下快捷键后，识别当前网页上的内容并翻译，译文插在原文下方。快捷键可以修改。

## 1. 客户端形态

| 方案 | 快捷键可设置 | 能翻登录后的页面和 SPA | 权限和安全 | 结论 |
|---|---|---|---|---|
| Chrome 扩展 (MV3) | 可以，用 `chrome.commands` | 可以，直接读用户看到的 DOM | 用 `activeTab`，按快捷键时才获得当前标签页权限 | 采用 |
| 油猴脚本 | 要在脚本里自己做 | 可以 | 要预先注入所有网站；依赖 Tampermonkey | 备选 |
| 服务端抓 URL 再渲染 | 没有快捷键 | 不行（没有登录态，JS 渲染的内容抓不到） | 服务端会去请求任意 URL（SSRF 风险） | 不采用 |

## 2. 整体架构

```
按下快捷键 (chrome.commands) 或点工具栏按钮
   │
background.js (service worker)
   │ 1. chrome.scripting 注入 content.js / content.css（用 activeTab 授权）
   │ 2. 代发翻译请求到本地服务（扩展有 host_permissions，不受页面 CORS 限制）
   ▼
content.js（在页面里）
   识别段落 → 先翻视口内的 → 滚动时继续 → 把译文插到原文下方
   │  chrome.runtime.sendMessage
   ▼
POST http://127.0.0.1:8765/api/web/translate（app/routes/web.py）
   → Runner.translate_all（复用缓存、同语言跳过、标点规范化）→ 现有 Translator
```

请求统一从 background 发出。content script 自己发 fetch 会受页面 CSP、CORS 和 Chrome Private Network Access
限制，很多网站上会被拦。

## 3. 快捷键

- `manifest.json` 声明命令 `toggle-translate`，默认 `Alt+T`（Mac 上是 Option+T）。
  - 第一次按：开始翻译。
  - 再按：隐藏 / 显示译文。
  - 翻译出错后再按：重试失败的段落。
- 出错分两种：
  - 单个请求失败（翻译服务报错或超时）：相关段落下显示"翻译失败"和"重试"按钮，其余段落照常翻译。
  - 每个请求都会遇到的错误（连不上本地服务、访问令牌不对、翻译服务不存在或未启用）：停止发送所有请求。
  - 两种情况下再按快捷键，都会重试所有失败的段落。
- 命令 `toggle-mode`：切换双语对照和仅显示译文，默认没有分配按键。
- 点工具栏上的扩展按钮，效果和按快捷键一样。
- 修改方式：在 Chrome 的 `chrome://extensions/shortcuts` 页面里改。这个页面任何焦点下都能触发，会自动处理和浏览器保留键的冲突。扩展没法用代码直接写入快捷键，所以设置页用 `chrome.commands.getAll()` 显示当前快捷键，并放一个"修改快捷键"按钮跳到那个页面。
- 不在页面里监听 keydown 做自定义快捷键：那样要申请"所有网站"权限常驻注入；焦点在地址栏或 iframe 里时不生效；还会和页面自己的快捷键冲突。

## 4. 识别页面内容（content.js）

1. **分段**：从 `body` 往下遍历。
   - 已知的行内标签（`a, span, em, strong, code, sub, sup` 等）并入所在段落；
   - 其他标签和自定义元素（标签名带 `-`）当作块，各自成段；
   - 行内标签的 computed `display` 不是 inline 时也当作块；
   - 行内元素里套着块元素时（卡片式链接常见），往里递归。
2. **跳过**：
   - `script, style, noscript, template, pre, textarea, input, select, button, svg, math, canvas, iframe, video, audio, img`；
   - `[translate="no"]`、`.notranslate`、`[contenteditable]`、`[aria-hidden="true"]`、`[hidden]`；
   - 少于 2 个字母的文本、纯 URL；
   - 隐藏的行内元素（比如 Wikipedia 标题旁"[编辑]"两边的方括号）：直接忽略，不把所在句子切成两段；
   - 只给读屏软件看的文字（进入视口时盒子不超过 1px）；
   - 扩展自己插入的译文节点。
3. **按视口优先**：用 `IntersectionObserver` 只翻进入视口（上下各留一屏缓冲）的段落。每攒最多 20 段或 3000 字发一次请求，最多 2 个请求同时进行。`display:none` 的内容不会进入视口，所以等它显示出来才翻。
4. **动态内容**：用 `MutationObserver` 发现新增节点（无限滚动、SPA 切路由）后，同样走识别流程，防抖 300ms。每段按第一个节点去重，避免重复翻译；同一位置已经有译文时直接替换，避免内容刷新后插出多份译文。
5. **插入译文**：插入 `<dt-trans lang="目标语言">` 自定义元素（`display:block`），放在段落内部或段落之后。不影响原文，不打乱 `li/td/h*` 的布局，页面样式也不太会匹配到它。所在元素是 flex、grid 或行内框时（GitHub 的话题标签等），译文改为行内，紧跟原文，不然会被挤成旁边的一列。
6. **写入方式**：模型输出从不作为 HTML 插入页面。
   - 没有格式的段落：用 `textContent` 写入纯文本。
   - 带链接、粗体等格式的段落：只保留这些标签，每个标签带编号 `i="N"`，去掉所有属性，以 HTML 发送。译文先在 `DOMParser` 生成的独立文档里解析（不执行脚本，不加载资源），再用 DOM 接口重建。能和原文对上的标签（先按编号，没有编号就按标签名依次对应），换成原页面元素的副本（去掉 `id`、`name`），所以 `href` 等属性都来自页面，不来自模型。其他标签只留文字，脚本、图片等直接丢弃。按编号对应是必要的：谷歌翻译和 LLM 都会调整标签顺序。
7. **配色**：译文有浅蓝底、左侧竖线，文字颜色也和原文不同，仅显示译文时也保留，方便区分哪些文字被改过。
   - 深浅配色按每条译文背后实际的背景决定：往上找最近的不透明背景，算亮度。浏览器是否深色模式说明不了自带配色的网站，浅色页面也可能有深色页头。
   - 页面完全没设背景时，看 `color-scheme` 和 `prefers-color-scheme`。
   - 文字对比度在白底和深色底上都在 9:1 以上。
8. **仅显示译文**：把原文的文本节点就地清空，原文元素加标记后用 CSS 隐藏。节点不删除也不包装，React 等框架之后更新或删除这些节点时还能找到它们。图片、表单控件照常显示。
9. **iframe 和属性**：
   - **iframe**：注入到扩展能访问的所有 frame。顶层 frame 负责提示框和快捷键状态，iframe 跟随顶层的状态（`dt-sync`）。
   - **`placeholder`**：随页面一起翻译。
   - **`title`**：鼠标悬停 300ms 后才翻译，因为大多数提示文字不会被看到。
   - **双语显示**：属性同时显示两种文字。页面之后自己改过的值不再覆盖。
10. **状态提示**：右下角的提示框放在 shadow DOM 里，不受页面样式影响。翻译中、隐藏/显示、出错都在这里显示。

## 5. 后端接口

网页翻译的接口都写在 `app/routes/web.py`，来源检查在 `app/main.py`。

`POST /api/web/translate`：

```python
class WebTranslateRequest(BaseModel):
    texts: list[str]               # 上限：50 条，单条 10000 字
    html: list[bool] = []          # 和 texts 对齐，标出哪些是 HTML 片段（交给 Runner 的 html 参数）
    target_lang: str = "zh-CN"
    source_lang: str = "auto"
    service_id: str = ""           # 空则用默认服务
    title: str = ""                # 页面标题，作为 LLM 上下文；上限 1000 字，发给模型时截到 200 字
    alt_lang: str = ""             # 已是 target_lang 的段落改译成这个语言（划词翻译用）；空则保留原文

class WebTranslateResponse(BaseModel):
    translations: list[str | None] # 和 texts 一一对应；None 表示保持原文（空白、已是目标语言、译文和原文相同）
    langs: list[str | None]        # 和 texts 对齐：每段译文的语言（target_lang 或 alt_lang）；保留原文的为 None
    detected_lang: str
```

- **`alt_lang`**：第一遍翻译后，Runner 判定为已是目标语言的段落，再用 `alt_lang` 新建一个 Translator 翻译一遍，源语言重新检测。
  - 这个 Translator 只在确实有段落需要时才创建。所以服务不支持 `alt_lang` 时，只有需要改译的请求会失败。

其他接口：

- **`POST /api/web/lookup`**：查词。
  - 请求：选中的词（最多 100 字）、所在的句子（最多 2000 字）、`target_lang`、`source_lang`、`alt_lang`、`service_id`、`title`。
  - 返回一条词典条目 `WordEntry`：原形、音标、若干义项、在这句话里的意思、条目所用的语言。
  - 只有 LLM 服务能写词条，其他服务返回 501，扩展改为翻译选中的文字。
  - 不缓存：词义取决于所在的句子，而且查一次只是一个很短的请求。
- **`GET` / `DELETE /api/web/cache`**：查看、清理网页翻译的缓存。返回 `{"entries": N}`。

- **复用现有能力**：内部用 `Runner(translator).translate_all(texts)`。
  - 缓存按 `translator.cache_key` 跨页面共享，导航栏、页脚这类重复内容直接命中；
  - 同语言跳过、标点和编号规范化都和文档翻译一样。
- **术语表**：不生成自动术语表，每页多一次请求，延迟不划算。手动术语表仍然生效。
- **Translator**：每个请求新建一个。Translator 上有 `detected_source`、`notes` 等按请求变化的状态，多个请求共用一个实例会互相覆盖。
- **记录和缓存**：网页翻译不写入"记录"页，因为记录是围绕文件任务设计的。
  - 网页译文的缓存条目统一记在 `WEB_CACHE_OWNER` 名下，所以可以单独清理，不影响文档的缓存。和文档共有的段落，清理后仍留给文档用。
  - 缓存满了以后，先淘汰没有文档用到的最旧条目，其中就包括网页译文。浏览网页不会把书的译文挤出缓存。
- **出错**：服务配置有问题返回 400，翻译服务出错返回 502。`detail` 里是中文错误信息，扩展直接显示给用户。
- **来源检查（Origin）**：现在由中间件统一处理，对所有会改动数据的请求生效（POST/PUT/PATCH/DELETE）。
  - 放行三类请求：本服务自己的页面、浏览器扩展、不带 Origin 的请求（命令行工具）。其他一律返回 403。
  - 原因：普通网页可以用表单或 `fetch(..., {mode: "no-cors"})` 跨域 POST 到 127.0.0.1，这种请求不触发预检。网页读不到结果，但能暂停、删除任务，或消耗翻译额度。浏览器跨域发这类请求时总会带 Origin，所以按 Origin 拦截就够了。
  - 只比较主机名和端口，不比较协议，这样放在 HTTPS 反向代理后面时，本服务自己的页面也能通过。
  - 网页翻译接口另有请求体大小上限（4 MB）和 `title` 长度上限。
- **CORS**：不加 `CORSMiddleware`，保留现有的 `TrustedHostMiddleware`。
- **设置页数据**：`GET /api/services` 和 `GET /api/languages` 已经有了，设置页直接调用。

## 6. 扩展设置

设置项定义在 `extension/settings.js`，后台脚本和设置页共用。除访问令牌外，都存在 `chrome.storage.sync`。

- 服务地址：默认 `http://127.0.0.1:8765`。
  - 安装时只能访问 `127.0.0.1` 和 `localhost`。
  - 其他地址属于 `optional_host_permissions`，用户点"连接"时才请求访问权限。请求要在任何 await 之前发出，否则不算用户操作，浏览器会拒绝。
  - 服务端返回 400 时，提示把该地址加入 `ALLOWED_HOSTS`。
- 访问令牌：服务在其他机器上时，填服务端 `ACCESS_TOKEN` 的值，以 `Authorization: Bearer` 发送。
  - 存在 `chrome.storage.local`，不随账号同步到其他浏览器。
- 显示方式：双语对照（默认）或仅显示译文。命令 `toggle-mode` 也能切换，默认没有分配按键。
- 翻译范围：整个网页（默认）或只翻译正文，见 8 的四期。
- 翻译服务：默认用服务端的默认服务。选过的服务被删除或停用后，打开设置页时改回默认服务并给出提示。
- 目标语言、源语言：下拉框数据从本地服务读取。
- 选中的文字已是目标语言时译成哪种语言（`altLang`）：默认英语。
- 网页翻译缓存：显示条数，可以一键清理。
- 当前快捷键和"修改快捷键"按钮。
- 连不上本地服务时显示启动命令。

## 7. 目录和技术选型

```
extension/
  manifest.json      # MV3: permissions [activeTab, scripting, storage, contextMenus],
                     #      host_permissions [127.0.0.1, localhost], optional_host_permissions [http, https]
  background.js      # 快捷键、工具栏按钮、右键菜单、注入、代发请求、浏览器里打开的 PDF
  settings.js        # 设置项和默认值，后台脚本和设置页共用
  content.js / content.css
  selection.js       # 划词翻译和查词的气泡
  sites.js           # 按站点自动翻译：权限请求、registerContentScripts
  autostart.js       # 在自动翻译的网站上先于 content.js 运行，让它像按了快捷键一样开始翻译
  options.html / options.js
scripts/build_firefox.py          # 生成 Firefox 版到 dist/firefox/
tests/test_web.py                 # 用 mock 服务测接口：对齐、跳过、alt_lang、缓存和清理、上限、服务不存在、出错
tests/test_origin.py              # 来源检查
tests/test_extension_browser.py   # 在真实浏览器里加载扩展，配合假的服务端测页面效果
```

前端用原生 JS，和 `static/` 的写法一致，不引入构建步骤。

## 8. 分期

- **一期（已完成）**：
  - 后端接口；
  - 扩展的快捷键和工具栏按钮注入、视口懒翻译、对照模式纯文本、再按一次显示/隐藏、出错后重试；
  - MutationObserver 处理动态内容；
  - 设置页。
- **二期（已完成）**：
  - 保留行内格式：没有复用服务端的 `_sanitize_translation`，改在前端用 DOM 接口重建，见 4.6。服务端清洗过的 HTML 到了页面里还得再解析一次，在前端重建更简单，也更安全。
  - 仅译文模式：命令 `toggle-mode`。
  - iframe。
  - `placeholder` 和 `title` 属性的翻译。`alt` 不翻译，它不显示在页面上，翻译只会消耗额度。
  - 非本机服务地址：`optional_host_permissions`。
  - 译文配色：区分译文和原文，按背景自动选深浅。
- **三期（已完成）**：
  - 划词翻译：右键菜单或命令 `translate-selection`。译文显示在选区旁的气泡里，气泡用封闭的 shadow DOM，文字用 `textContent` 写入。
  - 按站点自动翻译：用户点右键菜单时请求该网站的访问权限，然后用 `registerContentScripts` 注册脚本，只注入顶层 frame。网站列表放在 `storage.sync`，访问权限按浏览器分别授予，所以每个浏览器只为已授权的网站注册脚本。
  - Firefox 适配：`extension/` 是 Chrome 版；Firefox 版由 `scripts/build_firefox.py` 生成到 `dist/firefox/`，manifest 换成 `background.scripts`，加上 `gecko.id`，最低版本 128。两种写法放在同一份 manifest 里时，Chrome 会报警告，所以分开。`web-ext lint` 没有错误，但没在 Firefox 里实际运行过。
- **四期（已完成）**：
  - 翻译范围：整个网页，或只翻译正文。有 `<main>` 的网页只翻译其中的内容，正文里的导航和侧栏除外；没有 `<main>` 的网页翻译除页头、页脚、导航、侧栏以外的全部内容。
  - 所有接口的来源检查（见 5）。
  - 代码审查发现问题的修复：页面后来改动的文字在停止变化一秒后重新翻译，同一处替换超过十次就不再翻译；行内的 notranslate 会切开句子并原样保留；内存不再无限增长；oklch 等新颜色写法；密码框不会被读取；服务端回应异常或超时都不会让页面一直等待。
- **五期（已完成）**：
  - 网页翻译缓存：译文记在 `WEB_CACHE_OWNER` 名下，设置页可以查看条数和清理；缓存满时先淘汰网页译文（见 5）。
  - 浏览器里打开的 PDF：不注入页面，而是按原地址重新下载（带着该网站的登录状态），作为翻译任务提交给本服务，然后在新标签页打开它的阅读器。同一地址在设置不变时只提交一次。
  - shadow DOM：开放和封闭的 shadow root 里的文字都翻译，插槽里的子元素各自分段。行内的宿主元素留在所在句子里；页面加载后才定义的自定义元素，在页面下一次变动时补上翻译。
  - 访问令牌：服务在其他机器上时，扩展带上服务端的 `ACCESS_TOKEN`（见 6）。
  - 不再存在的翻译服务：设置页改回默认服务并提示。
  - 折叠的面板展开后才翻译其中的文字。
  - 句子里只给读屏软件看的文字直接略过；渲染好的公式（KaTeX、MathJax、MathML）发送时换成 TeX 源码，译文里放从页面复制的公式。行内代码（`code`、`kbd`、`samp`、`var`）也照原样保留。
  - 视口里的段落先发，再按远近发其余的；还没轮到就被滚出两屏以外的段落暂不发送。
  - 单个请求失败时，相关段落显示"重试"按钮，其余段落继续翻译（见 3）。
  - 解除截断：限定行数（`line-clamp`）或限定高度并隐藏溢出的框，往上最多看五层，只解除到译文能完整显示为止；隐藏译文后恢复原样式。
  - 划词时，已是目标语言的文字改译成第二语言（`alt_lang`，默认英语）。
  - 查词：选中的是一个词或短语时，用 LLM 服务写一条词典条目，而不是翻译（`/api/web/lookup`）。

## 9. 风险和已知限制

- **隐私**：页面内容会发给当前配置的翻译服务（比如 OpenAI）。只在用户按快捷键后，或在用户设为自动翻译的网站上（见三期），才翻译页面。
- **识别不准**：复杂页面（深层嵌套 div、flex 布局的导航项）可能识别不准，或译文和原文并排显示。一期先保证文章类页面效果。
- **截断**：`line-clamp` 和固定高度造成的截断已经解除（见五期）。但只往上看五层，更外层的框仍可能截掉译文。
- **LLM 延迟**：首屏一般 2 到 5 秒，靠视口优先和小批量并发缓解。Google 翻译最快。
- **无法注入的页面**：`chrome://` 和 Chrome 应用商店不能注入。按快捷键时，工具栏按钮上显示 `×`。浏览器内置的 PDF 阅读器也不能注入，所以改为提交翻译任务（见五期）。

# Doc Translator

English | [简体中文](README.zh-CN.md)

A web application that translates documents in EPUB, MOBI / AZW3, PDF, Word (.docx), HTML, Markdown and subtitles
(SRT, VTT, ASS), producing either a bilingual version or a translation only. The interface is in Chinese; this README gives each interface label in Chinese
followed by an English gloss.

## Getting started

Python 3.12 or later is required. The `python3` shipped with macOS is 3.9; install a newer one with
`brew install python@3.12`. On Linux, use the distribution's 3.12 (Ubuntu 24.04, for example) or install one with
pyenv. If the default `python3` is older than 3.12, replace `python3` in the first command below with
`python3.12`. On Debian and Ubuntu, install `python3-venv` first.

On macOS and Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Then open http://127.0.0.1:8765.

The server exits when the terminal closes. To keep it running in the background, start it with
`nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765 > server.log 2>&1 &`; output goes to
`server.log`.

The server accepts requests only for local host names such as `127.0.0.1` and `localhost`. To access it through a
LAN IP address or a domain name, add them to the environment variable `ALLOWED_HOSTS` (comma-separated), for
example `ALLOWED_HOSTS=192.168.1.20 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8765`.

## Translation services

Services are managed under 翻译服务 (Translation services) at the top of the page. Several services of the same
type can be added (with different keys, models or relay URLs), and each one can be enabled, made the default and
tested on its own.

| Type | API | Default URL |
| --- | --- | --- |
| Google Translate | Free web endpoint, no key needed; heavy use may be rate-limited | built in |
| OpenAI | Chat Completions | `https://api.openai.com/v1` |
| Claude | Anthropic Messages | `https://api.anthropic.com/v1` |
| Gemini | generateContent | `https://generativelanguage.googleapis.com/v1beta` |
| Grok | OpenAI-compatible | `https://api.x.ai/v1` |
| OpenAI-compatible API | Chat Completions | Entered by the user, such as DeepSeek, Qwen, Kimi, OpenRouter, Ollama |

Each service has an API key, a name, an API URL and a model (from a preset list, typed in, or fetched from the
API). The advanced settings cover concurrency, paragraphs and characters per batch, temperature, the system
prompt and the prompt.

The glossary is kept in the advanced settings, one `source = translation` entry per line. By default, LLM
services also count terms that recur in the document before translating (multi-word phrases, acronyms,
frequent words), and then use one request to let the model fix their translations for the document's field;
the whole document is translated with that glossary. Each batch carries only the entries occurring in its
source text, and entries typed by hand take precedence. The generated glossary is stored with the translation
cache, so continuing, retrying and generating the other version reuse the same one. The extra request can be
turned off in the advanced settings.

### Glossary library

The 术语表 (Glossaries) page keeps named glossaries, for example one per field or per book series, in
`data/glossaries.json`. Each has a name and entries in the same `source = translation` format. Glossaries are chosen
on the translate page when a file is submitted (several can be ticked) and are stored with the record, so
continuing, retrying and generating the other version use the same ones.

- A chosen glossary is added after the service's own glossary, which takes precedence; like it, only the entries
  occurring in a batch are sent. Glossary entries are part of the cache key: after a glossary is edited, continuing
  or retranslating a record that uses it translates every paragraph again with the new entries, while generating
  the other version and clearing the cache keep using the entries of the latest run. A record whose glossary was
  deleted continues without it.
- 从 CSV / TSV 文件导入 (Import from a CSV / TSV file) appends the file's first two columns (source, translation) to
  the entries; a header row and `source = translation` lines are accepted too. Nothing is saved until the glossary
  is saved. 导出 CSV (Export CSV) downloads a UTF-8 file that Excel opens correctly.
- The API is `GET` / `POST /api/glossaries`, `PUT` / `DELETE /api/glossaries/{id}`, `POST /api/glossaries/import`
  with `{"text": ...}` and `GET /api/glossaries/{id}/export`. Submitting a job takes the ids as repeated
  `glossary_ids` form fields.

### Translation records

Every translation is saved as a record (on the 记录 (Records) page; the translate page shows the latest 3).
Records survive page reloads and server restarts. Several files can be chosen or dropped on the translate page at
once; each becomes its own record with the same settings. Files that fail to upload are listed below the button
and stay selected, so they can be submitted again once the cause is fixed. When a job finishes or fails, the page
shows a message while it is in view and a system notification while it is in the background; the browser asks
whether to allow notifications on the first submit. Notifications need localhost, 127.0.0.1 or HTTPS; over plain
HTTP on a LAN address only the message in the page is shown.

- A record and its files (source file, translations, preview) are deleted only when deleted from the list by
  hand or when the retention period runs out.
- The retention period is set at the top of the Records page: 7 days, 30 days (default), 90 days, 1 year,
  forever, or a custom number of days. It is stored as `retention_days` in `data/settings.json` (0 means
  forever). It is checked when the server starts and then every hour.
- How many jobs translate at once is set at the top of the Records page (1 to 8, default 2) and stored as
  `max_jobs` in `data/settings.json`. Further jobs show 排队中 (Queued) with their place in the queue and start in
  the order they were submitted when a job finishes, fails, is paused or is deleted. 优先翻译 (Translate next)
  moves a queued job to the front, so it takes the next free place; jobs already translating go on. A paused job
  leaves the queue and joins its end when continued. A queued job takes the translation service's configuration
  when it starts, so a model or key changed while it waits applies; a job whose service was deleted fails when its
  turn comes, and the queue goes on.
- A running job can be paused (暂停, in the list and in the reader's top bar). A PDF paused while its layout is
  analyzed or its translation written stops after the current page. With several jobs running, 全部暂停 (Pause
  all) appears above the list. A paused job is resumed with 继续翻译 (Continue) or 全部继续 (Continue all), or
  can be retranslated (重译).
- Continuing rebuilds the engine from the service's current configuration. If the model changed while the job
  was paused, the cache key changes, and translated paragraphs are translated again with the new model instead
  of being reused; with an unchanged configuration they are reused. The reader can also switch to another
  translation service.
- Jobs unfinished when the server restarts are marked 已中断 (Interrupted) and can be continued; paused jobs stay
  paused after a restart.
- The layout analysis result of a PDF is stored by file content under `data/layout/<file digest>/` and shared by
  all records of the same file. When continuing, generating the other version or uploading the same book again,
  it is read back as long as the source file, both models and the program are unchanged, so pages are not
  analyzed again (about ten minutes saved for a book of several hundred pages); after a program update the
  layout is analyzed again. Retranslating with 全部重新翻译 (Translate everything again) starts over from layout
  analysis. Deleting a record deletes the result only when no other record uses the same file.
- The boxes found by the formula detection model and the layout model are stored separately (`boxes.json` in
  the same directory). When the layout is analyzed again after a program update or 全部重新翻译, they are read
  back as long as the source file and the models are unchanged, and only the rules run again: for an 86-page
  paper with both models enabled, analysis drops from 4.5 minutes to just over 20 seconds.
- A failed job can be retried (重试).
- 清理无用缓存… (Clear unused cache), under the retention period on the Records page, clears the cached
  translations no record uses: those of web pages and of records deleted before deleting cleared their cache.
  What records use stays, so they continue and regenerate without new requests. Entries cached before the cache
  recorded whose they are (before 2026-10-31) cannot be told apart and stay too; the dialog counts them, and the
  cache drops them when it fills up. It is refused while a job is translating. The API is `GET` /
  `DELETE /api/cache/unused`.
- Generated PDFs and EPUBs can be previewed (预览) in a new browser tab without downloading. When both versions
  exist, there are separate buttons for the translation and the bilingual version. PDFs use the browser's
  built-in viewer. EPUBs are served chapter by chapter from the file, with the book's table of contents on the
  left, previous and next chapter buttons (or the left and right arrow keys), and the current chapter kept on
  reload. Scripts in the book are not run.
- A finished job can generate the other version (bilingual ⇄ translation only). This uses the model and the
  translation cache of that time and only typesets again; when the cache does not match (the prompt or the
  document's segmentation changed), it asks for a retranslation instead of translating within that request.

### Reader (preview while translating)

After a job is submitted, the page stays on the translate page and the job
appears under 最近翻译 (Recent translations). 查看进度 (View progress) or 阅读 (Read) opens the full-screen
reader, which shows each paragraph as soon as it is translated. A PDF needs its layout analyzed before
translation and every page written after it, each taking from a minute or two to several minutes for a whole
book; during these stages the job and the reader show which page has been reached. With several PDF jobs at
once, layout analysis and writing handle one job at a time, the others show 等待其他任务 (Waiting for other
jobs), and translation still runs concurrently. PDFs are previewed by text block: short blocks clearly larger
than the body text, and bold short blocks slightly larger, are shown as subheadings. Records created before the
preview feature need a retranslation before they can be opened. Encrypted PDFs and PDFs without text fail to
translate, and the reader stops at the error message without polling further.

- Side by side on wide screens and stacked on narrow ones, switchable between 对照 / 译文 / 原文 (side by side /
  translation / original). Body text uses a serif font, and headings, lists and quotations have their own
  styles.
- The outline on the left comes from the document's chapter headings. A dot shows each chapter's progress (gray
  not started, blue translating, green done), and a heading shows its translation once translated. On narrow
  screens the outline goes into a drawer.
- The reader follows translation progress by default. Scrolling by hand or clicking the outline pauses
  following, and a 跟随翻译进度 (Follow progress) button appears at the bottom. After jumping somewhere, the
  backend translates what comes after that point first.
- The address bar shows `#read/<job id>`, so a reload stays on the same document; Esc goes back.
- The preview pushes only finished paragraphs and does not rebuild the output file, so it does not slow
  translation down.

### CC Switch

When [CC Switch](https://github.com/farion1231/cc-switch) is installed on the machine, the Translation services
page gets an extra group, CC Switch 当前生效 (Active in CC Switch), which uses the providers currently selected
for Claude, Codex (ChatGPT), Gemini and Grok:

- `~/.cc-switch/cc-switch.db` is opened read-only (another path can be given with the environment variable
  `CC_SWITCH_DB`) and read at each translation; keys are never copied into this project.
- After switching providers in CC Switch, click 刷新 (Refresh) in the list for it to take effect.
- URL and key come from CC Switch; model, concurrency, prompt and the like can be adjusted here.
- Codex uses the OpenAI Responses API by default, and Chat Completions only with `wire_api = "chat"`. Grok Build
  uses the Responses API only with `api_backend = "responses"`, and Chat Completions otherwise. Claude uses
  Bearer authentication with `ANTHROPIC_AUTH_TOKEN`.
- Configurations logged in with an official account (OAuth) have no API key and are shown as unavailable.

### Languages

Both the source and the target language can be chosen. The language table (128 languages, including Classical
Chinese, Cantonese and Northeastern Mandarin) is taken from Immersive Translate.

- The source language defaults to 自动检测 (Detect automatically): the document language is detected from a sample,
  fills `{{from}}` in the prompt, and selects prompts that depend on the source language, such as `wyw2zh-CN`.
- Paragraphs already in the target language are not translated (Chinese quoted in an English book is skipped
  when translating into Chinese). Simplified and Traditional Chinese do not count as the same language.
- When the detected document language equals the target language, the job page shows a warning, like
  sameLangCheck in the extension.
- Google Translate converts language codes with the extension's mapping (zh-HK→zh-TW, fil→tl and so on);
  unsupported languages (such as Classical Chinese) prompt switching to an AI service.
- Translated elements carry a `lang` attribute; right-to-left languages such as Arabic and Hebrew get
  `dir="rtl"` and are right-aligned in PDFs.

### Prompts

The default prompts and the segment protocol are taken from Immersive Translate 1.33.3 (`app/prompts.py`):

- The prompt is chosen by the `<from>2<to>` rules of the extension's langOverrides. Simplified Chinese,
  Traditional Chinese, English, Japanese, Korean, Russian, French, Spanish, Portuguese, Classical Chinese and
  Northeastern Mandarin have their own prompts; other languages use the generic English template.
- For Simplified Chinese, Traditional Chinese, Northeastern Mandarin and Classical-to-vernacular Chinese, the
  prompt asks for chapter numbers with a unit word in headings and tables of contents (Chapter 2, Volume II) to
  be written with Chinese numerals, bare numbers and dotted section numbers (2, 7.4, 1.2.3) to keep Arabic
  digits, and each term to have a single translation. When translating into Chinese, "第2卷" at the start of a
  heading or table of contents entry becomes "第二卷"; a source number such as 7.4 or 12 translated as "七点四"
  or "第十二章" is changed back to the original digits; body text and volume and issue numbers such as
  "第25卷第1期" stay as they are. Old headings already in the cache are rewritten at the next translation.
- Batches of paragraphs are marked with `[[p0]]`, `[[p1]]`… and end with `[[source_end]]`, and the model answers
  with the same markers; missing paragraphs are translated again on their own.
- The book or document title is filled into `{{title_prompt}}` as context.
- Custom prompts can use `{{to}}` (target language), `{{from}}` (source language), `{{text}}` (source text) and
  `{{title_prompt}}`; the segment protocol is appended automatically.

The configuration is saved in `data/services.json` (mode 600), and API keys are never returned to the frontend.

Translations are cached in `data/cache.sqlite3`. The cache key includes the service id, provider, model, prompt,
glossary, source language and target language. With the same configuration, the same text is taken from the
cache; after changing the model, prompt or glossary, the whole document is translated again with the new
configuration, and switching back hits the original cache entries. A job that failed midway continues from where
it stopped when submitted again.

## Web page translation (Chrome extension)

The browser extension in `extension/` translates the web page being viewed through this server: press the shortcut
and each paragraph's translation appears below it. The server must be running.

1. Open `chrome://extensions`, turn on developer mode, click "Load unpacked" and choose the `extension/` directory.
2. Press `Alt+T` (Option+T on macOS) or click the extension's toolbar button. The first press starts translating;
   the next ones hide and show the translations. After a failed request, the next press retries it.
3. The shortcut is changed on Chrome's `chrome://extensions/shortcuts` page (extensions cannot set it themselves).
   The extension's options page shows the current shortcut with a button that opens that page, and sets the server
   address, the translation service and the languages; the services are those enabled on the server, and keys,
   models and prompts are set on the server's page.

- Paragraphs come from the page's layout: inline elements belong to the surrounding paragraph, and a blank line
  (`<br><br>`) starts a new one. Code blocks, form controls, editable content (`contenteditable`) and blocks marked
  `translate="no"` or `notranslate` are skipped, as are paragraphs already in the target language and text visible
  only to screen readers. Marked text inside a sentence (a command name) stays in the translation unchanged. A
  translation goes below its original, except in flex and grid rows and inside inline boxes (tag chips,
  button-like links), where it follows the original on the same line so the layout holds.
- The translation scope (翻译范围) on the options page is the whole page (the default) or only the main content
  (只翻译正文). With only the main content, a page that marks it (`<main>` or `role="main"`) has only that
  translated, without the navigation and sidebars inside it; a page that does not has everything translated except
  its header, footer, navigation and sidebars (a header or footer inside an article or section is part of it). A
  change applies to pages translated from then on.
- Text the page changes later (chat replies streamed in, counters, content swapped by single-page apps) is
  translated again once it has stayed the same for a second, replacing the old translation; a paragraph replaced
  more than ten times (a ticker) is left in the original from then on.
- Translations keep the font of their paragraph but are marked so they can be told from the page's own text: a
  light blue tint, a blue bar on the left (not for translations on the original's line) and their own text color,
  also in translation-only mode. Light or dark colors are chosen from the background really behind each
  translation, the nearest opaque one among its ancestors, so a dark header on a light page or a dark site in a
  light browser gets the dark colors; where the page sets no background, its `color-scheme` and the browser's dark
  mode decide. Text contrast is above 9:1 on white and on dark backgrounds (WCAG asks for 4.5:1).
- The display is either bilingual (the default) or translation only, chosen on the options page or with the
  shortcut "切换双语对照 / 仅显示译文" (Switch bilingual / translation only; no key by default, assign one on
  `chrome://extensions/shortcuts`); translated pages switch at once. In translation-only mode the original of each
  translated paragraph is hidden (its nodes stay in the page with their text emptied), while images and form
  controls stay visible; hiding the translations brings the originals back.
- Only paragraphs within a screen of the viewport are sent; the rest are translated as the page scrolls, and
  content the page adds later (infinite scrolling, single-page apps) is picked up too.
- Links and inline formatting (bold, italic, underline, strikethrough, inline code, sub- and superscript,
  highlighting) are kept in translations. Such a paragraph is sent as HTML with only those tags, each numbered and
  without attributes. The translation is never inserted as HTML: it is parsed in an inert document and rebuilt
  with DOM APIs, every tag that matches one of the paragraph's (by number, else by tag in order) becoming a copy of
  that page element without its `id`, so `href` and other attributes come from the page, never from the model; any
  other tag keeps only its text, and scripts, images and the like are dropped. Paragraphs without formatting are
  sent and inserted as plain text.
- The extension asks for the current tab only when the shortcut is pressed or the button clicked (`activeTab`). At
  install it can reach only `127.0.0.1` and `localhost`; connecting to a server on another machine on the options
  page makes the browser ask for access to that address (the server must listen on an outside address and list it
  in `ALLOWED_HOSTS`; the options page says which is missing). An http address is not encrypted, so page text
  crosses the network in the clear: use one only on a trusted network. Page text is sent to the translation
  service chosen, and only when the shortcut is pressed or on sites chosen for automatic translation (below).
- Web page translations share the translation cache with documents but are not saved as records, and do not build
  a document glossary (one more request per page); the glossary typed into the service still applies.
- The options page shows how many web page translations are cached and clears them (清理网页翻译缓存); pages
  opened afterwards are translated again, and translations already shown stay. Sentences a document also uses are
  kept for the document. Translations of pages from before this option existed were not marked as such and are
  not cleared; the cache drops them when it fills up. The API is `GET` / `DELETE /api/web/cache`.
- The API is `POST /api/web/translate` with `{"texts": [...], "html": [...], "target_lang", "source_lang",
  "alt_lang", "service_id", "title"}`, at most 50 texts of 10,000 characters each; `html` is aligned with `texts` and
  marks HTML fragments, which LLM services get the HTML prompt for and Google Translate gets `format=html` for.
  Texts already in `target_lang` are kept, or translated into `alt_lang` when it is given. It answers
  `{"translations": [...], "langs": [...], "detected_lang"}`, where `null` keeps the original and `langs` gives the
  language of each translation. Like every request that changes something, it is rejected when another web page
  sends it (see Known limitations).
- iframes are translated along with the page when the extension may access their site (frames of the page's own
  site always; others only with access to that site) and follow the shortcut with it; an iframe added later starts
  on the next press. Input placeholders are translated with the page, and tooltips (`title`) when hovered, since
  most are never seen. In bilingual mode an attribute shows both texts (`original / translation`, tooltips on two
  lines); hiding the translations restores it, and a value the page set since is left alone. `alt` and
  `aria-label` are not translated: they are not shown on the page.
- Text in shadow DOM is translated too, closed shadow roots included. In a shadow host laid out as a block, the text
  in its shadow root and the children placed at its slots make paragraphs of their own, and the translation of an
  original placed at a named slot goes to the same slot. An inline host stays in the sentence around it: one that
  shows its children at a slot is read like any inline element, and one that shows only its own content (a relative
  date, a badge) is copied into the translation unchanged; standing alone, its content is translated. The page's
  styles do not reach into shadow roots, so each shadow root holding translations gets a `<style>` with the same
  rules as the page. Custom elements defined after the page loaded are translated on the page's next change.
- Chrome's own pages and the Web Store cannot be translated; the toolbar button shows `×` there.

### PDFs open in the browser

Pressing the shortcut or clicking the extension's button in the browser's own PDF viewer submits that PDF to this
server as a translation job, as an upload on the web page would, and opens its reader in a new tab. The button shows
`…` meanwhile. The target and source languages, the translation service and the display come from the extension's
options page.

- The extension downloads the PDF again from its address, with the browser's login on that site; it gets access to
  the site only when the button is pressed (`activeTab`).
- An address is submitted once per set of options: pressing again opens the earlier job. After its record is
  deleted on the server, or after the browser restarts, pressing again submits it anew.
- These cannot be submitted, and a notice at the bottom right of the page says to save the file and upload it on
  the web page: local files (`file://`), PDFs the page made (`blob:` addresses, answers to a form), downloads that
  turn out not to be a PDF (the site wants a login, the link expired), and files over 200 MB.
- Firefox's viewer cannot show the notice, so there only the button shows `×` on failure; a PDF whose address does
  not end in `.pdf` is not recognized in Firefox.

### Translating selected text

Select some text, right-click it and choose "翻译选中的文字" (translate selected text): the translation appears in
a small bubble below the selection, or above it near the bottom of the window. Esc or a click elsewhere closes it.
The command has no shortcut by default; one can be set on `chrome://extensions/shortcuts`.

- Each line of the selection is translated as a paragraph; lines longer than 10,000 characters are cut at a
  sentence end, and long selections are sent 50 paragraphs per request.
- Paragraphs already in the target language are translated into the language chosen under "选中的文字已是目标语言时译成"
  (when the selection is already in the target language) on the options page, English by default: with Chinese as
  the target, foreign text is translated into Chinese and Chinese text into English.
- Text selected in text boxes works too, also in shadow DOM, except in password and one-time-code fields and other
  fields for secrets, which are never sent. With the shortcut, the selection is taken from the frame the user is in; text in
  iframes the extension may enter works too (with `activeTab`, those from the same origin as the page).

### Sites translated automatically

Right-click an empty part of a page and choose "总是翻译此网站" (always translate this site): the browser asks for
access to that site, and from then on its pages start translating as soon as they load, as if the shortcut had
been pressed; the shortcut still hides and shows the translations. On such a site the entry reads "不再自动翻译此网站"
(stop translating this site automatically). The options page lists the sites, with a button to remove each.

- A site is a hostname, over both http and https; other subdomains are separate sites. Only the top frame is
  translated.
- These are the only sites the extension can access without the shortcut being pressed. Removing a site gives up
  that access, as does revoking it in the browser's settings, which also stops the automatic translation.
- The list is synced with the browser account (`chrome.storage.sync`), but access is granted per browser: on
  another computer the options page marks the site until access is granted there from the same menu entry.

### Firefox

Firefox 128 or later can run the extension too, though it is tested in Chrome only. Firefox needs a different
manifest (its background is an event page listed in `background.scripts`, a key Chrome warns about), so
`extension/` is the Chrome version and the Firefox one is built from it into `dist/firefox/`:

```bash
python3 scripts/build_firefox.py
```

Then open `about:debugging`, choose "This Firefox", click "Load Temporary Add-on…" and pick
`dist/firefox/manifest.json`. Build again after updating the extension. A temporary add-on is removed when Firefox
quits, and a permanent install needs the add-on signed by Mozilla. Shortcuts are changed under "Manage Extension
Shortcuts" in the gear menu of `about:addons`, which the options page's button opens from Firefox 137 on.

The extension's icon is drawn in `extension/icons/icon.svg`. Chrome takes only PNG icons in the manifest, so
after changing it, render the PNGs again with [librsvg](https://gitlab.gnome.org/GNOME/librsvg):

```bash
for s in 16 32 48 128; do rsvg-convert -w $s -h $s extension/icons/icon.svg -o extension/icons/icon-$s.png; done
```

## How each format is handled

| Format | Approach | Output |
| --- | --- | --- |
| Markdown | Split into blocks, skipping code blocks and front matter, keeping the structure of headings, lists, quotations and tables | `.md` |
| EPUB | Find the body files through the OPF spine and translate the innermost block elements, including the table of contents (nav / ncx); inline formatting within paragraphs (bold, italic, links, page anchors) is kept, and other resources are packed back unchanged | `.epub` |
| MOBI / AZW3 | Unpacked first. KF8 files yield their EPUB directly; old MOBI7 files are converted to XHTML and assembled into an EPUB. Then translated as EPUB | `.epub`, plus a copy in the original format when Calibre is installed |
| PDF | Extract text blocks, erase the original text and write the translation in its place; images, vector graphics and bookmarks are kept. See the PDF section below | Translation only: the original layout; bilingual: original and translation side by side |
| Word (.docx) | Translate the paragraphs of the body, tables, text boxes, headers, footers, footnotes and endnotes. Runs whose formatting differs from the rest of the paragraph (bold, italic, links) are sent as tags, so the formatting follows the translated words; images, tabs, breaks, footnote references and fields stay in place. Styles, numbering and the rest of the package are copied unchanged. A bibliography section keeps the original | `.docx`; bilingual: each paragraph followed by its translation, without list numbering |
| HTML (.html, .htm, .xhtml) | Translated like an EPUB chapter: innermost block elements, keeping inline formatting and links; scripts, styles and code blocks are left alone, and the page title is translated too. Read in the encoding the page declares | Same format, in UTF-8; bilingual: each translated block followed by its translation |
| Subtitles (.srt, .vtt, .ass, .ssa) | One paragraph per cue, its lines joined; a cue whose lines all start with a dash (two speakers) is translated line by line. Timings, cue numbers, cue settings, VTT headers and notes, ASS styles, override tags at the start of a line and vector drawings are kept; tags inside the text (VTT classes, ASS overrides mid-line) are dropped from the translation. Legacy encodings such as GBK, Big5 and Windows-1252 are detected | Same format, in UTF-8 with the original line endings; bilingual: original lines with the translation below |

### PDF

Pipeline: text blocks are extracted with PyMuPDF, the original text to be reflowed is erased glyph by glyph
(not by rectangle, which would also remove other text touching the box edge), and the translation is written in
its place. When it does not fit, these steps are tried in order: extending to the right (only for blocks clearly
narrower than the page, without covering text or figures on the right), reducing line spacing, and finally
reducing the font size.

- **Text frame**: the text frame is found from where long body lines start and end across the whole book, with
  odd and even pages counted separately. Short words outside the frame on the same line as body text, smaller
  than the body and aligned along one edge, are margin notes. Margin notes of up to four words that do not form
  a sentence are terms and keep the original; sentence-like margin notes are translated on their own and written
  back in place, with notes on the left right-aligned. Margin notes do not split the body text into pieces.
- **Columns and reading order**: two and three columns are written by column edges, without covering figures
  and vector graphics. When the gutters are clear, columns are read top to bottom one after another, and
  headings across columns and whole tables keep their vertical position. Paragraphs continuing across pages in
  the same column, or from the bottom of the left column to the top of the right, are joined into one
  paragraph, typeset as full lines, and end above the page number area. When a paragraph flows around a figure,
  the caption between the two halves does not break the paragraph, and the translated lines beside the figure
  also keep clear of the figure and its caption.
- **Tables**: in tables with horizontal and vertical rules, tables made of one shaded background per cell
  (including those with only top and bottom rules), and tables of short cells with aligned column edges, the
  cell text keeps the original, while table captions are translated as usual. When paragraphs are joined across
  pages, a table still blocks the bottom of the column, so a table header never joins the start of a paragraph
  on the next page. Pseudocode in an algorithm box is treated as a one-column table, one cell per line,
  translated line by line, written within its cell and never joined into paragraphs. A paragraph of body text is
  never taken for a table.
- **Formulas**: an inline formula is one unit within its paragraph. It becomes a placeholder before translation,
  is cropped from the original page afterwards, and is placed back on the baseline while typesetting. A pasted
  formula carries only its own glyphs, so selecting and copying never picks up the rest of the original page.
  Greek letters and math symbols in the body font, and upright digits and operator names right next to a
  formula, join the same placeholder; ordinals such as `i-th` stay in the text and are translated. Display
  formulas on lines of their own keep the original, and so do piecewise expressions with an extended brace set
  in the body font.
- **Typesetting**: the typesetter breaks lines itself. Chinese lines are justified, and punctuation never
  starts a line. Bold, italic, font size and text color that differ from the body within a block (blue citations
  and cross-references, strings marked in color) are kept, continuing onto the next line when they span lines.
  On pages whose lines share one center line, such as covers and title pages, the translation is centered on
  that line, and a smaller affiliation below an author name gets its own line. Right-to-left languages such as
  Arabic and Hebrew are right-aligned.
- **Kept in the original**: these are not translated: text in figures (vector figures, images, and the area
  from a Figure N caption up to the previous body paragraph, including node labels, tables and pseudocode drawn
  with glyphs; a body sentence starting with "Figure 2.13 shows" is not a caption); blocks almost entirely in a
  monospaced font (code, quoted prompts and dialogues, example sentences; monospace is recognized by font name
  or the PDF monospace flag), and whole shaded areas almost entirely in monospace (code point tables, code
  boxes); passages in a language other than the document's (foreign epigraphs, quotations and examples; short
  entries set in two columns are detected together); word-by-word and morpheme-by-morpheme glosses of
  linguistic examples, and pinyin set syllable by syllable below Chinese characters (the free translation below
  is translated); repeated running headers, footers and page numbers, including headers above the text frame
  that carry page numbers or change with the section. Caption labels in white on black are pasted back as they
  are; when a figure or table is wider than the body and its caption starts in the margin, the label still
  belongs to the caption and is not taken for a margin note. Table of contents entries translate only the
  title: the leader dots and page number at the end are not sent for translation, the page number stays in
  place, and dots are filled from the page number back to the end of the translation, each entry a paragraph of
  its own. Body text and captions on near-white backgrounds are still translated, and captions and table
  captions next to figures are translated on their own; footnotes at the page bottom are translated as usual
  but take no part in joining across pages.
- **Bitmap fonts**: when Type3 fonts produced by old versions of Ghostscript have no Unicode, tesseract
  recognizes the letters before translation. This requires tesseract installed on the machine (with the English
  language pack); without it the job reports an error.
- **Bilingual version**: the original page and the translated page sit side by side on one page.

### PDF layout model (optional)

By default the layout is divided by rules only, and no model is downloaded. To also keep the original for figure
legends, figures without captions and display formulas typeset in non-TeX fonts, connect a YOLO layout model on
the 版面模型 (Layout models) page:

1. Install the runtime with `.venv/bin/pip install onnxruntime` and restart the server. The page shows whether
   the installation worked.
2. Choose a download source (Hugging Face, HF-Mirror, ModelScope) and click 下载模型 (Download model). The file is
   saved to `data/models/` and checked against its checksum after downloading. An existing model file can also
   be used by entering its path.
3. Tick 翻译 PDF 时使用版面模型 (Use the layout model when translating PDFs) and save. 测试模型 (Test model) runs
   the model on a sample page first.

The model is the [ONNX version of DocLayout-YOLO DocStructBench](https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx)
(AGPL-3.0, about 75 MB), used through its class names such as title, figure and isolate_formula. Inference takes
just over a second per page on a CPU. Figures and display formulas boxed by the model keep the original; blocks
the rules took for body text are treated as headings or captions when the model boxes them so. Text in figures
and numbered headings that the rules already recognized are not changed by the model. Without onnxruntime, or
when the file is missing or fails to load, only the rules are used and translation proceeds as usual.

### PDF formula detection model (on by default)

The rules find inline formulas by font. When the body text is set in Times or similar fonts, italic variables
use the same font as italic words in the text, and expressions such as x or P(y = 1) are often split up and
translated. The formula detection model boxes inline formulas in the page image, and the text in each box
becomes one placeholder, pasted back into the translation as it was. It has its own settings below the layout
model on the Layout models page and is on by default:

1. The runtime is the same as above and needs onnxruntime.
2. Choose a download source (Hugging Face, HF-Mirror) and click 下载模型 (Download model); the file is saved to
   `data/models/`. Without the model file only the rules are used.
3. For more speed, change 输入尺寸 (Input size) to 768; to turn the model off, untick
   翻译 PDF 时使用公式检测模型 (Use the formula detection model when translating PDFs). The same page also sets the
   number of pages inferred at once and the inference threads per page, with defaults computed from the CPU
   core count. Save after changing.

The model is [Pix2Text MFD 1.5](https://huggingface.co/breezedeus/pix2text-mfd-1.5) (YOLO11m, about 80 MB),
whose class embedding is an inline formula. The model card states the MIT license, while the file metadata
states AGPL-3.0. The page uses an input side of 1024 by default, which can be changed to 768: layout analysis
gets about 30% faster, and more single-letter variables are missed (97.4% against 99.0% on 36 papers). The model
infers the following pages in background threads while the current page is read: with 4 or more cores, two
pages are inferred at once by default, each with its own session and half the cores, and inference threads do
not spin while waiting, leaving the cores to the thread reading pages. On a 6-core CPU this is about 20% faster
than one page with 4 threads. Page images are padded with gray only to a multiple of 32 rather than to a square,
which for letter-size pages saves over 20% of the pixels and makes both models about 20% faster. Layout analysis
takes about 0.8 seconds more per page at 1024, about 0.4 seconds at 768. The number of pages inferred at once
and the thread count affect only speed, not the analysis result. A comma or period at the end of a box stays in
the text, and boxes holding only digits are not formulas. Blocks the model boxes as display formulas (isolated)
keep the original, so display formulas set in the body font (such as perplexity(T) = P(red red blue)) are no
longer translated into Chinese; explanatory sentences in a box (with three or more function words such as
articles and prepositions, as in (where c is the correct class)) and headings are still translated.
Segmentation and roles still follow the rules and the layout model. On 36 arXiv papers, the share of inline
formulas recognized correctly (the whole formula in a placeholder, with at most digits or units left outside)
rose from 78.4% to 99.0%, and for papers in non-TeX fonts from 67.2% to 98.5%.

### int8 models (used by default)

The int8 versions convert the models' weights and activations to 8-bit integers. Formula detection runs about
twice as fast and the layout model about 30% faster, with analysis results essentially the same as the original
models. The int8 files are generated once on the machine, separately for each model:

1. Install onnx with `.venv/bin/pip install onnx` and restart the server. onnx is needed only for generating,
   not for translating.
2. Once a model is downloaded, click 生成 int8 模型 (Generate int8 model) under its tab on the Layout models
   page. The program samples 64 random pages from the PDFs in the translation records for calibration, or uses
   a generated sample page when no PDF has been translated yet. It takes about three minutes on a 6-core CPU;
   the resulting file of about 20 MB is placed next to the original model with `.int8` added to its name (such
   as `pix2text-mfd-1.5.int8.onnx`). After generating, the box counts on a few pages are compared with the
   original model, and the file is discarded when it finds less than half. Only one model is generated at a
   time.
3. 精度 (Precision) defaults to int8: the int8 file is used when it exists, the original model otherwise.
   Choosing fp32 always uses the original model.

When quantizing, the decoding part of the detection head stays fp32, weights use only 7 bits, and activation
ranges are a moving average over pages; the reasons are given at the top of `app/quantize.py`. On 36 arXiv
papers, calibrated with 17 translated PDFs (textbooks and papers): formula detection recognizes 98.9% of inline
formulas correctly, the same as the original model, with segmentation unchanged and inference per page on a
6-core CPU down from 1.11 to 0.56 seconds; the layout model's segmentation is the same as the original's, 98.6%
of boxes match the original's, and inference per page drops from 1.05 to 0.66 seconds. The evaluation method is
described in `eval/README.md`. After changing the precision, old jobs analyzed again no longer match their box
cache and are inferred again.

### Bibliographies

Bibliography sections (Bibliography / References / Works Cited / Further Reading / Endnotes / 参考文献, including
headings with colons or numbers) keep the original and are not translated; the heading itself is still
translated and marked as skipped in the preview.

- **EPUB**: detected by heading level and `epub:type`, and may continue across spine files. A heading at the same
  or a higher level, or the next chapter in a flat document, ends the section; subheadings do not. A file whose
  base name says it is the bibliography is skipped entirely.
- **PDF**: entered when a heading larger than the body text (or bold) is directly followed by citation entries.
  Citation entries have `[J].` / `[M].` and a year after the title, or are numbered entries with a publisher (the
  year and publisher may be on the next line); years and publishers scattered in body text do not count. The
  section ends at a bold heading close to or still clearly larger than the body size; subheadings such as Books
  or Articles smaller than the bibliography heading stay in the section. A bibliography without a heading (books
  often set it right after the exercises at a chapter end) is detected by its entries: a run of author-year
  entries in the same size starting with "Surname, Initials." keeps the original, including the continuation
  lines in between.
- Skipped PDF entries stay on the original page without being erased and reflowed, so hanging indents and
  hyphenation at line ends are unchanged; a body paragraph identical to a citation is still translated.

## Layout

```
app/
  main.py          FastAPI app: middleware, startup cleanup, and the routes below
  routes/          the API by area: services, glossaries, web, jobs, settings, models
  state.py         stores, running jobs and model downloads shared by the routes (read as state.jobs)
  job_control.py   starting, pausing, resuming, deleting and clearing jobs
  model_files.py   which layout and formula model files translation uses
  history.py       translation records (data/jobs/<id>/job.json) and retention cleanup
  cleanup.py       clearing the caches and layout analysis results a record left behind
  settings.py      application settings (data/settings.json)
  runner.py        deduplication, caching, batching, concurrency
  background.py    threaded work such as reading layout and writing output, stopped when a job is paused
  translators.py   translation engines (Google / OpenAI / Claude / Gemini / Grok / compatible APIs)
  services.py      creating, reading, updating and deleting service configurations
  ccswitch.py      reading the providers active in CC Switch
  prompts.py       default prompts and the segment protocol
  languages.py     language table, prompt override rules, Google language code mapping
  langdetect.py    document language detection
  glossary.py      glossary: finding candidate terms, filtering per batch, CSV import and export
  glossaries.py    the glossary library (data/glossaries.json)
  layout_model.py  status, download and test of the layout and formula detection models
  quantize.py      generating int8 versions of both models
  formats/         parsing and writing back each format (docx.py, html_page.py and subtitles.py for Word, HTML
                   and subtitles)
static/            frontend pages
extension/         Chrome extension for web page translation
scripts/           build_firefox.py: writes the Firefox version of the extension to dist/firefox/
tests/             tests (pytest)
eval/              evaluation scripts for PDF segmentation, inline formulas and models; usage and the paper list
                   are in eval/README.md
```

## Known limitations

- Scanned PDFs that need OCR are not supported yet. Tables without rules and without aligned column edges are not
  recognized, and their text is translated. When gutters are unclear, the reading order may not follow the
  columns. A line break followed by an English capital letter still starts a new paragraph.
- In Word documents, text in inline content controls, comments, charts and SmartArt is not translated. A subtitle
  sentence split over several cues is translated cue by cue. A translated HTML page refers to its images and styles
  by the same relative paths, so it shows them only when saved next to the original's files.
- The server has no authentication and is meant for local use only (bound to 127.0.0.1 by default). Web pages
  open in the same browser can still send requests to it, so requests that change something (POST, PUT, PATCH,
  DELETE) are accepted only from the server's own pages, browser extensions, and tools that send no `Origin`
  header (curl, scripts); any other `Origin` gets 403. Otherwise any site could pause or delete jobs or spend the
  translation quota without being able to read the answer. Reads are not checked: without CORS headers other pages
  cannot read them.

## License

[AGPL-3.0](LICENSE).

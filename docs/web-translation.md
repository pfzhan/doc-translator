# Web page translation design

English | [简体中文](web-translation.zh-CN.md)

At the press of a shortcut, the extension finds the content of the current web page and translates it, inserting
each translation below its original. The shortcut can be changed.

## 1. Kind of client

| Option | Configurable shortcut | Pages behind a login, SPAs | Permissions and security | Decision |
|---|---|---|---|---|
| Chrome extension (MV3) | Yes, with `chrome.commands` | Yes, it reads the DOM the user sees | `activeTab`: access to the current tab only when the shortcut is pressed | Chosen |
| Userscript | Has to be built in the script | Yes | Injected into every site in advance; depends on Tampermonkey | Fallback |
| Server fetches the URL and renders it | No shortcut | No (no login state, content rendered by JS is missed) | The server requests arbitrary URLs (SSRF risk) | Rejected |

## 2. Architecture

```
Shortcut (chrome.commands) or the toolbar button
   │
background.js (service worker)
   │ 1. chrome.scripting injects content.js / content.css (granted by activeTab)
   │ 2. sends translation requests to the local server on the page's behalf
   │    (the extension has host_permissions, so the page's CORS does not apply)
   ▼
content.js (in the page)
   finds paragraphs → translates those in view first → continues on scroll → inserts translations below originals
   │  chrome.runtime.sendMessage
   ▼
POST http://127.0.0.1:8765/api/web/translate (app/routes/web.py)
   → Runner.translate_all (cache, same-language skipping, punctuation normalization) → the existing Translator
```

All requests go out from the background. A content script sending fetch itself is subject to the page's CSP, CORS
and Chrome's Private Network Access, and is blocked on many sites.

## 3. Shortcuts

- `manifest.json` declares the command `toggle-translate`, `Alt+T` by default (Option+T on a Mac).
  - First press: start translating.
  - Later presses: hide / show the translations.
  - Press after an error: retry the paragraphs that failed.
- Errors come in two kinds:
  - One request fails (the translation service reports an error or times out): the paragraphs concerned show
    翻译失败 (Translation failed) and a 重试 (Retry) button, and the other paragraphs go on.
  - An error every request would get (the local server cannot be reached, the access token is wrong, the
    translation service does not exist or is disabled): no more requests are sent.
  - In both cases, pressing the shortcut again retries all the failed paragraphs.
- Command `toggle-mode`: switches between bilingual and translation-only display; no key is assigned by default.
- Clicking the extension's toolbar button does the same as the shortcut.
- Shortcuts are changed on Chrome's `chrome://extensions/shortcuts` page. Shortcuts set there work whatever has the
  focus, and conflicts with keys the browser reserves are handled. An extension cannot set a shortcut from code, so
  the options page shows the current shortcuts with `chrome.commands.getAll()` and has a 修改快捷键 (Change
  shortcuts) button that opens that page.
- No custom shortcut from keydown listeners in the page: that would need the "all sites" permission and a script
  injected everywhere; it does not work while the focus is in the address bar or an iframe; and it conflicts with
  the page's own shortcuts.

## 4. Finding the page's content (content.js)

1. **Paragraphs**: a walk down from `body`.
   - Known inline tags (`a, span, em, strong, code, sub, sup` and so on) join the paragraph they are in;
   - other tags and custom elements (a `-` in the tag name) are blocks, each its own paragraph;
   - an inline tag whose computed `display` is not inline is a block as well;
   - an inline element holding block elements (common in card-like links) is walked into.
2. **Skipped**:
   - `script, style, noscript, template, pre, textarea, input, select, button, svg, math, canvas, iframe, video,
     audio, img`;
   - `[translate="no"]`, `.notranslate`, `[contenteditable]`, `[aria-hidden="true"]`, `[hidden]`;
   - text with fewer than 2 letters, and bare URLs;
   - hidden inline elements (such as the brackets around "[edit]" next to a Wikipedia heading): ignored, without
     cutting the sentence around them in two;
   - text meant only for screen readers (a box of at most 1px when it enters the viewport);
   - the translation nodes the extension inserted itself.
3. **Viewport first**: `IntersectionObserver` translates only the paragraphs that enter the viewport (with one
   screen of margin above and below). Up to 20 paragraphs or 3000 characters go in one request, with at most 2
   requests at a time. Content with `display:none` never enters the viewport, so it is translated once it shows.
4. **Dynamic content**: `MutationObserver` finds added nodes (infinite scrolling, SPA route changes), which go through
   the same steps, debounced by 300ms. Paragraphs are deduplicated by their first node, so nothing is translated
   twice; where a translation already exists at the same place it is replaced, so refreshed content does not end up
   with several translations.
5. **Inserting translations**: a `<dt-trans lang="target language">` custom element (`display:block`) goes inside
   or after the paragraph. It leaves the original alone, does not break the layout of `li/td/h*`, and page styles
   rarely match it. Inside a flex or grid container or an inline box (GitHub's topic tags, for example) the
   translation becomes inline, right after the original, or it would be squeezed into a column beside it.
6. **How translations are written**: model output is never inserted into the page as HTML.
   - Paragraphs without formatting: plain text written with `textContent`.
   - Paragraphs with links, bold and other formatting: only these tags are kept, each numbered `i="N"` and stripped
     of all attributes, and sent as HTML. The translation is parsed in a separate document made by `DOMParser` (no
     scripts run, no resources load) and rebuilt with DOM calls. Tags that match the original (by number, else by
     tag name in order) become copies of the page's own elements (without `id` and `name`), so `href` and the other
     attributes come from the page, not from the model. Other tags keep only their text; scripts, images and the
     like are dropped. Matching by number is needed: Google Translate and LLMs both reorder tags.
7. **Colors**: translations have a light blue background, a line on the left and a text color of their own, kept in
   translation-only mode, so that changed text stands out.
   - Light or dark colors follow the background actually behind each translation: the nearest opaque background up
     the tree, by its luminance. Whether the browser is in dark mode says nothing about sites with their own colors,
     and a light page can have a dark header.
   - When the page sets no background at all, `color-scheme` and `prefers-color-scheme` decide.
   - Text contrast is above 9:1 on white and on dark backgrounds.
8. **Translation only**: the original's text nodes are emptied in place and the original elements are marked and
   hidden with CSS. Nodes are neither removed nor wrapped, so frameworks such as React still find them when they
   later update or remove them. Images and form controls stay visible.
9. **iframes and attributes**:
   - **iframes**: content.js is injected into every frame the extension can access. The top frame owns the status
     box and the shortcut state; iframes follow the top frame's state (`dt-sync`).
   - **`placeholder`**: translated with the page.
   - **`title`**: translated only after the pointer rests on the element for 300ms, as most tooltips are never seen.
   - **Bilingual display**: attributes show both texts. A value the page changes later is not overwritten.
10. **Status**: the status box at the bottom right lives in a shadow DOM, out of reach of the page's styles. It
    shows progress, hiding and showing, and errors.

## 5. Server endpoints

The web translation endpoints are in `app/routes/web.py`; the origin check is in `app/main.py`.

`POST /api/web/translate`:

```python
class WebTranslateRequest(BaseModel):
    texts: list[str]  # at most 50, each at most 10000 characters
    html: list[bool] = []  # aligned with texts: which are HTML fragments (the html argument of Runner)
    target_lang: str = "zh-CN"
    source_lang: str = "auto"
    service_id: str = ""  # empty: the default service
    title: str = ""  # page title, context for LLMs; at most 1000 characters, cut to 200 for the model
    alt_lang: str = ""  # texts already in target_lang go into this language (selected text); empty: kept


class WebTranslateResponse(BaseModel):
    translations: list[str | None]  # aligned with texts; None keeps the original (blank, already in the target
    # language, or translated unchanged)
    langs: list[str | None]  # aligned with texts: the language of each translation (target_lang or alt_lang);
    # None where the original is kept
    detected_lang: str
```

- **`alt_lang`**: after the first pass, the paragraphs Runner found already in the target language are translated
  again by a new Translator for `alt_lang`, which detects the source language anew.
  - That Translator is created only when some paragraph needs it, so when the service does not support `alt_lang`,
    only requests that need it fail.

Other endpoints:

- **`POST /api/web/lookup`**: looks a word up.
  - Request: the selected word (at most 100 characters), the sentence it is in (at most 2000 characters),
    `target_lang`, `source_lang`, `alt_lang`, `service_id`, `title`.
  - Answer: one dictionary entry, `WordEntry`: the dictionary form, pronunciation, senses, the meaning in this
    sentence, and the language the entry is written in.
  - Only LLM services write entries; the others answer 501, and the extension translates the selection instead.
  - Not cached: the sense depends on the sentence, and one lookup is a single short request.
- **`GET` / `DELETE /api/web/cache`**: counts and clears the cached translations of web pages. Answers
  `{"entries": N}`.

- **What is reused**: `Runner(translator).translate_all(texts)` does the work.
  - The cache, keyed by `translator.cache_key`, is shared across pages, so repeated content such as navigation bars
    and footers hits it;
  - same-language skipping and the normalization of punctuation and numbering are those of document translation.
- **Glossary**: no automatic glossary, which costs one more request per page, too slow for a page being read.
  Glossaries the user wrote still apply.
- **Translator**: a new one per request. A Translator holds per-request state such as `detected_source` and
  `notes`, which requests sharing an instance would overwrite.
- **Records and cache**: web translations do not appear on the records page, which is built around file jobs.
  - Cache entries of web pages are recorded under one owner, `WEB_CACHE_OWNER`, so they can be cleared on their own
    without touching documents' entries. Sentences a document also uses stay for that document after clearing.
  - When the cache is full, the oldest entries no document uses go first, web page translations among them.
    Browsing does not push a book's translations out of the cache.
- **Errors**: a problem with the service configuration is a 400, an error of the translation service a 502. `detail`
  holds a Chinese message that the extension shows to the user as it is.
- **Origin check**: done by a middleware for every request that changes data (POST/PUT/PATCH/DELETE).
  - Three kinds of request pass: this server's own pages, browser extensions, and requests without an Origin
    (command-line tools). Everything else gets a 403.
  - Why: any web page can POST across origins to 127.0.0.1 with a form or `fetch(..., {mode: "no-cors"})`, which
    triggers no preflight. The page cannot read the answer, but it can pause or delete jobs, or spend the
    translation quota. Browsers always send Origin on such cross-origin requests, so checking it is enough.
  - Only the host name and port are compared, not the scheme, so this server's own pages pass behind an HTTPS
    reverse proxy too.
  - The web translation endpoint also limits the request body (4 MB) and the length of `title`.
- **CORS**: no `CORSMiddleware`; the existing `TrustedHostMiddleware` stays.
- **Data for the options page**: `GET /api/services` and `GET /api/languages` already exist and are called directly.

## 6. Extension settings

Settings are defined in `extension/settings.js`, shared by the background script and the options page. All but the
access token are kept in `chrome.storage.sync`.

- Server address: `http://127.0.0.1:8765` by default.
  - Once installed, the extension can reach only `127.0.0.1` and `localhost`.
  - Other addresses fall under `optional_host_permissions`: access is requested when the user clicks 连接
    (Connect). The request must be made before any await, or it no longer counts as a user action and the browser
    refuses it.
  - When the server answers 400, the page says to add the address to `ALLOWED_HOSTS`.
- Access token: for a server on another machine, the value of the server's `ACCESS_TOKEN`, sent as
  `Authorization: Bearer`.
  - Kept in `chrome.storage.local`, so it does not sync to the account's other browsers.
- Display: bilingual (default) or translation only. The command `toggle-mode` switches too; no key is assigned by
  default.
- Scope: the whole page (default) or the main content only; see phase 4 in section 8.
- Translation service: the server's default service by default. When the chosen service has been deleted or
  disabled, the options page switches back to the default and says so.
- Target and source language: the lists come from the local server.
- The language selected text already in the target language is translated into (`altLang`): English by default.
- Web page cache: shows how many entries there are, and clears them in one click.
- The current shortcuts and a 修改快捷键 (Change shortcuts) button.
- The start command, when the local server cannot be reached.

## 7. Files and technology

```
extension/
  manifest.json      # MV3: permissions [activeTab, scripting, storage, contextMenus],
                     #      host_permissions [127.0.0.1, localhost], optional_host_permissions [http, https]
  background.js      # shortcuts, toolbar button, context menus, injection, requests, PDFs open in the browser
  settings.js        # settings and their defaults, shared by the background script and the options page
  content.js / content.css
  selection.js       # the bubble of selection translation and word lookup
  sites.js           # automatic translation per site: permission requests, registerContentScripts
  autostart.js       # runs before content.js on sites translated automatically, so it starts as if the shortcut
                     # had been pressed
  options.html / options.js
scripts/build_firefox.py          # writes the Firefox version to dist/firefox/
tests/test_web.py                 # the endpoints with a mock service: alignment, skipping, alt_lang, cache and
                                  # clearing, limits, unknown services, errors
tests/test_origin.py              # the origin check
tests/test_extension_browser.py   # the extension loaded in a real browser, against a stand-in server
```

The front end is plain JavaScript, written like `static/`, with no build step.

## 8. Phases

- **Phase 1 (done)**:
  - the server endpoint;
  - the extension's shortcut and toolbar button injection, translation of what is in view, bilingual plain text,
    hiding and showing on the next press, retrying after errors;
  - MutationObserver for dynamic content;
  - the options page.
- **Phase 2 (done)**:
  - Inline formatting kept: rather than reusing the server's `_sanitize_translation`, the front end rebuilds it with
    DOM calls (see 4.6). HTML cleaned on the server would have to be parsed again in the page; rebuilding it there is
    simpler and safer.
  - Translation-only mode: the command `toggle-mode`.
  - iframes.
  - Translation of the `placeholder` and `title` attributes. `alt` is not translated: it is not shown on the page,
    so translating it would only spend quota.
  - Server addresses other than this machine: `optional_host_permissions`.
  - Translation colors: translations told apart from originals, light or dark by the background.
- **Phase 3 (done)**:
  - Selection translation: the context menu or the command `translate-selection`. The translation shows in a bubble
    next to the selection, in a closed shadow DOM, written with `textContent`.
  - Automatic translation per site: access to the site is requested when the user picks the context menu item, then
    `registerContentScripts` registers the script, for the top frame only. The list of sites is in `storage.sync`,
    while access is granted per browser, so each browser registers the script only for the sites it was granted.
  - Firefox: `extension/` is the Chrome version; `scripts/build_firefox.py` writes the Firefox version to
    `dist/firefox/`, with `background.scripts` in the manifest, a `gecko.id` and a minimum version of 128. Chrome
    warns when both forms are in one manifest, hence two. `web-ext lint` reports no errors, but it has not been run
    in Firefox.
- **Phase 4 (done)**:
  - Scope: the whole page, or the main content only. A page with `<main>` has only its content translated, without
    the navigation and sidebars inside it; a page without `<main>` has everything translated except the header,
    footer, navigation and sidebars.
  - The origin check on every endpoint (see 5).
  - Fixes of what code review found: text the page changes later is translated again once it has stayed the same
    for a second, and no more after ten replacements at the same place; inline notranslate cuts the sentence and is
    kept as it is; memory no longer grows without bound; newer color syntaxes such as oklch; password fields are
    never read; a malformed or late answer from the server no longer leaves the page waiting.
- **Phase 5 (done)**:
  - Web page cache: translations are recorded under `WEB_CACHE_OWNER`; the options page counts and clears them; a
    full cache drops web page translations first (see 5).
  - PDFs open in the browser: rather than injecting into the page, the extension downloads the PDF again from its
    address (with the site's login state), submits it to the server as a translation job, and opens the job's
    reader in a new tab. The same address with the same settings is submitted only once.
  - Shadow DOM: text in open and closed shadow roots is translated, and slotted children are paragraphs of their
    own. Inline hosts stay in the sentence around them; custom elements defined after the page loaded are
    translated at the page's next change.
  - Access token: the extension sends the server's `ACCESS_TOKEN` for a server on another machine (see 6).
  - Services that no longer exist: the options page switches back to the default service and says so.
  - Collapsed panels are translated once they open.
  - Text in a sentence meant only for screen readers is skipped; rendered formulas (KaTeX, MathJax, MathML) are sent
    as their TeX source, and the translation gets the formula copied from the page. Inline code (`code`, `kbd`,
    `samp`, `var`) is kept as it is too.
  - Paragraphs in view are sent first, then the rest by distance; paragraphs scrolled more than two screens away
    before their turn are held back.
  - When one request fails, its paragraphs show a 重试 (Retry) button and the rest go on (see 3).
  - Truncation lifted: boxes with a line clamp (`line-clamp`), or a fixed height with the overflow hidden, are
    looked for up to five levels up and lifted only as far as the translation needs; the original styles come back
    when translations are hidden.
  - Selected text already in the target language is translated into a second language (`alt_lang`, English by
    default).
  - Word lookup: a selected word or short phrase gets a dictionary entry written by the LLM service instead of a
    translation (`/api/web/lookup`).

## 9. Risks and known limits

- **Privacy**: page content is sent to the configured translation service (OpenAI, for example). Pages are
  translated only after the user presses the shortcut, or on sites the user set to be translated automatically
  (see phase 3).
- **Misdetection**: complex pages (deeply nested divs, navigation items laid out with flex) may be split wrongly, or
  show translations beside their originals. Phase 1 aimed at articles first.
- **Truncation**: truncation by `line-clamp` and fixed heights is lifted (see phase 5), but only up to five levels
  up; a box further out can still cut a translation off.
- **LLM latency**: the first screen usually takes 2 to 5 seconds, eased by translating what is in view first in
  small concurrent batches. Google Translate is the fastest.
- **Pages that cannot be injected**: `chrome://` pages and the Chrome Web Store. Pressing the shortcut there shows
  `×` on the toolbar button. The browser's built-in PDF viewer cannot be injected either, so PDFs are submitted as
  translation jobs instead (see phase 5).

// Service worker: handles the shortcut and the toolbar button, injects the content script, and sends translation
// requests to the local server. Requests go from here rather than from the page, where the page's CSP, CORS and
// Private Network Access would block them.

// Firefox runs this as an event page without importScripts, and loads the shared files itself (background.scripts in
// the manifest written by scripts/build_firefox.py)
if (typeof importScripts === 'function') importScripts('settings.js', 'sites.js');

/**
 * @param {number} tabId
 * @param {string} text
 */
async function flashBadge(tabId, text) {
  await chrome.action.setBadgeText({ tabId, text });
  setTimeout(() => chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {}), 3000);
}

/**
 * Send a message to the content script in the top frame of tab, injecting the script first into every frame that
 * has none yet (frames of other sites need host access; without it they are skipped). The iframes then follow the
 * state the top frame reports.
 * @param {chrome.tabs.Tab | undefined} tab
 * @param {ControlMessage} message
 */
async function sendToTab(tab, message) {
  const tabId = tab?.id;
  if (!tabId) return;
  /** @type {PageState | undefined} */
  let state;
  try {
    const frames = await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      func: () => !!window.__docTranslator,
    });
    const missing = frames.filter((f) => !f.result).map((f) => f.frameId);
    // The top frame first and on its own: an iframe that went away meanwhile (rotating ads) fails only itself
    if (missing.includes(0)) await inject(tabId, 0);
    await Promise.allSettled(missing.filter((id) => id !== 0).map((id) => inject(tabId, id)));
    // The content script answers every ControlMessage with its PageState (handle in content.js)
    const top = /** @type {PageState} */ (await chrome.tabs.sendMessage(tabId, message, { frameId: 0 }));
    state = top;
    const subframes = frames.map((f) => f.frameId).filter((id) => id !== 0);
    /** @type {ControlMessage} */
    const sync = { type: 'dt-sync', ...top };
    await Promise.allSettled(subframes.map((frameId) => chrome.tabs.sendMessage(tabId, sync, { frameId })));
  } catch {
    // chrome:// pages and the Web Store cannot be injected
    if (!state) await flashBadge(tabId, '×');
  }
}

/**
 * @param {number} tabId
 * @param {number} frameId
 */
async function inject(tabId, frameId) {
  const target = { tabId, frameIds: [frameId] };
  await chrome.scripting.insertCSS({ target, files: ['content.css'] });
  await chrome.scripting.executeScript({ target, files: ['content.js'] });
}

/** @param {chrome.tabs.Tab | undefined} tab */
async function toggle(tab) {
  if (tab && (await showsPdf(tab))) await translatePdf(tab);
  else await sendToTab(tab, { type: 'dt-toggle' });
}

/** @param {chrome.tabs.Tab | undefined} tab */
async function toggleMode(tab) {
  const { mode } = await loadSettings();
  // Pages already translated follow the change through storage.onChanged
  await chrome.storage.sync.set({ mode: mode === 'translated' ? 'bilingual' : 'translated' });
  await sendToTab(tab, { type: 'dt-start' });
}

chrome.action.onClicked.addListener(toggle);

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (!tab) [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (command === 'toggle-translate') await toggle(tab);
  else if (command === 'toggle-mode') await toggleMode(tab);
});

// ---------- PDFs ----------

// A PDF open in the browser's viewer is translated as a document: the service worker downloads it, submits it to the
// server as a translation job and opens the job's reader in a new tab. The button or shortcut gives it access to the
// tab's site (activeTab), so the download carries the user's cookies.

const MAX_PDF = 200 * 1024 * 1024; // the server's upload limit
const SAVE_AND_UPLOAD = '请保存后在 Doc Translator 网页上传';
/** @type {Set<string>} */
const pdfBusy = new Set(); // addresses being downloaded or submitted: pressing again meanwhile does nothing

/**
 * Chrome's viewer page takes scripts and reports the PDF's type; Firefox's takes none, so the address decides.
 * @param {chrome.tabs.Tab} tab
 */
async function showsPdf(tab) {
  if (!tab.id) return false;
  try {
    const [frame] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.contentType,
    });
    return frame?.result === 'application/pdf';
  } catch {
    const url = tab.url || '';
    return /^https?:/.test(url) && /\.pdf$/i.test(new URL(url).pathname);
  }
}

/** @param {chrome.tabs.Tab} tab */
async function translatePdf(tab) {
  const { id: tabId, url = '' } = tab;
  if (tabId === undefined || pdfBusy.has(url)) return;
  pdfBusy.add(url);
  await chrome.action.setBadgeText({ tabId, text: '…' });
  let error = '';
  try {
    const s = await loadSettings();
    // The same file with the same settings opens the job submitted before, while the server still has it
    const key = `pdf ${JSON.stringify([s.server, url, s.targetLang, s.sourceLang, s.serviceId, s.mode])}`;
    const known = field(await chrome.storage.session.get(key), key);
    const job = typeof known === 'string' && (await fetchJob(s, known));
    const id = typeof known === 'string' && job && job.ok ? known : await submitPdf(s, await downloadPdf(url));
    await chrome.storage.session.set({ [key]: id });
    await chrome.tabs.create({ url: `${s.server}/#read/${id}`, index: tab.index + 1, openerTabId: tabId });
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    pdfBusy.delete(url);
  }
  await chrome.action.setBadgeText({ tabId, text: '' });
  if (error) {
    await notice(tabId, error);
    await flashBadge(tabId, '×');
  }
}

/**
 * The job submitted before, or null when the server cannot be reached.
 * @param {Settings} s
 * @param {string} id
 */
function fetchJob(s, id) {
  return fetch(`${s.server}/api/jobs/${id}`, { headers: authHeaders(s) }).catch(() => null);
}

/**
 * @param {string} url
 * @returns {Promise<{ blob: Blob, name: string }>}
 */
async function downloadPdf(url) {
  // A PDF from the user's disk, or one made by the page (blob:, the answer to a form), cannot be fetched again here
  if (!/^https?:/.test(url)) throw new Error(`这个 PDF 无法直接提交，${SAVE_AND_UPLOAD}`);
  let resp;
  try {
    resp = await fetch(url, { credentials: 'include' });
  } catch {
    throw new Error(`PDF 下载失败，${SAVE_AND_UPLOAD}`);
  }
  if (!resp.ok) throw new Error(`PDF 下载失败（HTTP ${resp.status}），${SAVE_AND_UPLOAD}`);
  const tooLarge = '文件太大（上限 200MB）';
  if (Number(resp.headers.get('content-length')) > MAX_PDF) throw new Error(tooLarge);
  const blob = await resp.blob();
  if (blob.size > MAX_PDF) throw new Error(tooLarge);
  // A site that wants a login answers the second request with a page instead. The header may follow some junk
  const head = new TextDecoder('latin1').decode(await blob.slice(0, 1024).arrayBuffer());
  if (!head.includes('%PDF-')) throw new Error(`下载到的不是 PDF，网站可能要求登录，${SAVE_AND_UPLOAD}`);
  return { blob, name: pdfName(url, resp.headers.get('content-disposition')) };
}

/**
 * The file name the site gives, else the last part of the address. The server takes files by their suffix, and
 * addresses such as /pdf/2401.01234 have none.
 * @param {string} url
 * @param {string | null} disposition the Content-Disposition header
 */
function pdfName(url, disposition) {
  const given = disposition?.match(/filename\*=(?:UTF-8'')?"?([^";]+)/i) || disposition?.match(/filename="?([^";]+)/i);
  let name = given ? given[1] : new URL(url).pathname.split('/').pop() || '';
  try {
    name = decodeURIComponent(name);
  } catch {
    // not percent-encoded
  }
  name = name.trim() || 'document';
  return /\.pdf$/i.test(name) ? name : `${name}.pdf`;
}

/**
 * @param {Settings} s
 * @param {{ blob: Blob, name: string }} file
 * @returns {Promise<string>} the id of the new job
 */
async function submitPdf(s, { blob, name }) {
  const form = new FormData();
  form.append('file', blob, name);
  form.append('target_lang', s.targetLang);
  form.append('source_lang', s.sourceLang);
  form.append('service_id', s.serviceId);
  form.append('mode', s.mode);
  let resp;
  try {
    resp = await fetch(`${s.server}/api/jobs`, { method: 'POST', body: form, headers: authHeaders(s) });
  } catch {
    throw new Error(unreachable(s.server));
  }
  /** @type {unknown} */
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new Error(errorText(body, resp.status));
  const id = field(body, 'id');
  if (typeof id !== 'string') {
    throw new Error(`${s.server} 的回应不是 Doc Translator 的翻译任务，请检查扩展设置里的服务地址`);
  }
  return id;
}

/**
 * A message at the corner of the tab. Firefox's viewer takes no scripts: there only the badge shows.
 * @param {number} tabId
 * @param {string} text
 */
async function notice(tabId, text) {
  await chrome.scripting.executeScript({ target: { tabId }, func: showNotice, args: [text] }).catch(() => {});
}

/**
 * Runs in the page; looks like the status box of content.js.
 * @param {string} text
 */
function showNotice(text) {
  const host = document.createElement('dt-toast');
  const shadow = host.attachShadow({ mode: 'closed' });
  shadow.innerHTML = `<style>
    div { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; max-width: 360px;
      padding: 8px 12px; border-radius: 8px; background: rgba(32, 33, 36, .92); color: #fff;
      font: 13px/1.5 system-ui, sans-serif; box-shadow: 0 2px 8px rgba(0, 0, 0, .3); }
  </style><div role="status"></div>`;
  const box = shadow.querySelector('div');
  if (box) box.textContent = text;
  document.documentElement.append(host);
  setTimeout(() => host.remove(), 6000);
}

// ---------- Selected text ----------

/**
 * frameId: the frame holding the selection (the context menu says which); without it (the shortcut) the frame is
 * the one the user is in, or else the first that has a selection. Other frames may still hold an old selection.
 * @param {chrome.tabs.Tab | undefined} tab
 * @param {number} [frameId]
 */
async function translateSelection(tab, frameId) {
  const tabId = tab?.id;
  if (!tabId) return;
  try {
    if (frameId === undefined) frameId = await selectionFrame(tabId);
    if (frameId === undefined) return;
    await chrome.scripting.executeScript({ target: { tabId, frameIds: [frameId] }, files: ['selection.js'] });
  } catch {
    await flashBadge(tabId, '×');
  }
}

/** @param {number} tabId */
async function selectionFrame(tabId) {
  const frames = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
    func: () => {
      // The focused element, also in a shadow root (see focusedElement in selection.js). Page elements are told
      // apart by localName: this runs in the extension's isolated world, not the page's
      /** @param {Element} el */
      const shadowOf = (el) =>
        el.shadowRoot ||
        chrome.dom?.openOrClosedShadowRoot?.(/** @type {HTMLElement} */ (el)) ||
        el.openOrClosedShadowRoot;
      let field = document.activeElement;
      for (let inner = field && shadowOf(field)?.activeElement; inner; inner = shadowOf(inner)?.activeElement) {
        field = inner;
      }
      const editable =
        field && ['input', 'textarea'].includes(field.localName)
          ? /** @type {HTMLInputElement | HTMLTextAreaElement} */ (field)
          : null;
      const inField = (editable?.selectionEnd ?? 0) > (editable?.selectionStart ?? 0);
      return { focused: document.hasFocus(), selected: inField || !!getSelection()?.toString().trim() };
    },
  });
  const withSelection = frames.filter((f) => f.result?.selected);
  return (withSelection.find((f) => f.result?.focused) || withSelection[0])?.frameId;
}

chrome.runtime.onInstalled.addListener(() => {
  // Menus persist across updates: creating the same id again would fail
  chrome.contextMenus.remove('dt-selection', () => {
    void chrome.runtime.lastError; // not there on a fresh install
    chrome.contextMenus.create({ id: 'dt-selection', title: '翻译选中的文字', contexts: ['selection'] });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'dt-selection') translateSelection(tab, info.frameId);
});

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== 'translate-selection') return;
  if (!tab) [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  await translateSelection(tab);
});

// Sites translated automatically (sites.js)
listenForSites();

/**
 * A property of a parsed JSON value, or undefined when it is not an object. For answers whose shape is not trusted.
 * @param {unknown} value
 * @param {string} key
 * @returns {unknown}
 */
function field(value, key) {
  return typeof value === 'object' && value !== null ? /** @type {Record<string, unknown>} */ (value)[key] : undefined;
}

/**
 * The server's error: FastAPI answers {detail: string}, or a list of {msg} for invalid requests.
 * @param {unknown} body
 * @param {number} status
 */
function errorText(body, status) {
  const detail = field(body, 'detail');
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => field(d, 'msg')).join('; ');
  return `HTTP ${status}`;
}

// Also the error when access to a server on another machine was not granted (see options.js)
/** @param {string} server */
const unreachable = (server) => `连不上翻译服务（${server}），请先启动 Doc Translator，或在扩展的设置页重新连接`;

// A request the server has not answered by then has failed; LLM services take tens of seconds for a full batch
const REQUEST_TIMEOUT = 120000;
// Answers that no other request would get past either: an invalid request or a service that does not exist or is
// not enabled (400), a missing or wrong access token (401), a page or address the server does not accept (403), an
// address that is not this server (404). A failure of the translation service itself (502) or a timeout may pass
// on the next request
const SETTINGS_ERRORS = new Set([400, 401, 403, 404]);
// The server rejects a longer page title, and uses only the start of one as context
const MAX_TITLE = 1000;

/**
 * html: aligned with texts, true where the text is an HTML fragment (see richOf in content.js); empty means none.
 * selected: the texts are selected text, and those already in the target language go into the second language
 * (altLang) instead of being kept. langs in the answer is aligned with texts and gives the language of each
 * translation.
 * @param {string[]} texts
 * @param {string} title
 * @param {boolean[]} [html]
 * @param {boolean} [selected]
 * @returns {Promise<TranslateAnswer>}
 */
async function translate(texts, title, html = [], selected = false) {
  const s = await loadSettings();
  let resp;
  try {
    resp = await fetch(`${s.server}/api/web/translate`, {
      method: 'POST',
      headers: authHeaders(s, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        texts,
        html,
        title: title.slice(0, MAX_TITLE),
        target_lang: s.targetLang,
        source_lang: s.sourceLang,
        alt_lang: selected ? s.altLang : '',
        service_id: s.serviceId,
      }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT),
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') return { error: `翻译服务 ${REQUEST_TIMEOUT / 1000} 秒内没有回应` };
    return { error: unreachable(s.server), fatal: true };
  }
  /** @type {unknown} */
  const body = await resp.json().catch(() => null);
  if (!resp.ok) return { error: errorText(body, resp.status), fatal: SETTINGS_ERRORS.has(resp.status) };
  // Another program on the configured port, or a proxy page, answers 200 with something else
  /**
   * @param {unknown} list
   * @returns {list is (string | null)[]}
   */
  const aligned = (list) =>
    Array.isArray(list) && list.length === texts.length && list.every((t) => t === null || typeof t === 'string');
  const notOurs = {
    error: `${s.server} 的回应不是 Doc Translator 的翻译结果，请检查扩展设置里的服务地址`,
    fatal: true,
  };
  const list = field(body, 'translations');
  if (!aligned(list)) return notOurs;
  // Servers from before alt_lang leave langs out and translate everything into the target language
  const langs = field(body, 'langs') ?? list.map((t) => (t === null ? null : s.targetLang));
  if (!aligned(langs)) return notOurs;
  return { translations: list, lang: s.targetLang, langs };
}

// The lookup is one short request; past this the selection is translated instead (see selection.js)
const LOOKUP_TIMEOUT = 60000;

/**
 * A dictionary entry for a selected word or short phrase. Any failure, including a server from before lookups
 * (404), is given as an error: selection.js then translates the selection as before.
 * @param {string} text
 * @param {string} context the sentence the text was selected from
 * @param {string} title
 * @returns {Promise<LookupAnswer>}
 */
async function lookUp(text, context, title) {
  const s = await loadSettings();
  let resp;
  try {
    resp = await fetch(`${s.server}/api/web/lookup`, {
      method: 'POST',
      headers: authHeaders(s, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        text,
        context,
        title: title.slice(0, MAX_TITLE),
        target_lang: s.targetLang,
        source_lang: s.sourceLang,
        service_id: s.serviceId,
      }),
      signal: AbortSignal.timeout(LOOKUP_TIMEOUT),
    });
  } catch {
    return { error: unreachable(s.server) };
  }
  /** @type {unknown} */
  const body = await resp.json().catch(() => null);
  if (!resp.ok) return { error: errorText(body, resp.status) };
  /** @param {unknown} value */
  const isText = (value) => typeof value === 'string';
  const senses = field(body, 'senses');
  const entry = {
    headword: field(body, 'headword'),
    phonetic: field(body, 'phonetic'),
    senses,
    in_context: field(body, 'in_context'),
  };
  const valid =
    isText(entry.headword) &&
    isText(entry.phonetic) &&
    isText(entry.in_context) &&
    Array.isArray(senses) &&
    senses.every((sense) => isText(field(sense, 'pos')) && isText(field(sense, 'meaning')));
  if (!valid) return { error: `${s.server} 的回应不是 Doc Translator 的释义` };
  return { entry: /** @type {WordEntry} */ (entry), lang: s.targetLang };
}

chrome.runtime.onMessage.addListener((/** @type {unknown} */ msg, _sender, sendResponse) => {
  const type = field(msg, 'type');
  if (type === 'dt-lookup') {
    const { text, context, title } = /** @type {LookupMessage} */ (msg);
    lookUp(text, context, title)
      .catch((e) => ({ error: e instanceof Error ? e.message : String(e) }))
      .then(sendResponse);
    return true;
  }
  if (type === 'dt-css') {
    // For the shadow roots content.js enters, which content.css injected into the page does not reach
    fetch(chrome.runtime.getURL('content.css'))
      .then((resp) => resp.text())
      .then(sendResponse, () => sendResponse(''));
    return true;
  }
  if (type !== 'dt-translate' && type !== 'dt-translate-selected') return false;
  // Only this extension's own content scripts reach onMessage, and they send these with a TranslateMessage
  const { texts, title, html } = /** @type {TranslateMessage} */ (msg);
  // Whatever goes wrong, the page gets an answer: an unanswered message would leave it waiting for good
  translate(texts, title, html, type === 'dt-translate-selected')
    .catch((e) => ({ error: `翻译失败：${e instanceof Error ? e.message : String(e)}` }))
    .then(sendResponse);
  return true; // respond asynchronously
});

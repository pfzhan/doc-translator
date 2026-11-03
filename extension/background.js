// Service worker: handles the shortcut and the toolbar button, injects the content script, and sends translation
// requests to the local server. Requests go from here rather than from the page, where the page's CSP, CORS and
// Private Network Access would block them.

// Firefox runs this as an event page without importScripts, and loads the shared files itself (background.scripts in
// the manifest written by scripts/build_firefox.py)
if (typeof importScripts === 'function') importScripts('settings.js', 'sites.js');

async function flashBadge(tabId, text) {
  await chrome.action.setBadgeText({ tabId, text });
  setTimeout(() => chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {}), 3000);
}

/** Send a message to the content script in the top frame of tab, injecting the script first into every frame that
 * has none yet (frames of other sites need host access; without it they are skipped). The iframes then follow the
 * state the top frame reports. */
async function sendToTab(tab, message) {
  if (!tab?.id) return;
  let state;
  try {
    const frames = await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      func: () => !!window.__docTranslator,
    });
    const missing = frames.filter((f) => !f.result).map((f) => f.frameId);
    // The top frame first and on its own: an iframe that went away meanwhile (rotating ads) fails only itself
    if (missing.includes(0)) await inject(tab.id, 0);
    await Promise.allSettled(missing.filter((id) => id !== 0).map((id) => inject(tab.id, id)));
    state = await chrome.tabs.sendMessage(tab.id, message, { frameId: 0 });
    const subframes = frames.map((f) => f.frameId).filter((id) => id !== 0);
    await Promise.allSettled(
      subframes.map((frameId) => chrome.tabs.sendMessage(tab.id, { type: 'dt-sync', ...state }, { frameId })),
    );
  } catch {
    // chrome:// pages and the Web Store cannot be injected
    if (!state) await flashBadge(tab.id, '×');
  }
}

async function inject(tabId, frameId) {
  const target = { tabId, frameIds: [frameId] };
  await chrome.scripting.insertCSS({ target, files: ['content.css'] });
  await chrome.scripting.executeScript({ target, files: ['content.js'] });
}

async function toggle(tab) {
  if (await showsPdf(tab)) await translatePdf(tab);
  else await sendToTab(tab, { type: 'dt-toggle' });
}

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
const pdfBusy = new Set(); // addresses being downloaded or submitted: pressing again meanwhile does nothing

/** Chrome's viewer page takes scripts and reports the PDF's type; Firefox's takes none, so the address decides. */
async function showsPdf(tab) {
  if (!tab?.id) return false;
  try {
    const [frame] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.contentType,
    });
    return frame?.result === 'application/pdf';
  } catch {
    return /^https?:/.test(tab.url || '') && /\.pdf$/i.test(new URL(tab.url).pathname);
  }
}

async function translatePdf(tab) {
  const url = tab.url;
  if (pdfBusy.has(url)) return;
  pdfBusy.add(url);
  await chrome.action.setBadgeText({ tabId: tab.id, text: '…' });
  let error = '';
  try {
    const s = await loadSettings();
    // The same file with the same settings opens the job submitted before, while the server still has it
    const key = `pdf ${JSON.stringify([s.server, url, s.targetLang, s.sourceLang, s.serviceId, s.mode])}`;
    const { [key]: known } = await chrome.storage.session.get(key);
    const kept = known && (await fetch(`${s.server}/api/jobs/${known}`).catch(() => null))?.ok;
    const id = kept ? known : await submitPdf(s, await downloadPdf(url));
    await chrome.storage.session.set({ [key]: id });
    await chrome.tabs.create({ url: `${s.server}/#read/${id}`, index: tab.index + 1, openerTabId: tab.id });
  } catch (e) {
    error = e.message;
  } finally {
    pdfBusy.delete(url);
  }
  await chrome.action.setBadgeText({ tabId: tab.id, text: '' });
  if (error) {
    await notice(tab.id, error);
    await flashBadge(tab.id, '×');
  }
}

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

/** The file name the site gives, else the last part of the address. The server takes files by their suffix, and
 * addresses such as /pdf/2401.01234 have none. */
function pdfName(url, disposition) {
  const given = disposition?.match(/filename\*=(?:UTF-8'')?"?([^";]+)/i) || disposition?.match(/filename="?([^";]+)/i);
  let name = given ? given[1] : new URL(url).pathname.split('/').pop();
  try {
    name = decodeURIComponent(name);
  } catch {
    // not percent-encoded
  }
  name = name.trim() || 'document';
  return /\.pdf$/i.test(name) ? name : `${name}.pdf`;
}

async function submitPdf(s, { blob, name }) {
  const form = new FormData();
  form.append('file', blob, name);
  form.append('target_lang', s.targetLang);
  form.append('source_lang', s.sourceLang);
  form.append('service_id', s.serviceId);
  form.append('mode', s.mode);
  let resp;
  try {
    resp = await fetch(`${s.server}/api/jobs`, { method: 'POST', body: form });
  } catch {
    throw new Error(unreachable(s.server));
  }
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new Error(errorText(body, resp.status));
  if (typeof body?.id !== 'string') {
    throw new Error(`${s.server} 的回应不是 Doc Translator 的翻译任务，请检查扩展设置里的服务地址`);
  }
  return body.id;
}

/** A message at the corner of the tab. Firefox's viewer takes no scripts: there only the badge shows. */
async function notice(tabId, text) {
  await chrome.scripting.executeScript({ target: { tabId }, func: showNotice, args: [text] }).catch(() => {});
}

/** Runs in the page; looks like the status box of content.js. */
function showNotice(text) {
  const host = document.createElement('dt-toast');
  const shadow = host.attachShadow({ mode: 'closed' });
  shadow.innerHTML = `<style>
    div { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; max-width: 360px;
      padding: 8px 12px; border-radius: 8px; background: rgba(32, 33, 36, .92); color: #fff;
      font: 13px/1.5 system-ui, sans-serif; box-shadow: 0 2px 8px rgba(0, 0, 0, .3); }
  </style><div role="status"></div>`;
  shadow.querySelector('div').textContent = text;
  document.documentElement.append(host);
  setTimeout(() => host.remove(), 6000);
}

// ---------- Selected text ----------

/** frameId: the frame holding the selection (the context menu says which); without it (the shortcut) the frame is
 * the one the user is in, or else the first that has a selection. Other frames may still hold an old selection. */
async function translateSelection(tab, frameId) {
  if (!tab?.id) return;
  try {
    if (frameId === undefined) frameId = await selectionFrame(tab.id);
    if (frameId === undefined) return;
    await chrome.scripting.executeScript({ target: { tabId: tab.id, frameIds: [frameId] }, files: ['selection.js'] });
  } catch {
    await flashBadge(tab.id, '×');
  }
}

async function selectionFrame(tabId) {
  const frames = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
    func: () => {
      const field = document.activeElement;
      const inField = field?.selectionEnd > field?.selectionStart;
      return { focused: document.hasFocus(), selected: inField || !!getSelection()?.toString().trim() };
    },
  });
  const withSelection = frames.filter((f) => f.result?.selected);
  return (withSelection.find((f) => f.result.focused) || withSelection[0])?.frameId;
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

function errorText(body, status) {
  const detail = body?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ');
  return `HTTP ${status}`;
}

// Also the error when access to a server on another machine was not granted (see options.js)
const unreachable = (server) => `连不上翻译服务（${server}），请先启动 Doc Translator，或在扩展的设置页重新连接`;

// A request the server has not answered by then has failed; LLM services take tens of seconds for a full batch
const REQUEST_TIMEOUT = 120000;

/** html: aligned with texts, which are HTML fragments (see richOf in content.js); empty means none. */
async function translate(texts, title, html = []) {
  const s = await loadSettings();
  let resp;
  try {
    resp = await fetch(`${s.server}/api/web/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        texts,
        html,
        title,
        target_lang: s.targetLang,
        source_lang: s.sourceLang,
        service_id: s.serviceId,
      }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT),
    });
  } catch (e) {
    if (e.name === 'TimeoutError') return { error: `翻译服务 ${REQUEST_TIMEOUT / 1000} 秒内没有回应` };
    return { error: unreachable(s.server) };
  }
  const body = await resp.json().catch(() => null);
  if (!resp.ok) return { error: errorText(body, resp.status) };
  // Another program on the configured port, or a proxy page, answers 200 with something else
  const list = body?.translations;
  if (!Array.isArray(list) || list.length !== texts.length || list.some((t) => t !== null && typeof t !== 'string')) {
    return { error: `${s.server} 的回应不是 Doc Translator 的翻译结果，请检查扩展设置里的服务地址` };
  }
  return { translations: list, lang: s.targetLang };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== 'dt-translate') return false;
  // Whatever goes wrong, the page gets an answer: an unanswered message would leave it waiting for good
  translate(msg.texts, msg.title, msg.html)
    .catch((e) => ({ error: `翻译失败：${e.message}` }))
    .then(sendResponse);
  return true; // respond asynchronously
});

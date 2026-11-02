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
    // chrome:// pages, the Web Store and the built-in PDF viewer cannot be injected
    if (!state) await flashBadge(tab.id, '×');
  }
}

async function inject(tabId, frameId) {
  const target = { tabId, frameIds: [frameId] };
  await chrome.scripting.insertCSS({ target, files: ['content.css'] });
  await chrome.scripting.executeScript({ target, files: ['content.js'] });
}

function toggle(tab) {
  return sendToTab(tab, { type: 'dt-toggle' });
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
    // Also the error when access to a server on another machine was not granted (see options.js)
    return { error: `连不上翻译服务（${s.server}），请先启动 Doc Translator，或在扩展的设置页重新连接` };
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

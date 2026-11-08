// Options page. Settings are defined in settings.js; the lists of services and languages come from the local
// server.

const START_COMMAND = '.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765';

/**
 * The element with this id, which options.html always has.
 * @param {string} id
 * @returns {HTMLElement}
 */
function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} is missing`);
  return el;
}

/**
 * The same, of a given kind.
 * @template {HTMLElement} T
 * @param {string} id
 * @param {{ new (): T, name: string }} kind
 * @returns {T}
 */
function byId(id, kind) {
  const el = $(id);
  if (!(el instanceof kind)) throw new Error(`#${id} is not a ${kind.name}`);
  return el;
}

/**
 * The server's answers this page reads (app/routes/services.py, app/routes/web.py, app/languages.py).
 * @typedef {{ services: { id: string, name: string, enabled: boolean }[], default: string }} ServiceList
 * @typedef {{ code: string, name: string, native: string }} Language
 * @typedef {{ entries: number }} CacheSize
 */

const LOCAL_HOSTS = ['127.0.0.1', 'localhost'];

/**
 * The server's origin, or null when the value is not an http(s) URL.
 * @param {string} value
 */
function normalizeServer(value) {
  let url;
  try {
    url = new URL(value.trim());
  } catch {
    return null;
  }
  return ['http:', 'https:'].includes(url.protocol) ? url.origin : null;
}

/**
 * Access to this machine over http is granted at install (host_permissions); other servers, and https here, are
 * optional_host_permissions, asked for when the user connects to one.
 * @param {string} server
 * @returns {chrome.permissions.Permissions | null}
 */
function accessFor(server) {
  const url = new URL(server);
  const local = url.protocol === 'http:' && LOCAL_HOSTS.includes(url.hostname);
  return local ? null : { origins: [`${url.protocol}//${url.hostname}/*`] };
}

class HTTPError extends Error {
  /** @param {number} status */
  constructor(status) {
    super(`HTTP ${status}`);
    this.status = status;
  }
}

/**
 * @param {string} text
 * @param {'' | 'error' | 'ok'} [kind]
 */
function setStatus(text, kind = '') {
  const el = $('status');
  el.textContent = text;
  el.className = `hint ${kind}`;
}

/**
 * @param {HTMLSelectElement} select
 * @param {[string, string][]} options value and label of each option
 * @param {string} value the option to select, when there is one with it
 */
function fill(select, options, value) {
  select.replaceChildren(...options.map(([v, label]) => new Option(label, v)));
  select.value = options.some(([v]) => v === value) ? value : options[0]?.[0] ?? '';
}

/**
 * The answer of the server the user chose; its shape is taken on trust, as the rest of the page uses it.
 * @template T
 * @param {Settings} s
 * @param {string} path
 * @returns {Promise<T>}
 */
async function getJSON(s, path) {
  const resp = await fetch(s.server + path, { headers: authHeaders(s) });
  if (!resp.ok) throw new HTTPError(resp.status);
  return resp.json();
}

/**
 * Give up access to a server the extension no longer uses, unless it is also a site translated automatically.
 * Access to this machine over http comes from manifest.json and cannot be given up.
 * @param {string} server
 */
async function releaseServer(server) {
  const access = normalizeServer(server || '') && accessFor(server);
  if (!access || (await loadAutoSites()).includes(siteOf(server))) return;
  await chrome.permissions.remove(access).catch(() => {});
}

let attempts = 0;

/**
 * byUser: the user pressed 连接; only then may the browser ask for access to another server.
 * @param {boolean} [byUser]
 */
async function connect(byUser = false) {
  // A slower earlier attempt (the one on page load, say) must not overwrite what a later one shows
  const attempt = ++attempts;
  const stale = () => attempt !== attempts;
  const serverInput = byId('server', HTMLInputElement);
  const server = normalizeServer(serverInput.value);
  if (!server) {
    setStatus('地址要以 http:// 或 https:// 开头，例如 http://127.0.0.1:8765', 'error');
    return;
  }
  const access = accessFor(server);
  // permissions.request needs the user's click, so it comes before anything is awaited
  const granted = !access || (byUser ? chrome.permissions.request(access) : chrome.permissions.contains(access));
  serverInput.value = server;
  if (!(await granted)) {
    // Not saved: every translation would fail, and the address would sync to the user's other browsers
    if (!stale()) setStatus(`没有访问 ${server} 的权限，请点「连接」并在浏览器的提示中允许`, 'error');
    return;
  }
  const previous = (await loadSettings()).server;
  await chrome.storage.sync.set({ server });
  await chrome.storage.local.set({ token: byId('token', HTMLInputElement).value.trim() });
  if (byUser && previous !== server) await releaseServer(previous);
  const s = await loadSettings();
  if (stale()) return;
  setStatus('连接中…');
  /** @type {ServiceList} */
  let services;
  /** @type {Language[]} */
  let languages;
  try {
    [services, languages] = await Promise.all([
      getJSON(s, '/api/services'),
      /** @type {Promise<Language[]>} */ (getJSON(s, '/api/languages')),
    ]);
  } catch (e) {
    if (stale()) return;
    const status = e instanceof HTTPError ? e.status : 0; // 0: no answer at all
    if (status === 401) {
      const problem = s.token ? '访问令牌不对' : `${server} 需要访问令牌`;
      setStatus(`${problem}，请填服务端 ACCESS_TOKEN 的值`, 'error');
    } else if (status === 403 && access) {
      // The server has no ACCESS_TOKEN and answers only requests from its own machine
      setStatus(`${server} 只接受本机的请求：请在服务端设置环境变量 ACCESS_TOKEN，并在这里填它的值`, 'error');
    } else if (status === 400 && access) {
      // TrustedHostMiddleware rejects host names that are not in ALLOWED_HOSTS
      setStatus(`${server} 拒绝了请求：请在服务端的环境变量 ALLOWED_HOSTS 中加入 ${new URL(server).hostname}`, 'error');
    } else if (access) {
      setStatus(`连不上 ${server}，请确认服务已启动，并监听在外部可访问的地址上（如 --host 0.0.0.0）`, 'error');
    } else {
      setStatus(`连不上 ${server}。请在 Doc Translator 目录下启动服务：${START_COMMAND}`, 'error');
    }
    return;
  }
  if (stale()) return;
  const defaultName = services.services.find((x) => x.id === services.default)?.name || services.default;
  const enabled = services.services.filter((x) => x.enabled);
  /** @type {[string, string][]} */
  const serviceOptions = [['', `默认服务（${defaultName}）`]];
  for (const x of enabled) serviceOptions.push([x.id, x.name]);
  fill(byId('service', HTMLSelectElement), serviceOptions, s.serviceId);
  // Removed or disabled on the server since: the list falls back to the default, and translations must use what it
  // shows, or each would fail while this page looks fine
  const serviceGone = s.serviceId !== '' && !enabled.some((x) => x.id === s.serviceId);
  if (serviceGone) await chrome.storage.sync.set({ serviceId: '' });
  const names = languages.map(
    (l) => /** @type {[string, string]} */ ([l.code, l.code === 'auto' ? '自动检测' : `${l.native}（${l.name}）`]),
  );
  fill(byId('target', HTMLSelectElement), names.filter(([code]) => code !== 'auto'), s.targetLang);
  fill(byId('source', HTMLSelectElement), names, s.sourceLang);
  fill(byId('alt', HTMLSelectElement), names.filter(([code]) => code !== 'auto'), s.altLang);
  setStatus(serviceGone ? '已连接。之前选的翻译服务已删除或停用，已改用默认服务' : '已连接', 'ok');
  showCache(s);
}

// ---------- Translation cache ----------

/** @param {Settings} s */
async function showCache(s) {
  const clear = byId('clear-cache', HTMLButtonElement);
  try {
    /** @type {CacheSize} */
    const { entries } = await getJSON(s, '/api/web/cache');
    $('cache-count').textContent = `${entries.toLocaleString()} 条`;
    clear.disabled = entries === 0;
  } catch {
    // The server predates the cache endpoints, or went away since connecting
    $('cache-count').textContent = '无法获取';
    clear.disabled = true;
  }
}

async function clearCache() {
  const s = await loadSettings();
  const { server } = s;
  if (!confirm('清理网页翻译的缓存？之后打开的网页会重新请求翻译服务。')) return;
  byId('clear-cache', HTMLButtonElement).disabled = true;
  try {
    const resp = await fetch(`${server}/api/web/cache`, { method: 'DELETE', headers: authHeaders(s) });
    if (!resp.ok) throw new HTTPError(resp.status);
    /** @type {CacheSize} */
    const { entries } = await resp.json();
    $('cache-status').textContent = `已清理 ${entries.toLocaleString()} 条`;
  } catch {
    $('cache-status').textContent = `清理失败，请确认 ${server} 的服务已启动`;
  }
  await showCache(s);
}

async function showShortcut() {
  const commands = (await chrome.commands.getAll()).filter((c) => c.description);
  $('shortcut-list').replaceChildren(
    ...commands.map((c) => {
      const li = document.createElement('li');
      const kbd = document.createElement('kbd');
      kbd.textContent = c.shortcut || '未设置';
      li.append(`${c.description}：`, kbd);
      return li;
    }),
  );
}

async function init() {
  const s = await loadSettings();
  const server = byId('server', HTMLInputElement);
  const token = byId('token', HTMLInputElement);
  server.value = s.server;
  token.value = s.token;
  $('connect').addEventListener('click', () => connect(true));
  $('clear-cache').addEventListener('click', clearCache);
  server.addEventListener('keydown', (e) => e.key === 'Enter' && connect(true));
  token.addEventListener('keydown', (e) => e.key === 'Enter' && connect(true));
  /** @type {[string, keyof SyncedSettings][]} */
  const selects = [
    ['service', 'serviceId'],
    ['target', 'targetLang'],
    ['source', 'sourceLang'],
    ['alt', 'altLang'],
    ['mode', 'mode'],
    ['scope', 'scope'],
  ];
  for (const [id, key] of selects) {
    const select = byId(id, HTMLSelectElement);
    select.addEventListener('change', () => chrome.storage.sync.set({ [key]: select.value }));
  }
  const mode = byId('mode', HTMLSelectElement);
  mode.value = s.mode;
  byId('scope', HTMLSelectElement).value = s.scope;
  // The toggle-mode shortcut changes the setting while this page may be open
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.mode) mode.value = String(changes.mode.newValue);
  });
  // Extensions cannot set shortcuts themselves; the browser's own page does it. Firefox opens its page through the
  // API, and does not open chrome:// URLs
  $('shortcuts').addEventListener('click', () => {
    const opened = chrome.commands.openShortcutSettings
      ? chrome.commands.openShortcutSettings()
      : chrome.tabs.create({ url: 'chrome://extensions/shortcuts' });
    // Firefox before 137 has neither the API nor chrome:// pages
    opened.catch(() => alert('请在 about:addons 页面点齿轮图标，选「管理扩展快捷键」'));
  });
  // Refresh after coming back from the shortcuts page
  window.addEventListener('focus', showShortcut);
  await showShortcut();
  await connect();
}

init();

// ---------- Sites translated automatically (sites.js) ----------

let siteListVersion = 0;

async function showSites() {
  // Several events redraw the list; a slower earlier call must not draw an older list over a newer one
  const version = ++siteListVersion;
  /** @type {HTMLLIElement[]} */
  const items = [];
  for (const host of await loadAutoSites()) {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.className = 'host';
    name.textContent = host;
    li.append(name);
    if (!(await hasSitePermission(host))) {
      // Listed on another computer (storage.sync); access is granted per browser
      const note = document.createElement('span');
      note.className = 'hint';
      note.textContent = '这台电脑上未授权，在该网站上点右键「总是翻译此网站」';
      li.append(note);
    }
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '移除';
    remove.addEventListener('click', () => removeAutoSite(host));
    li.append(remove);
    items.push(li);
  }
  if (!items.length) {
    const empty = document.createElement('li');
    empty.className = 'hint';
    empty.textContent = '还没有';
    items.push(empty);
  }
  if (version === siteListVersion) $('sites').replaceChildren(...items);
}

chrome.storage.onChanged.addListener((changes, area) => area === 'sync' && changes.autoSites && showSites());
chrome.permissions.onRemoved.addListener(showSites);
chrome.permissions.onAdded.addListener(showSites);
showSites();

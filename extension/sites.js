// Sites translated as soon as a page opens, without pressing the shortcut. Shared by the service worker
// (importScripts) and the options page (<script>); needs settings.js.
//
// The list of hostnames is kept in chrome.storage.sync, but access to a site is granted per browser: on each one the
// user grants it from the context menu ("总是翻译此网站"), and only sites granted here get the content script, which
// is registered with chrome.scripting so that it also runs on pages opened later.

const SCRIPT_PREFIX = 'dt-site:';

/** @returns {Promise<string[]>} hostnames, as listed in storage.sync */
async function loadAutoSites() {
  // Only this extension writes autoSites, always a list of hostnames
  const { autoSites } = /** @type {{ autoSites: string[] }} */ (await chrome.storage.sync.get({ autoSites: [] }));
  return autoSites;
}

/**
 * The hostname of an http(s) URL; '' for other pages.
 * @param {string} url
 */
function siteOf(url) {
  try {
    const { protocol, hostname } = new URL(url);
    return ['http:', 'https:'].includes(protocol) ? hostname : '';
  } catch {
    return '';
  }
}

/**
 * Host permission patterns for a site, each within one entry of optional_host_permissions in manifest.json.
 * @param {string} host
 */
function siteOrigins(host) {
  return [`http://${host}/*`, `https://${host}/*`];
}

/** @param {string} host */
function hasSitePermission(host) {
  return chrome.permissions.contains({ origins: siteOrigins(host) });
}

/**
 * From the context menu. permissions.request needs the click's user gesture, so it comes before any await.
 * @param {string} host
 * @param {number} tabId the tab the menu was opened on
 */
async function addAutoSite(host, tabId) {
  if (!(await chrome.permissions.request({ origins: siteOrigins(host) }))) return;
  const sites = await loadAutoSites();
  if (!sites.includes(host)) await chrome.storage.sync.set({ autoSites: [...sites, host] });
  // Already listed (from another browser) but not granted here: the storage change above did not happen
  await syncAutoScripts();
  try {
    // The page that is open is translated now; content.js ignores the second load if the shortcut started it
    await chrome.scripting.executeScript({ target: { tabId }, files: ['autostart.js'] });
    await chrome.scripting.insertCSS({ target: { tabId }, files: ['content.css'] });
    await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
  } catch {
    // The tab was closed or navigated away
  }
}

/**
 * From the context menu or the options page.
 * @param {string} host
 */
async function removeAutoSite(host) {
  await chrome.storage.sync.set({ autoSites: (await loadAutoSites()).filter((h) => h !== host) });
  // The service worker unregisters the script when it sees the change (syncAutoScripts). Access to the translation
  // server stays
  if (siteOf((await loadSettings()).server) === host) return;
  for (const origin of siteOrigins(host)) {
    // Fails for the hosts manifest.json requires; they cannot be given up, the other one still is
    await chrome.permissions.remove({ origins: [origin] }).catch(() => {});
  }
}

// ---------- Service worker only ----------

/** @type {Promise<void>} */
let syncing = Promise.resolve();

/** Registers the content script for each listed site granted here, and unregisters it for the others. */
function syncAutoScripts() {
  // Storage changes and context-menu clicks can arrive together: run one at a time
  syncing = syncing.then(reconcileAutoScripts, reconcileAutoScripts);
  return syncing;
}

async function reconcileAutoScripts() {
  /** @type {string[]} */
  const wanted = [];
  for (const host of await loadAutoSites()) {
    if (await hasSitePermission(host)) wanted.push(SCRIPT_PREFIX + host);
  }
  const registered = (await chrome.scripting.getRegisteredContentScripts())
    .map((s) => s.id)
    .filter((id) => id.startsWith(SCRIPT_PREFIX));
  const stale = registered.filter((id) => !wanted.includes(id));
  if (stale.length) await chrome.scripting.unregisterContentScripts({ ids: stale });
  const missing = wanted.filter((id) => !registered.includes(id));
  if (!missing.length) return;
  await chrome.scripting.registerContentScripts(
    missing.map((id) => ({
      id,
      matches: siteOrigins(id.slice(SCRIPT_PREFIX.length)),
      css: ['content.css'],
      js: ['autostart.js', 'content.js'],
      allFrames: false,
      persistAcrossSessions: true,
    })),
  );
}

function createSiteMenus() {
  /** @type {Pick<chrome.contextMenus.CreateProperties, 'contexts' | 'documentUrlPatterns'>} */
  const pages = { contexts: ['page'], documentUrlPatterns: ['http://*/*', 'https://*/*'] };
  // Menus persist across updates: creating the same ids again would fail
  for (const id of ['dt-site-add', 'dt-site-remove']) chrome.contextMenus.remove(id, () => void chrome.runtime.lastError);
  chrome.contextMenus.create({ id: 'dt-site-add', title: '总是翻译此网站', ...pages });
  chrome.contextMenus.create({ id: 'dt-site-remove', title: '不再自动翻译此网站', visible: false, ...pages });
}

/** Menus cannot differ per tab, so the entry shown follows the active tab. */
async function refreshSiteMenu() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  // tab.url is only readable on sites we may access, which every granted auto site is
  const host = siteOf(tab?.url || '');
  const on = host !== '' && (await loadAutoSites()).includes(host) && (await hasSitePermission(host));
  await chrome.contextMenus.update('dt-site-add', { visible: !on });
  await chrome.contextMenus.update('dt-site-remove', { visible: on });
}

/**
 * @param {chrome.contextMenus.OnClickData} info
 * @param {chrome.tabs.Tab} [tab] missing when the menu was not opened on a tab
 */
function onSiteMenuClicked(info, tab) {
  const host = siteOf(info.pageUrl || '');
  if (!host) return;
  if (info.menuItemId === 'dt-site-add' && tab?.id !== undefined) addAutoSite(host, tab.id).then(refreshSiteMenu);
  if (info.menuItemId === 'dt-site-remove') removeAutoSite(host).then(refreshSiteMenu);
}

/** Called once at the top of the service worker, so that the listeners are there when an event wakes it. */
function listenForSites() {
  chrome.contextMenus.onClicked.addListener(onSiteMenuClicked);
  chrome.runtime.onInstalled.addListener(async () => {
    createSiteMenus();
    // Registered content scripts may or may not survive an update; those that do keep the files and options of the
    // old version, so all are registered afresh
    const ids = (await chrome.scripting.getRegisteredContentScripts())
      .map((s) => s.id)
      .filter((id) => id.startsWith(SCRIPT_PREFIX));
    if (ids.length) await chrome.scripting.unregisterContentScripts({ ids });
    await syncAutoScripts();
    await refreshSiteMenu();
  });
  chrome.runtime.onStartup.addListener(syncAutoScripts);
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'sync' && changes.autoSites) syncAutoScripts().then(refreshSiteMenu);
  });
  // Access can also be revoked (or granted) from the browser's own site settings
  chrome.permissions.onRemoved.addListener(() => syncAutoScripts().then(refreshSiteMenu));
  chrome.permissions.onAdded.addListener(() => syncAutoScripts().then(refreshSiteMenu));
  chrome.tabs.onActivated.addListener(refreshSiteMenu);
  chrome.tabs.onUpdated.addListener((_tabId, change, tab) => {
    if (tab.active && (change.status || change.url)) refreshSiteMenu();
  });
  chrome.windows.onFocusChanged.addListener(refreshSiteMenu);
}

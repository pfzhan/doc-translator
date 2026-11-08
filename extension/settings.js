// Settings shared by the service worker (importScripts) and the options page (<script>), kept in chrome.storage.sync.

/**
 * @typedef {object} SyncedSettings
 * @property {string} server origin of the Doc Translator server
 * @property {string} targetLang
 * @property {string} sourceLang 'auto' or a language code
 * @property {string} altLang selected text already in the target language is translated into this one
 * @property {string} serviceId empty: the server's default service
 * @property {'bilingual' | 'translated'} mode 'translated' hides the originals of translated paragraphs (read by
 *   content.js directly)
 * @property {'page' | 'main'} scope 'main' leaves out the header, footer, navigation and sidebars (read by content.js
 *   directly)
 */

/** @typedef {SyncedSettings & { token: string }} Settings token: the server's ACCESS_TOKEN, kept in storage.local */

/** @type {SyncedSettings} */
const DEFAULTS = {
  server: 'http://127.0.0.1:8765',
  targetLang: 'zh-CN',
  sourceLang: 'auto',
  altLang: 'en',
  serviceId: '',
  mode: 'bilingual',
  scope: 'page',
};

/** @returns {Promise<Settings>} */
async function loadSettings() {
  // The access token stays in storage.local: a secret should not sync to every browser of the account
  const { token = '' } = /** @type {{ token?: string }} */ (await chrome.storage.local.get('token'));
  // Only this extension writes these keys, with values of these types; get(DEFAULTS) fills in the missing ones
  const synced = /** @type {SyncedSettings} */ (await chrome.storage.sync.get(DEFAULTS));
  return { ...synced, token };
}

/**
 * Headers for a request to the server: the access token, needed when the server is on another machine.
 * @param {Settings} s
 * @param {Record<string, string>} [headers]
 * @returns {Record<string, string>}
 */
function authHeaders(s, headers = {}) {
  return s.token ? { ...headers, Authorization: `Bearer ${s.token}` } : headers;
}

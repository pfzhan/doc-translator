// Settings shared by the service worker (importScripts) and the options page (<script>), kept in chrome.storage.sync.

const DEFAULTS = {
  server: 'http://127.0.0.1:8765',
  targetLang: 'zh-CN',
  sourceLang: 'auto',
  altLang: 'en', // selected text already in the target language is translated into this one
  serviceId: '', // empty: the server's default service
  mode: 'bilingual', // or 'translated': hide the originals of translated paragraphs (read by content.js directly)
  scope: 'page', // or 'main': leave out the header, footer, navigation and sidebars (read by content.js directly)
};

async function loadSettings() {
  // The access token stays in storage.local: a secret should not sync to every browser of the account
  const { token = '' } = await chrome.storage.local.get('token');
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))), token };
}

/** Headers for a request to the server: the access token, needed when the server is on another machine. */
function authHeaders(s, headers = {}) {
  return s.token ? { ...headers, Authorization: `Bearer ${s.token}` } : headers;
}

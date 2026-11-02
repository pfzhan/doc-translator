// Settings shared by the service worker (importScripts) and the options page (<script>), kept in chrome.storage.sync.

const DEFAULTS = {
  server: 'http://127.0.0.1:8765',
  targetLang: 'zh-CN',
  sourceLang: 'auto',
  serviceId: '', // empty: the server's default service
  mode: 'bilingual', // or 'translated': hide the originals of translated paragraphs (read by content.js directly)
  scope: 'page', // or 'main': leave out the header, footer, navigation and sidebars (read by content.js directly)
};

async function loadSettings() {
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
}

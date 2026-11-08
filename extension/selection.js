// Translates the selected text and shows the translation in a bubble next to it. The service worker injects this
// file each time the context menu entry or the shortcut is used; Esc or a click elsewhere closes the bubble.
// Translations are written with textContent only, so model output is never parsed as HTML in the page.

(() => {
  // Set up once per page; later injections only translate the new selection
  if (window.__docTranslatorSelection) {
    window.__docTranslatorSelection();
    return;
  }

  // Limits of one request to the server (POST /api/web/translate)
  const MAX_TEXTS = 50;
  const MAX_TEXT = 10000;
  const GAP = 8; // between the bubble and the selection, and the bubble and the edges of the viewport

  // Fields whose selected text may be translated; passwords and other secrets never leave the page
  const TEXT_FIELDS = new Set(['text', 'search', 'url', 'tel', 'email']);
  const SECRET_AUTOCOMPLETE = /\b(one-time-code|cc-[a-z-]+|current-password|new-password)\b/;

  /**
   * The open bubble. rect: the selection's box; scroll: the page's scroll position when the bubble opened.
   * @typedef {{ host: HTMLElement, box: HTMLElement, text: HTMLElement, status: HTMLElement, rect: DOMRect,
   *   scroll: { x: number, y: number } }} Bubble
   */
  /** @type {Bubble | null} */
  let bubble = null;

  /**
   * The focused element. For one in a shadow root, document.activeElement is the host; closed roots are open to
   * extensions only (chrome.dom in Chrome, openOrClosedShadowRoot in Firefox).
   * @returns {Element | null}
   */
  function focusedElement() {
    /** @param {Element} el */
    const shadowOf = (el) =>
      el.shadowRoot ||
      chrome.dom?.openOrClosedShadowRoot?.(/** @type {HTMLElement} */ (el)) ||
      el.openOrClosedShadowRoot;
    let el = document.activeElement;
    for (let inner = el && shadowOf(el)?.activeElement; inner; inner = shadowOf(inner)?.activeElement) el = inner;
    return el;
  }

  /**
   * The selected text and the box to show the bubble next to; null when nothing is selected.
   * @returns {{ text: string, rect: DOMRect } | null}
   */
  function selection() {
    // Text selected in a form field is not part of the document's selection (Chrome still gives it as the
    // document's selected text, so a field inside a shadow root must be found too, or secrets would get through)
    const focused = focusedElement();
    let text, rect;
    // Told apart by localName rather than instanceof: page elements come from the page's world, not this one
    if (focused && ['input', 'textarea'].includes(focused.localName)) {
      const field = /** @type {HTMLInputElement | HTMLTextAreaElement} */ (focused);
      const plain = field.localName === 'textarea' || TEXT_FIELDS.has(field.type);
      const { selectionStart, selectionEnd } = field;
      if (
        !plain ||
        SECRET_AUTOCOMPLETE.test(field.autocomplete) ||
        selectionStart === null ||
        selectionEnd === null ||
        !(selectionEnd > selectionStart)
      ) {
        return null;
      }
      text = field.value.slice(selectionStart, selectionEnd);
      rect = field.getBoundingClientRect();
    } else {
      const sel = getSelection();
      if (!sel?.rangeCount) return null;
      text = sel.toString();
      rect = sel.getRangeAt(0).getBoundingClientRect();
    }
    return text.trim() ? { text, rect } : null;
  }

  /**
   * The selection split into paragraphs, the longer ones cut so that each fits in a request.
   * @param {string} text
   */
  function paragraphs(text) {
    /** @type {string[]} */
    const out = [];
    for (const line of text.split(/\n+/)) {
      let rest = line.replace(/\s+/g, ' ').trim();
      while (rest.length > MAX_TEXT) {
        const cut = cutPoint(rest.slice(0, MAX_TEXT));
        out.push(rest.slice(0, cut).trim());
        rest = rest.slice(cut).trim();
      }
      if (rest) out.push(rest);
    }
    return out;
  }

  /**
   * Where to cut text too long for a request: after a sentence, else between words, else at the limit.
   * @param {string} head
   */
  function cutPoint(head) {
    const sentence = head.match(/^[\s\S]*[.!?。！？]/)?.[0].length || 0;
    if (sentence > head.length / 2) return sentence;
    const space = head.lastIndexOf(' ');
    return space > head.length / 2 ? space : head.length;
  }

  // ---------- Bubble ----------

  /** @param {PointerEvent} e */
  function onPointerDown(e) {
    if (bubble && !e.composedPath().includes(bubble.host)) close();
  }

  /** @param {KeyboardEvent} e */
  function onKeyDown(e) {
    if (e.key === 'Escape') close();
  }

  function close() {
    if (!bubble) return;
    bubble.host.remove();
    document.removeEventListener('pointerdown', onPointerDown, true);
    document.removeEventListener('keydown', onKeyDown, true);
    bubble = null;
  }

  /**
   * @param {DOMRect} rect
   * @returns {Bubble}
   */
  function open(rect) {
    close();
    const host = document.createElement('dt-selection');
    // A closed shadow root keeps the page's styles out of the bubble; all: initial stops the inherited ones too
    const shadow = host.attachShadow({ mode: 'closed' });
    shadow.innerHTML = `<style>
      .bubble { all: initial; display: block; position: absolute; z-index: 2147483647; box-sizing: border-box;
        width: max-content; max-width: min(420px, 100vw - 16px); max-height: min(320px, 50vh); overflow: auto;
        padding: 10px 12px; border: 1px solid #dadce0; border-radius: 8px; background: #fff; color: #202124;
        font: 14px/1.6 system-ui, sans-serif; overflow-wrap: anywhere; box-shadow: 0 2px 10px rgba(0, 0, 0, .2); }
      p { margin: 0 0 .5em; }
      p:last-child { margin: 0; }
      .status { color: #5f6368; font-size: 13px; }
      .text:not(:empty) + .status:not(:empty) { margin-top: .5em; }
      .status:empty { display: none; }
      @media (prefers-color-scheme: dark) {
        .bubble { border-color: #3c4043; background: #202124; color: #e8eaed; }
        .status { color: #9aa0a6; }
      }
    </style><div class="bubble" role="dialog" aria-label="译文">
      <div class="text"></div><div class="status" role="status"></div>
    </div>`;
    // Outside body, like content.js's status box, so content.js's MutationObserver does not see it
    document.documentElement.append(host);
    /** @param {string} selector one of the divs of the markup above */
    const part = (selector) => {
      const el = /** @type {HTMLElement | null} */ (shadow.querySelector(selector));
      if (!el) throw new Error(`${selector} is missing`);
      return el;
    };
    const opened = {
      host,
      box: part('.bubble'),
      text: part('.text'),
      status: part('.status'),
      rect,
      scroll: { x: scrollX, y: scrollY },
    };
    bubble = opened;
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('keydown', onKeyDown, true);
    return opened;
  }

  /** Below the selection, or above it when there is more room there; kept inside the viewport horizontally. */
  function place() {
    if (!bubble) return;
    const { box, rect, scroll } = bubble;
    const view = document.documentElement;
    const left = Math.max(GAP, Math.min(rect.left, view.clientWidth - box.offsetWidth - GAP));
    const below = view.clientHeight - rect.bottom;
    const top =
      below >= box.offsetHeight + 2 * GAP || below >= rect.top
        ? rect.bottom + GAP
        : Math.max(GAP, rect.top - GAP - box.offsetHeight);
    // The bubble scrolls with the page, so it is placed in page coordinates as of when it opened
    box.style.left = `${left + scroll.x}px`;
    box.style.top = `${top + scroll.y}px`;
  }

  /**
   * @param {Bubble} open
   * @param {string} message
   */
  function setStatus(open, message) {
    open.status.textContent = message;
    place();
  }

  /**
   * langs: aligned with texts, the language of each translation; null for an original kept as it is.
   * @param {Bubble} open
   * @param {string[]} texts
   * @param {(string | null)[]} langs
   */
  function append(open, texts, langs) {
    texts.forEach((text, i) => {
      const p = document.createElement('p');
      const lang = langs[i];
      if (lang) p.lang = lang;
      p.textContent = text;
      open.text.append(p);
    });
    place();
  }

  // ---------- Translating ----------

  async function translateSelection() {
    const sel = selection();
    if (!sel) return;
    const texts = paragraphs(sel.text);
    const current = open(sel.rect);
    setStatus(current, '翻译中…');
    let translated = false;
    for (let i = 0; i < texts.length; i += MAX_TEXTS) {
      const batch = texts.slice(i, i + MAX_TEXTS);
      /** @type {TranslateMessage} */
      const message = { type: 'dt-translate-selected', texts: batch, title: document.title };
      /** @type {TranslateAnswer | undefined} */
      let result;
      try {
        result = await chrome.runtime.sendMessage(message);
      } catch {
        // The extension was reloaded or removed; the script left in this page can no longer reach it
        result = { error: '扩展已更新，请刷新页面后重试' };
      }
      if (bubble !== current) return; // closed, or replaced by a newer selection
      if (!result || 'error' in result) {
        setStatus(current, `翻译失败：${result?.error || '未知错误'}`);
        return;
      }
      const { translations, langs } = result;
      // null keeps the original, so the bubble still reads in the order of the selection
      translated ||= translations.some((t) => t);
      append(
        current,
        batch.map((text, j) => translations[j] || text),
        langs,
      );
    }
    if (!translated) current.text.replaceChildren();
    setStatus(current, translated ? '' : '选中的文字不需要翻译');
  }

  window.__docTranslatorSelection = translateSelection;
  translateSelection();
})();

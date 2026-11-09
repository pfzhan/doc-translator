// Content script: splits the page into paragraphs, translates those near the viewport through the background
// service worker, and inserts each translation after its original. Model output never reaches the page as HTML:
// plain translations are written with textContent, and formatted ones are rebuilt node by node (see buildRich).
//
// Each shortcut press: the first starts translating; after a failed request it retries; otherwise it hides or shows
// the translations. Nothing new is sent while they are hidden. In translation-only mode (setting "mode") the original
// of each translated paragraph is hidden; hiding the translations brings the originals back. With the setting
// "scope" set to "main", only the page's main content is translated (see inScope).

(() => {
  if (window.__docTranslator) return; // injected once per page; later presses only send messages
  window.__docTranslator = true;

  const TRANS_TAG = 'dt-trans';
  // The script runs in every frame it may access; the top frame owns the status box and the shortcut's state, and
  // iframes follow it (see dt-sync)
  const IS_TOP = window === window.top;
  // Never translated, and not part of the surrounding sentence either. A <slot> shows light DOM content, which is
  // collected where it is; its own children show only when nothing is assigned to it
  const IGNORED_TAGS = new Set([
    'script', 'style', 'noscript', 'template', 'link', 'meta', 'wbr', 'hr',
    'img', 'picture', 'video', 'audio', 'canvas', 'iframe', 'object', 'embed',
    'svg', 'input', 'textarea', 'select', 'button', 'option', 'slot', TRANS_TAG,
  ]);
  // Text kept for screen readers only, by the class names sites and CSS frameworks give it. Other such text is
  // recognized by its style (see visuallyHidden)
  const SR_ONLY = '.sr-only, .visually-hidden, .visuallyhidden, .screen-reader-text';
  // Rendered math: KaTeX, MathJax 2 and 3, Wikipedia, and MathML. Each formula is copied whole into the translation,
  // within its sentence; a display formula on its own line is left as it is. Their parts for screen readers and
  // their glyphs are never read as text
  const MATH = '.katex, .MathJax, mjx-container, .mwe-math-element, math';
  // Hosts of this extension's own boxes (status, selected text, PDF notices)
  const OWN_HOSTS = new Set(['dt-toast', 'dt-selection']);
  // Blocks kept as they are. Inline code inside a sentence stays part of the sentence
  const SKIP_TAGS = ['pre', 'code', 'kbd', 'samp'];
  // Inline code, keys, sample output and variables: sent with their sentence, and copied from the page into the
  // translation, so neither their text nor their formatting comes from the model
  const CODE_TAGS = new Set(['code', 'kbd', 'samp', 'var']);
  // Tags that only occur inside a sentence. Hidden ones are dropped instead of splitting the sentence around them
  const PHRASING_TAGS = new Set([
    'a', 'abbr', 'b', 'bdi', 'bdo', 'cite', 'data', 'dfn', 'em', 'font', 'i', 'label', 'mark', 'q', 's',
    'small', 'span', 'strong', 'sub', 'sup', 'time', 'u', 'var',
  ]);
  // Inline formatting kept in translations. A paragraph containing one is sent as HTML with these tags only, each
  // numbered (i="N") and without attributes; other inline wrappers (span, font...) are sent as their text
  const RICH_TAGS = new Set(['a', 'b', 'strong', 'i', 'em', 'u', 's', 'sub', 'sup', 'mark']);
  // Content being edited is never collected, so what the user types is not sent anywhere
  const EDITABLE = '[contenteditable]:not([contenteditable="false"])';
  const OPT_OUT = `[translate="no"], .notranslate, ${EDITABLE}`;
  // With scope "main": the main content when the page marks one, else everything outside these landmarks
  const MAIN_SELECTOR = 'main, [role="main"]';
  const CHROME_SELECTOR =
    'header, footer, nav, aside, [role="banner"], [role="contentinfo"], [role="navigation"], [role="complementary"]';
  // Inside the main content, headers and footers belong to the article (title, byline); only these are left out
  const CHROME_IN_MAIN = 'nav, aside, [role="navigation"], [role="complementary"]';
  // Same as the server's default batch size (Translator.max_batch_items / max_batch_chars) and text limit
  const BATCH_ITEMS = 20;
  const BATCH_CHARS = 3000;
  const MAX_TEXT = 10000;
  const MAX_INFLIGHT = 2;
  const RESCAN_DELAY = 300;
  // A paragraph the page changes is translated again once it has stayed the same this long (replies streamed in),
  // and no more after its translation was replaced this many times (tickers, pages that keep re-rendering)
  const SETTLE_DELAY = 1000;
  const MAX_CHURN = 10;

  // Elements that keep showing in translation-only mode: hiding them would lose content the translation lacks
  const KEEP_SELECTOR = 'img, picture, svg, video, audio, canvas, iframe, object, embed, input, select, textarea, button';

  /**
   * A paragraph: a run of inline content with the same parent. Its translation el goes before after (null: at the
   * end of parent). A paragraph the page changed later is stale: a new one takes its place.
   * @typedef {object} Paragraph
   * @property {undefined} [attr]
   * @property {ChildNode[]} nodes
   * @property {Element | ShadowRoot} parent
   * @property {ChildNode | null} after
   * @property {string} text what the paragraph says; set when registered
   * @property {string} payload what is sent: its text, or its HTML when it has inline formatting (see richOf)
   * @property {Rich | null} rich
   * @property {HTMLElement} [el] its translation, once shown
   * @property {HTMLElement} [reuse] the translation of its earlier version, which it takes over
   * @property {boolean} [stale]
   * @property {boolean} [redo] read again on the next scan even if unchanged (see rescan)
   */
  /**
   * An attribute translated along with the paragraphs.
   * @typedef {object} AttrUnit
   * @property {'placeholder' | 'title'} attr
   * @property {Element} target the element carrying it
   * @property {string} text
   * @property {string} payload
   * @property {string} [translation]
   * @property {string | null} [current] the value this script last wrote into the attribute
   * @property {boolean} [stale]
   */
  /** @typedef {Paragraph | AttrUnit} Unit */
  /** @typedef {{ html: string, refs: Element[] }} Rich a paragraph as HTML, and the elements its i numbers stand for */

  let started = false;
  let hidden = false;
  /** @type {'bilingual' | 'translated'} */
  let mode = 'bilingual';
  /** @type {'page' | 'main'} */
  let scope = 'page';
  let pageHasMain = false; // whether the page marks its main content; updated by collect
  /** @type {Unit[]} */
  let failed = []; // paragraphs of failed requests; non-empty means the next press retries
  let inflight = 0;
  /** @type {WeakMap<Node, Paragraph>} */
  const unitOf = new WeakMap(); // node of a collected paragraph -> that paragraph
  /** @type {WeakMap<Element, Paragraph>} */
  const ownerOf = new WeakMap(); // dt-trans -> the paragraph it translates
  /** @type {WeakMap<Node, number>} */
  const churn = new WeakMap(); // parent element -> how many of its translations were replaced
  /** @type {Record<AttrUnit['attr'], WeakSet<Element>>} */
  const seenAttrs = { placeholder: new WeakSet(), title: new WeakSet() }; // elements whose attribute was collected
  /** @type {Set<Unit>} */
  const shown = new Set(); // paragraphs and attributes whose translation is on the page
  /** @type {Map<Node, string>} */
  const originals = new Map(); // text node emptied in translation-only mode -> its text
  /** @type {Map<Element, Unit[]>} */
  const waiting = new Map(); // observed element -> its paragraphs not yet near the viewport
  /** @type {Unit[]} */
  const queue = []; // paragraphs near the viewport, not sent yet

  /** Screen-reader-only text is clipped to a 1px box, and a collapsed panel has no height: translating what such a
   * box holds would only spend quota.
   * @param {DOMRectReadOnly} box */
  const hasArea = ({ width, height }) => width > 1 && height > 1;

  const visibility = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        visibility.unobserve(entry.target);
        if (!hasArea(entry.boundingClientRect)) {
          // Still in view when it opens, so the visibility observer would not report it again
          growth.observe(entry.target);
          continue;
        }
        queue.push(...(waiting.get(entry.target) || []).filter(current));
        waiting.delete(entry.target);
      }
      pump();
    },
    { rootMargin: '100% 0px' }, // one screen above and below
  );

  /** Boxes too small to translate what they hold, until they grow. */
  const growth = new ResizeObserver((entries) => {
    for (const { target } of entries) {
      if (!hasArea(target.getBoundingClientRect())) continue;
      growth.unobserve(target);
      visibility.observe(target);
    }
  });

  // ---------- Shadow DOM ----------

  // Shadow roots are entered like the rest of the page. closest() and contains() stop at a shadow root, so the
  // helpers below go on to its host. The page's styles and content.css do not reach into shadow roots either: each
  // root a translation goes into gets content.css as a <style> of its own

  /** The shadow root of el. Closed ones are open to extensions only: chrome.dom in Chrome, openOrClosedShadowRoot in
   * Firefox. Those are only looked for on custom elements, where nearly all are; an open one on any element. */
  /** @param {Element} el
   * @returns {ShadowRoot | null} */
  function shadowOf(el) {
    if (el.shadowRoot) return el.shadowRoot;
    if (!el.localName.includes('-') || OWN_HOSTS.has(el.localName)) return null;
    // Custom elements (a hyphen in the name) are HTML elements
    const custom = /** @type {HTMLElement} */ (el);
    if (chrome.dom?.openOrClosedShadowRoot) return chrome.dom.openOrClosedShadowRoot(custom) || null;
    return el.openOrClosedShadowRoot || null;
  }

  /** Whether el is a shadow host that does not show its own children: its shadow root has no slot for them.
   * @param {Element} el */
  function slotless(el) {
    const shadow = shadowOf(el);
    return !!shadow && !shadow.querySelector('slot');
  }

  // Node types and casts rather than instanceof: content scripts run in an isolated world, where nodes of other
  // frames' documents are not instances of this world's classes
  /** @param {Node | null | undefined} node
   * @returns {node is ShadowRoot} */
  const isShadow = (node) =>
    node?.nodeType === Node.DOCUMENT_FRAGMENT_NODE && !!(/** @type {ShadowRoot} */ (node)).host;
  /** @param {Node} node
   * @returns {node is Element} */
  const isElement = (node) => node.nodeType === Node.ELEMENT_NODE;
  /** @param {Node} node
   * @returns {node is Text} */
  const isText = (node) => node.nodeType === Node.TEXT_NODE;
  /** The element itself, or the host of a shadow root.
   * @param {Element | ShadowRoot} node */
  const elementOf = (node) => (isShadow(node) ? node.host : node);
  /** The parent element, or the host for an element at the top of a shadow root.
   * @param {Element} el */
  const parentOf = (el) => el.parentElement || (isShadow(el.parentNode) ? el.parentNode.host : null);
  /** The host of the shadow root node is in; null in the document itself.
   * @param {Node} node */
  const hostOf = (node) => {
    const root = node.getRootNode();
    return isShadow(root) ? root.host : null;
  };

  /** @param {Element | ShadowRoot} node
   * @param {string} selector */
  function closestComposed(node, selector) {
    for (let /** @type {Element | null} */ el = elementOf(node); el; el = hostOf(el)) {
      const found = el.closest(selector);
      if (found) return found;
    }
    return null;
  }

  /** @param {Node} outer
   * @param {Node} node */
  function containsComposed(outer, node) {
    for (let /** @type {Node | null} */ n = node; n; n = hostOf(n)) if (outer.contains(n)) return true;
    return false;
  }

  /** @type {Set<ShadowRoot>} */
  const shadowRoots = new Set(); // shadow roots entered: watched for changes like body
  /** @type {Set<Element>} */
  const bareHosts = new Set(); // custom elements not defined yet when scanned; checked for a shadow root once they are
  /** @type {WeakMap<ShadowRoot, HTMLStyleElement>} */
  const ownStyles = new WeakMap(); // shadow root -> its copy of content.css
  /** @type {WeakSet<Node>} */
  const styleEls = new WeakSet(); // those copies: their mutation records are ours
  /** @type {Promise<string> | null} */
  let css = null; // promise of the text of content.css

  /** @returns {Promise<string>} */
  async function loadCss() {
    // Content scripts cannot fetch the extension's files; the service worker can
    try {
      /** @type {CssMessage} */
      const message = { type: 'dt-css' };
      const text = await chrome.runtime.sendMessage(message);
      return typeof text === 'string' ? text : '';
    } catch {
      return '';
    }
  }

  /** @param {ShadowRoot} root */
  function enterShadow(root) {
    if (shadowRoots.has(root)) return;
    shadowRoots.add(root);
    mutations.observe(root, CHANGES);
    css ||= loadCss();
  }

  /** Give the shadow root el is in content.css, again if the page replaced the root's content since.
   * @param {Element} el */
  async function styleShadow(el) {
    const root = el.getRootNode();
    if (!isShadow(root) || ownStyles.get(root)?.parentNode === root) return;
    const style = document.createElement('style');
    styleEls.add(style);
    ownStyles.set(root, style);
    style.textContent = await (css ||= loadCss());
    // At the end, so the page's own first children keep their places
    if (ownStyles.get(root) === style) root.append(style);
  }

  // ---------- Finding paragraphs ----------

  /** @param {Element} el */
  function optedOut(el) {
    // translate="no" on <html> or <body> is usually aimed at the browser's own translator; the user asked for this
    // page explicitly, so only opt-outs below body count
    return el !== document.body && el !== document.documentElement && el.matches(OPT_OUT);
  }

  /** A rendered formula (see MATH): read as one piece, never entered.
   * @param {Element} el */
  const isMath = (el) => el.matches(MATH);

  /** Copied whole into the translation: text marked not to translate, formulas, inline code (see CODE_TAGS), and
   * inline shadow hosts whose text is all in their shadow root.
   * @param {Element} el */
  const keptWhole = (el) => optedOut(el) || isMath(el) || CODE_TAGS.has(el.localName) || slotless(el);

  /** Whether el is clipped out of sight while screen readers still read it: positioned and cut to at most a pixel by
   * its size, clip or clip-path. Being positioned makes it a block, which would split the sentence around it.
   * @param {Element} el
   * @param {CSSStyleDeclaration} style */
  function visuallyHidden(el, style) {
    if (el.matches(SR_ONLY)) return true;
    if (style.position !== 'absolute' && style.position !== 'fixed') return false;
    const tiny = parseFloat(style.width) <= 1 && parseFloat(style.height) <= 1;
    if (tiny && (style.overflow !== 'visible' || style.clip !== 'auto' || style.clipPath !== 'none')) return true;
    if (/^inset\(\s*50%/.test(style.clipPath)) return true;
    // rect(top, right, bottom, left): nothing shows when the rectangle has no area
    const rect = style.clip.match(/^rect\(([-\d.]+)px,? ([-\d.]+)px,? ([-\d.]+)px,? ([-\d.]+)px\)$/);
    if (!rect) return false;
    const [top, right, bottom, left] = rect.slice(1).map(Number);
    return right - left <= 1 && bottom - top <= 1;
  }

  /** ignore: contributes nothing; inline: part of the current paragraph; block: ends it; skip: ends it, not entered
   * @param {Element} el
   * @returns {'ignore' | 'inline' | 'block' | 'skip'} */
  function classify(el) {
    if (IGNORED_TAGS.has(el.localName)) return 'ignore';
    // Hidden by translation-only mode: still the inline element it was when collected
    if (el.hasAttribute('data-dt-original')) return 'inline';
    const style = getComputedStyle(el);
    const display = style.display;
    if (display === 'none' && PHRASING_TAGS.has(el.localName)) return 'ignore';
    // Screen-reader-only text inside a sentence (a link's longer name, the MathML behind a rendered formula) is
    // dropped rather than splitting the sentence; such a block is not worth translating either
    if (visuallyHidden(el, style)) return 'ignore';
    // MathML's own display value is "math" inline and "block math" on a line of its own
    const inline = display.startsWith('inline') || display === 'contents' || display === 'math';
    // Text marked not to translate stays in its sentence and comes back unchanged (see richOf); editable content
    // and marked blocks are left out
    if (optedOut(el)) return inline && !el.matches(EDITABLE) ? 'inline' : 'skip';
    // A formula in a sentence is copied into its translation (see keptWhole); one on its own line stays as it is
    if (isMath(el)) return inline ? 'inline' : 'skip';
    // A shadow host shows its shadow root, with its own children placed at the slots. An inline one holding only
    // inline content stays in its sentence: one that shows its children (a styled link) like any inline element, one
    // that does not (a date, a badge) copied whole into the translation (see keptWhole). Any other is laid out apart
    // from the text around it
    const shadow = shadowOf(el);
    if (shadow) return inline && !hasBlockChild(shadow) && !hasBlockChild(el) ? 'inline' : 'block';
    // Icon fonts write their icon names as aria-hidden text
    if (inline) return el.getAttribute('aria-hidden') === 'true' ? 'ignore' : 'inline';
    if (SKIP_TAGS.includes(el.localName)) return 'skip';
    // display:none blocks are still collected: they are translated once shown, when they enter the viewport
    return 'block';
  }

  /** @param {Element | ShadowRoot} el */
  function hasBlockChild(el) {
    for (const child of el.children) {
      const kind = classify(child);
      if (kind === 'block' || kind === 'skip') return true;
      // A formula's own layout (stacked fractions, matrices) is made of blocks inside an inline box
      if (kind === 'inline' && !isMath(child) && hasBlockChild(child)) return true;
    }
    return false;
  }

  /** The page's header, footer, navigation or sidebar that el is part of. A header or footer of an article or
   * section is part of that (its title, byline).
   * @param {Element | ShadowRoot} el */
  function landmarkOf(el) {
    /** @param {Element} mark
     * @param {string} selector */
    const above = (mark, selector) => {
      const parent = parentOf(mark);
      return parent && closestComposed(parent, selector);
    };
    for (let mark = closestComposed(el, CHROME_SELECTOR); mark; mark = above(mark, CHROME_SELECTOR)) {
      if (!/^(header|footer)$/.test(mark.localName) || !above(mark, 'article, section')) return mark;
    }
    return null;
  }

  /** Whether el (an element or a shadow root) is part of what the user asked to translate (setting "scope").
   * @param {Element | ShadowRoot} el */
  function inScope(el) {
    if (scope !== 'main') return true;
    const main = closestComposed(el, MAIN_SELECTOR);
    if (main) {
      const aside = closestComposed(el, CHROME_IN_MAIN);
      return !aside || !containsComposed(main, aside);
    }
    // Pages that mark no main content: everything but the header, footer, navigation and sidebars
    return !pageHasMain && !landmarkOf(el);
  }

  /** Whether anything under el can be in scope; a wrapper outside it may still hold the main content.
   * @param {Element | ShadowRoot} el */
  const inScopeTree = (el) => inScope(el) || (pageHasMain && !!el.querySelector(MAIN_SELECTOR));

  /** Paragraphs under root (a block, an element the page added, or a shadow root), with those in the shadow roots
   * below it. A shadow root holds paragraphs of its own: its text is laid out apart from its host's children.
   * @param {Element | ShadowRoot} root
   * @returns {Unit[]} */
  function collect(root) {
    if (scope === 'main') pageHasMain = !!document.querySelector(MAIN_SELECTOR);
    /** @type {Paragraph[]} */
    const out = [];
    /** @type {ShadowRoot[]} */
    const shadows = [];
    /** Note the shadow root of el, scanned after this tree when el is a block (see classify); one under an opt-out
     * stays as it is.
     * @param {Element} el
     * @param {string} kind */
    const noteShadow = (el, kind) => {
      if (optedOut(el)) return;
      const shadow = shadowOf(el);
      if (shadow) {
        if (kind === 'block') shadows.push(shadow);
      } else if (el.matches(':not(:defined)') && !OWN_HOSTS.has(el.localName)) bareHosts.add(el);
    };
    /** @type {ChildNode[]} */
    let run = [];
    let breaks = 0; // <br> in a row since the last text
    const flush = () => {
      // Line breaks at the end belong to the layout, not to the paragraph
      while (run.length && isBreak(run[run.length - 1])) run.pop();
      const hosts = run.length ? hostsAlone(run) : null;
      // Each host was found by shadowOf
      if (hosts) shadows.push(...hosts.map((host) => /** @type {ShadowRoot} */ (shadowOf(host))));
      else if (run.length) {
        const last = run[run.length - 1];
        // The nodes of a run are children of the element or shadow root being visited
        const parent = /** @type {Element | ShadowRoot} */ (last.parentNode);
        out.push({ nodes: run, parent, after: last.nextSibling, text: '', payload: '', rich: null });
      }
      run = [];
      breaks = 0;
    };
    /** @param {ChildNode} node */
    const visit = (node) => {
      if (isText(node)) {
        // Emptied by translation-only mode: still part of its paragraph
        if (run.length || node.data.trim() || originals.has(node)) run.push(node);
        if (node.data.trim()) breaks = 0;
        return;
      }
      if (!isElement(node)) return;
      if (node.localName === 'br') {
        // A blank line (<br><br>) separates paragraphs, as in poems and old forum posts
        if (++breaks === 2) flush();
        else if (run.length) run.push(node);
        return;
      }
      const kind = classify(node);
      if (kind === 'ignore') return;
      breaks = 0;
      if (kind !== 'skip') noteShadow(node, kind);
      // An inline element is taken whole, unless it wraps blocks (a card inside a link): then its children are
      // split like a block's
      if (kind === 'inline' && (isMath(node) || !hasBlockChild(node))) {
        run.push(node);
        // Custom elements in it defined later may change how it is read (see rescan)
        for (const el of node.querySelectorAll(':not(:defined)')) noteShadow(el, kind);
        return;
      }
      flush();
      // A host's own children show only at a slot of its shadow root
      if (kind !== 'skip' && !slotless(node) && inScopeTree(node)) node.childNodes.forEach(visit);
      flush();
    };
    if (inScopeTree(root)) {
      // A host scanned on its own (added to body, or its shadow root came late) stands alone
      if (isElement(root)) noteShadow(root, 'block');
      if (!isElement(root) || !slotless(root)) root.childNodes.forEach(visit);
    }
    flush();
    shadows.forEach(enterShadow);
    return [...out.filter((unit) => inScope(unit.parent)), ...placeholders(root), ...shadows.flatMap(collect)];
  }

  /** A text node's text, also while translation-only mode keeps it empty.
   * @param {Text} node */
  const dataOf = (node) => (node.data === '' ? originals.get(node) ?? '' : node.data);

  /** @param {Node} node */
  const isBreak = (node) => isElement(node) && node.localName === 'br';

  /** The text of a paragraph, without what classify ignores (hidden labels, icon names) and without formulas, whose
   * glyphs are no words to translate. code: with the text of inline code (see CODE_TAGS).
   * @param {Node[]} nodes */
  function textOf(nodes, code = true) {
    let text = '';
    /** @param {Node} node */
    const walk = (node) => {
      if (isText(node)) text += dataOf(node);
      else if (isBreak(node)) text += ' ';
      else if (!isElement(node) || isMath(node) || (!code && CODE_TAGS.has(node.localName))) return;
      else if (classify(node) !== 'ignore') node.childNodes.forEach(walk);
    };
    nodes.forEach(walk);
    return text.replace(/\s+/g, ' ').trim();
  }

  /** The shadow hosts copied whole in a run that has no text of its own to translate (a date alone in a table cell):
   * not part of a sentence, so their shadow roots are translated instead. Null when the run is a paragraph.
   * @param {Node[]} nodes */
  function hostsAlone(nodes) {
    /** @type {Element[]} */
    const hosts = [];
    let text = '';
    /** @param {Node} node */
    const walk = (node) => {
      if (isText(node)) text += dataOf(node);
      else if (!isElement(node) || optedOut(node) || isMath(node) || classify(node) === 'ignore') return;
      else if (slotless(node)) hosts.push(node);
      else node.childNodes.forEach(walk);
    };
    nodes.forEach(walk);
    return hosts.length && !worth(text) ? hosts : null;
  }

  /** @param {string} text */
  function worth(text) {
    if (/^(https?:\/\/|www\.)\S+$/i.test(text)) return false;
    return (text.match(/\p{L}/gu) || []).length >= 2;
  }

  const MAX_MATH_TEXT = 200;

  /** What stands for a formula in the text sent: its TeX source when the page has it, so the model reads the
   * sentence with the formula in it. The formula itself is copied from the page (see buildRich).
   * @param {Element} el */
  function mathText(el) {
    const source =
      el.querySelector('annotation[encoding="application/x-tex"]')?.textContent ||
      (el.localName === 'math' ? el : el.querySelector('math'))?.getAttribute('alttext') ||
      el.querySelector('img[alt]')?.getAttribute('alt') ||
      el.textContent ||
      '';
    const text = source.replace(/\s+/g, ' ').trim();
    return text.length > MAX_MATH_TEXT ? `${text.slice(0, MAX_MATH_TEXT)}…` : text;
  }

  /** @type {Record<string, string>} */
  const ENTITIES ={ '&': 'amp', '<': 'lt', '>': 'gt', '"': 'quot' };
  /** @param {string} s */
  const escapeHTML = (s) => s.replace(/[&<>"]/g, (c) => `&${ENTITIES[c]};`);

  /** The paragraph as HTML keeping only RICH_TAGS and what is copied whole (as <span translate="no">, which engines
   * leave alone), with those elements in order of their i; null when it has none.
   * @param {Node[]} nodes
   * @returns {Rich | null} */
  function richOf(nodes) {
    /** @type {Element[]} */
    const refs = [];
    /** @param {Node} node
     * @returns {string} */
    const walk = (node) => {
      if (isText(node)) return escapeHTML(dataOf(node));
      if (isBreak(node)) return ' ';
      if (!isElement(node) || classify(node) === 'ignore') return '';
      if (keptWhole(node)) {
        refs.push(node);
        const text = isMath(node) ? mathText(node) : textOf([node]);
        return `<span translate="no" i="${refs.length - 1}">${escapeHTML(text)}</span>`;
      }
      const inner = [...node.childNodes].map(walk).join('');
      if (!RICH_TAGS.has(node.localName)) return inner;
      refs.push(node);
      return `<${node.localName} i="${refs.length - 1}">${inner}</${node.localName}>`;
    };
    const html = nodes.map(walk).join('').replace(/\s+/g, ' ').trim();
    return refs.length ? { html, refs } : null;
  }

  /** Rebuild a translated HTML fragment with DOM APIs. The model decides only the text and where the paragraph's own
   * tags go: every element is a copy of the original element it stands for (matched by i, else by tag in order),
   * so attributes such as href come from the page, never from the model. What is copied whole (see keptWhole) comes
   * from the page, whatever the model put inside. Other tags keep only their text.
   * @param {string} translated
   * @param {Element[]} refs */
  function buildRich(translated, refs) {
    // DOMParser documents are inert: no scripts run and no resources load
    const doc = new DOMParser().parseFromString(`<body>${translated}`, 'text/html');
    /** @type {Set<Element>} */
    const used = new Set();
    /** @param {Element} el */
    const original = (el) => {
      const i = el.getAttribute('i');
      const ref = i === null ? null : refs[Number(i)];
      if (ref && !used.has(ref) && (keptWhole(ref) ? el.localName === 'span' : ref.localName === el.localName)) {
        return ref;
      }
      if (!RICH_TAGS.has(el.localName)) return null;
      return refs.find((r) => r.localName === el.localName && !keptWhole(r) && !used.has(r));
    };
    /** @param {Node} node
     * @param {Element | DocumentFragment} into */
    const build = (node, into) => {
      for (const child of node.childNodes) {
        if (isText(child)) {
          into.append(child.data);
        } else if (isElement(child) && !IGNORED_TAGS.has(child.localName)) {
          const ref = original(child);
          if (!ref) {
            build(child, into);
            continue;
          }
          used.add(ref);
          const kept = keptWhole(ref);
          const copy = /** @type {Element} */ (ref.cloneNode(kept));
          // The original keeps its ids; marks of this extension describe the original only
          for (const el of [copy, ...(kept ? copy.querySelectorAll('*') : [])]) {
            for (const name of el.getAttributeNames()) {
              if (name === 'id' || name === 'name' || name.startsWith('data-dt-')) el.removeAttribute(name);
            }
          }
          if (!kept) build(child, copy);
          into.append(copy);
        }
      }
    };
    const out = document.createDocumentFragment();
    build(doc.body, out);
    return out;
  }

  /** Input fields in root (or root itself) whose placeholder is collected along with the paragraphs.
   * @param {Element | ShadowRoot} root
   * @returns {AttrUnit[]} */
  function placeholders(root) {
    const selector = 'input[placeholder], textarea[placeholder]';
    const fields = [...root.querySelectorAll(selector), ...(isElement(root) && root.matches(selector) ? [root] : [])];
    return fields
      .filter((el) => {
        const opt = closestComposed(el, OPT_OUT); // one on body or html does not count, see optedOut
        return (!opt || opt === document.body || opt === document.documentElement) && inScope(el);
      })
      .map((target) => ({ attr: 'placeholder', target, text: '', payload: '' }));
  }

  /** Paragraphs and attributes still on the page.
   * @param {Unit} unit */
  const alive = (unit) => (unit.attr ? unit.target.isConnected : unit.nodes[0].isConnected);
  /** Still on the page and not replaced by a newer version.
   * @template {Unit} T
   * @param {T | undefined} unit
   * @returns {unit is T} */
  const current = (unit) => !!unit && !unit.stale && alive(unit);
  /** A translation no current paragraph owns: left by an earlier version of one, or by one the page removed.
   * @param {ChildNode | null | undefined} el
   * @returns {el is HTMLElement} */
  const free = (el) =>
    !!el && isElement(el) && el.localName === TRANS_TAG && el.isConnected && !current(ownerOf.get(el));

  /** Whether a run found again is the paragraph collected before, unchanged. One marked redo is read again anyway:
   * a custom element in it was defined since.
   * @param {Paragraph} old
   * @param {Paragraph} unit */
  const sameUnit = (old, unit) =>
    !old.redo && old.nodes.length === unit.nodes.length && old.nodes.every((node, i) => node === unit.nodes[i]) &&
    textOf(unit.nodes) === old.text;

  /** Translations a rescan may remove, unless a paragraph found again takes them over.
   * @type {Set<Element>} */
  const reused = new Set();

  /** @param {Unit[]} units */
  function register(units) {
    for (const unit of units) {
      if (unit.attr) registerAttr(unit);
      else registerParagraph(unit);
    }
  }

  /** @param {AttrUnit} unit */
  function registerAttr(unit) {
    if (seenAttrs[unit.attr].has(unit.target)) return;
    seenAttrs[unit.attr].add(unit.target);
    // Collected for having the attribute (see placeholders, onPointerOver)
    unit.text = unit.payload = (unit.target.getAttribute(unit.attr) ?? '').trim();
    if (!worth(unit.text) || unit.text.length > MAX_TEXT) return;
    if (unit.attr === 'title') queue.push(unit); // hovered, so in view already
    else watch(unit, unit.target);
  }

  /** @param {Paragraph} unit */
  function registerParagraph(unit) {
    const olds = new Set(unit.nodes.map((node) => unitOf.get(node)).filter(current));
    if (olds.size === 1 && sameUnit([...olds][0], unit)) return;
    // Text was added to, changed in or removed from a paragraph collected before: this one takes its place, and
    // its translation once translated
    for (const old of olds) retire(old);
    unit.reuse = [...[...olds].map((old) => old.el), unit.after].find(free);
    unit.text = textOf(unit.nodes);
    // Not recorded when not worth translating: text added later may make it worth it. Inline code alone (an option
    // name in a table cell) comes back unchanged anyway
    const wanted = worth(unit.text) && worth(textOf(unit.nodes, false));
    if (wanted) unit.nodes.forEach((node) => unitOf.set(node, unit));
    if (wanted && (churn.get(unit.parent) || 0) < MAX_CHURN) {
      unit.rich = richOf(unit.nodes);
      // What is sent: the HTML when the paragraph has inline formatting, else its text. Longer ones are dropped:
      // the server takes at most MAX_TEXT characters
      unit.payload = unit.rich ? unit.rich.html : unit.text;
    }
    if (!unit.payload || unit.payload.length > MAX_TEXT) {
      // The old version's translation would stay next to a text it no longer matches
      unit.reuse?.remove();
      return;
    }
    if (unit.reuse) reused.add(unit.reuse);
    // A changed paragraph waits until the page stops changing it
    const observed = boxOf(elementOf(unit.parent)); // text nodes cannot be observed; their parent element is
    if (olds.size) setTimeout(() => current(unit) && watch(unit, observed), SETTLE_DELAY);
    else watch(unit, observed);
  }

  /** @param {Unit} unit
   * @param {Element} observed */
  function watch(unit, observed) {
    let units = waiting.get(observed);
    if (!units) {
      waiting.set(observed, (units = []));
      visibility.observe(observed);
    }
    units.push(unit);
  }

  /** The nearest element that has a box: display:contents elements never intersect the viewport.
   * @param {Element} el */
  function boxOf(el) {
    let up;
    while ((up = parentOf(el)) && getComputedStyle(el).display === 'contents') el = up;
    return el;
  }

  /** A paragraph is no longer current: its original is shown again, and a newer version may take its translation.
   * @param {Unit} unit */
  function retire(unit) {
    if (shown.has(unit) && !unit.attr) churn.set(unit.parent, (churn.get(unit.parent) || 0) + 1);
    unit.stale = true;
    shown.delete(unit);
    if (!unit.attr) setOriginalVisible(unit, true, true);
  }

  // ---------- Translating ----------

  // Queued paragraphs farther than this many screens from the viewport (the page was scrolled on before their turn)
  // wait again until they come near; they entered the queue within one screen (see visibility)
  const FAR_SCREENS = 2;

  /** How far the unit is from the viewport, in pixels: 0 when any of it is in view, Infinity when it has no box.
   * @param {Unit} unit */
  function distance(unit) {
    let box;
    if (unit.attr) box = unit.target.getBoundingClientRect();
    else {
      // The paragraph's own text, rather than its parent, which may hold many paragraphs. The page may have removed
      // some of its nodes meanwhile
      const nodes = unit.nodes.filter((node) => node.parentNode === unit.parent);
      const range = document.createRange();
      if (nodes.length) {
        range.setStartBefore(nodes[0]);
        range.setEndAfter(nodes[nodes.length - 1]);
      }
      box = range.getBoundingClientRect();
      if (!box.width && !box.height) box = boxOf(elementOf(unit.parent)).getBoundingClientRect();
    }
    if (!box.width && !box.height) return Infinity;
    if (box.bottom < 0) return -box.bottom;
    return Math.max(0, box.top - innerHeight);
  }

  /** Put what is in view first, then the nearest, and send the paragraphs the page was scrolled away from back to
   * wait for the viewport. Translations then follow the reader instead of the order paragraphs were found in. */
  function prioritize() {
    const far = FAR_SCREENS * innerHeight;
    /** @type {{ unit: Unit, d: number }[]} */
    const near = [];
    for (const unit of queue) {
      if (!current(unit)) continue; // removed or changed by the page before its turn
      const d = distance(unit);
      if (d <= far) near.push({ unit, d });
      else watch(unit, unit.attr ? unit.target : boxOf(elementOf(unit.parent)));
    }
    // A stable sort: paragraphs equally far, all those in view among them, keep the order of the page
    near.sort((a, b) => a.d - b.d);
    queue.splice(0, queue.length, ...near.map(({ unit }) => unit));
  }

  function nextBatch() {
    /** @type {Unit[]} */
    const batch = [];
    let size = 0;
    while (queue.length && batch.length < BATCH_ITEMS) {
      const unit = queue[0];
      if (!current(unit)) {
        queue.shift(); // removed or changed by the page before its turn
        continue;
      }
      if (batch.length && size + unit.payload.length > BATCH_CHARS) break;
      queue.shift();
      batch.push(unit);
      size += unit.payload.length;
    }
    return batch;
  }

  function pump() {
    if (!started || hidden || failed.length || inflight >= MAX_INFLIGHT || !queue.length) return;
    prioritize();
    while (inflight < MAX_INFLIGHT && queue.length) {
      const batch = nextBatch();
      if (batch.length) send(batch);
    }
  }

  /** A reply the content script can use: one string or null per paragraph sent.
   * @param {unknown} result
   * @param {Unit[]} batch
   * @returns {result is { translations: (string | null)[], lang: string }} */
  function usable(result, batch) {
    if (typeof result !== 'object' || result === null) return false;
    const { translations: list, lang } = /** @type {{ translations?: unknown, lang?: unknown }} */ (result);
    return Array.isArray(list) && list.length === batch.length && typeof lang === 'string' &&
      list.every((t) => t === null || typeof t === 'string');
  }

  /** The error the service worker reported, or '' when there is none.
   * @param {unknown} result */
  const errorOf = (result) =>
    typeof result === 'object' && result !== null && 'error' in result && result.error ? String(result.error) : '';

  /** @param {Unit[]} batch */
  async function send(batch) {
    inflight++;
    toast('翻译中…', true);
    /** @type {TranslateMessage} */
    const message = {
      type: 'dt-translate',
      texts: batch.map((u) => u.payload),
      html: batch.map((u) => !u.attr && !!u.rich),
      title: document.title,
    };
    /** @type {unknown} */
    let result;
    try {
      result = await chrome.runtime.sendMessage(message);
    } catch {
      // The extension was reloaded or removed; the script left in this page can no longer reach it
      result = { error: '扩展已更新，请刷新页面后重试' };
    }
    inflight--;
    const error = errorOf(result);
    if (error || !usable(result, batch)) {
      failed.push(...batch);
      toast(`翻译失败：${error || '翻译服务返回的结果无法识别'}。按快捷键重试`, true);
      return;
    }
    const { translations, lang } = result;
    batch.forEach((unit, i) => show(unit, translations[i], lang));
    pump();
    if (!inflight && !failed.length && !hidden) toast('');
  }

  /** @param {Unit} unit
   * @param {string | null} translation
   * @param {string} lang */
  function show(unit, translation, lang) {
    if (!translation || !current(unit)) return;
    if (unit.attr) {
      // The page may have changed the attribute while the translation was on its way
      const value = unit.target.getAttribute(unit.attr);
      if (value?.trim() !== unit.text) return;
      unit.translation = translation;
      unit.current = value;
      shown.add(unit);
      setAttrVisible(unit, originalVisible());
      return;
    }
    const after = unit.after?.parentNode === unit.parent ? unit.after : null;
    // The translation of this paragraph's earlier version, or one the page left right after it; never one that
    // belongs to another current paragraph
    const el = [unit.reuse, after].find(free) || document.createElement(TRANS_TAG);
    el.setAttribute('lang', lang);
    el.setAttribute('translate', 'no');
    // Text at the top of a shadow root is laid out in its host
    const box = elementOf(unit.parent);
    // In a flex or grid row, or inside an inline box (a tag chip, a button-like link), a block below the original
    // would become a squeezed item beside it or break the line; such translations follow the original inline
    const display = getComputedStyle(box).display;
    el.toggleAttribute('data-dt-inline', /flex|grid/.test(display) || display.startsWith('inline'));
    el.setAttribute('data-dt-backdrop', darkBehind(box) ? 'dark' : 'light');
    // The paragraph's text can have another size than its parent (font-size: 0 on a parent hides the gaps between
    // inline blocks): the translation takes the size of the text
    const textEl = unit.nodes.find(isElement);
    const size = textEl && getComputedStyle(textEl).fontSize;
    el.style.fontSize = size && size !== getComputedStyle(box).fontSize ? size : '';
    // Among a shadow host's children, the translation goes into the slot its original is shown in
    const slot = textEl?.getAttribute('slot');
    if (slot) el.setAttribute('slot', slot);
    else el.removeAttribute('slot');
    if (unit.rich) el.replaceChildren(buildRich(translation, unit.rich.refs));
    else el.textContent = translation;
    if (el.parentNode !== unit.parent) unit.parent.insertBefore(el, after);
    styleShadow(el);
    unit.el = el;
    ownerOf.set(el, unit);
    shown.add(unit);
    setOriginalVisible(unit, originalVisible());
  }

  // ---------- Colors ----------

  // Translations are tinted so they can be told from the page's own text. Whether the tint is for a light or a dark
  // background is decided by what is really behind each one: the browser's dark mode says nothing about a site that
  // keeps its own colors, and a light page can still have a dark header
  /** @type {WeakMap<Element, boolean>} */
  const backdrops = new WeakMap(); // element -> whether its background is dark
  // A new canvas always has a 2d context to give
  const painter = /** @type {CanvasRenderingContext2D} */ (
    document.createElement('canvas').getContext('2d', { willReadFrequently: true })
  );

  /** [r, g, b, a] in 0-255 of a CSS color in any syntax (rgb, oklch, lab, color()), by painting one pixel. A color the
   * canvas cannot read comes out transparent, so the parent's background decides instead.
   * @param {string} color */
  function rgbaOf(color) {
    painter.clearRect(0, 0, 1, 1);
    painter.fillStyle = 'transparent';
    painter.fillStyle = color;
    painter.fillRect(0, 0, 1, 1);
    return painter.getImageData(0, 0, 1, 1).data;
  }

  /** @param {Element} el
   * @returns {boolean} */
  function darkBehind(el) {
    const known = backdrops.get(el);
    if (known !== undefined) return known;
    let dark;
    let parent;
    const [r8, g8, b8, a8] = rgbaOf(getComputedStyle(el).backgroundColor);
    if (a8 >= 128) {
      // Relative luminance of sRGB (WCAG)
      const [r, g, b] = [r8, g8, b8].map((c) => {
        const v = c / 255;
        return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
      });
      dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.18;
    } else if ((parent = parentOf(el))) {
      dark = darkBehind(parent);
    } else {
      // No background set anywhere: the canvas is dark only when the page allows a dark scheme and the browser uses one
      const scheme = getComputedStyle(el).colorScheme;
      dark = scheme.includes('dark') && (!scheme.includes('light') || matchMedia('(prefers-color-scheme: dark)').matches);
    }
    backdrops.set(el, dark);
    return dark;
  }

  // ---------- Translation-only mode ----------

  const originalVisible = () => hidden || mode !== 'translated';

  /** Hide or show the original of a translated paragraph. Text nodes are emptied in place rather than removed or
   * wrapped, so frameworks that later update or remove them still find them where they left them. forget: the
   * paragraph is being dropped, so nodes no longer on the page are forgotten too.
   * @param {Paragraph} unit
   * @param {boolean} visible */
  function setOriginalVisible(unit, visible, forget = false) {
    for (const node of unit.nodes) {
      if (isText(node)) {
        if (!visible && !originals.has(node)) {
          originals.set(node, node.data);
          ourWrites.add(node);
          node.data = '';
        } else if (visible && originals.has(node)) {
          // Text the page wrote since is newer than ours
          if (node.data === '') {
            ourWrites.add(node);
            node.data = originals.get(node) ?? '';
          }
          originals.delete(node);
        }
      } else if (isElement(node)) {
        // A formula is copied into the translation, images and all
        const keep = !isMath(node) && (node.matches(KEEP_SELECTOR) || node.querySelector(KEEP_SELECTOR));
        node.toggleAttribute('data-dt-original', !visible && !keep);
      }
    }
    if (forget) unit.nodes.forEach((node) => originals.delete(node));
  }

  /** Attributes cannot show two texts apart: bilingual mode joins them (tooltips on two lines).
   * @param {AttrUnit} unit
   * @param {boolean} originalShown */
  function setAttrVisible(unit, originalShown) {
    const { target, attr } = unit;
    if (target.getAttribute(attr) !== unit.current) {
      shown.delete(unit); // the page set its own value since; it is the page's now
      return;
    }
    if (!originalShown) unit.current = unit.translation ?? unit.text;
    else if (hidden) unit.current = unit.text;
    else unit.current = `${unit.text}${attr === 'title' ? '\n' : ' / '}${unit.translation}`;
    target.setAttribute(attr, unit.current);
  }

  function applyMode() {
    document.documentElement.toggleAttribute('data-dt-translated', mode === 'translated');
    for (const el of prune()) el.remove();
    for (const unit of shown) {
      if (unit.attr) setAttrVisible(unit, originalVisible());
      else setOriginalVisible(unit, originalVisible());
    }
  }

  /** Drop what the page removed, so that long sessions in single-page apps do not keep growing, and return the
   * translations it left behind. A paragraph whose translation the page removed while keeping the original gets its
   * original back and is translated again. */
  function prune() {
    /** @type {HTMLElement[]} */
    const orphans = [];
    for (const unit of shown) {
      // A paragraph is shown with its translation el
      if (alive(unit) && (unit.attr || unit.el?.isConnected)) continue;
      retire(unit);
      if (unit.attr) continue;
      if (alive(unit)) dirty.add(unit.parent);
      else if (unit.el?.isConnected) orphans.push(unit.el);
    }
    for (const [el, units] of waiting) {
      if (el.isConnected && units.some(current)) continue;
      visibility.unobserve(el);
      growth.unobserve(el);
      waiting.delete(el);
    }
    for (const node of originals.keys()) if (!node.isConnected) originals.delete(node);
    /** @type {Set<Node>[]} */
    const nodeSets = [shadowRoots, bareHosts];
    for (const set of nodeSets) for (const node of set) if (!node.isConnected) set.delete(node);
    if (dirty.size) scheduleRescan();
    return orphans;
  }

  // ---------- Content added or changed later (infinite scrolling, single-page apps) ----------

  /** @type {Set<Node>} */
  const dirty = new Set();
  /** @type {Set<Node>} */
  const ourWrites = new Set(); // text nodes this script just emptied or restored: their mutation records are ours
  let rescanTimer = 0;
  /** @type {HTMLElement | null} */
  let observedBody = null;

  const scheduleRescan = () => rescanTimer || (rescanTimer = setTimeout(rescan, RESCAN_DELAY));

  /** Nodes this script adds: translations, its own boxes and the styles in shadow roots.
   * @param {Node} node */
  const ours = (node) =>
    isElement(node) && (node.localName === TRANS_TAG || OWN_HOSTS.has(node.localName) || styleEls.has(node));

  const CHANGES = { childList: true, subtree: true, characterData: true };

  const mutations = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'characterData') {
        // At the top of a shadow root, a text node's parent is the root
        const parent = record.target.parentNode;
        if (!ourWrites.has(record.target) && parent && !(isElement(parent) && parent.closest(TRANS_TAG))) {
          dirty.add(parent);
        }
        continue;
      }
      // Our own insertions and their text
      if (isElement(record.target) && record.target.closest(TRANS_TAG)) continue;
      const added = [...record.addedNodes].filter((n) => !ours(n));
      if (record.target === document.body || record.target === document.documentElement) {
        // Content added to the page itself (popups, a replaced <body>): only the new content is scanned
        for (const node of added) dirty.add(node.nodeType === Node.ELEMENT_NODE ? node : record.target);
      } else if (added.length || record.removedNodes.length) {
        dirty.add(record.target);
      }
    }
    ourWrites.clear();
    if (dirty.size) scheduleRescan();
  });

  /** @param {HTMLElement} body */
  function watchBody(body) {
    observedBody = body;
    mutations.disconnect();
    mutations.observe(body, CHANGES);
    // Navigation that replaces <body> (Turbo and the like) is seen on its parent
    mutations.observe(document.documentElement, { childList: true });
    for (const root of shadowRoots) mutations.observe(root, CHANGES);
  }

  /** The nearest block around el, or the shadow root at the top of its tree.
   * @param {Element | ShadowRoot} el */
  function blockAncestor(el) {
    while (isElement(el) && el !== document.body && classify(el) !== 'block') {
      const up = el.parentElement || (isShadow(el.parentNode) ? el.parentNode : null);
      if (!up) break;
      el = up;
    }
    return el;
  }

  function rescan() {
    rescanTimer = 0;
    if (document.body && observedBody !== document.body) watchBody(document.body);
    const orphans = prune();
    // Custom elements defined since they were scanned. The paragraph one with a shadow root was read in is read
    // again: the host may now end it, or be copied whole into its translation
    for (const host of bareHosts) {
      if (!host.matches(':defined')) continue;
      bareHosts.delete(host);
      if (!shadowOf(host)) continue;
      /** @type {Node | null} */
      let node = host;
      while (node && !unitOf.has(node)) node = node.parentNode;
      const unit = node && unitOf.get(node);
      if (unit) unit.redo = true;
      const parent = parentOf(host);
      dirty.add(parent && parent !== document.body ? parent : host);
    }
    /** @type {Set<Element | ShadowRoot>} */
    const roots = new Set();
    for (const target of dirty) {
      if (!isElement(target) && !isShadow(target)) continue;
      if (!target.isConnected || target === document.documentElement) continue;
      // An element added to body is scanned on its own: scanning body again would scan the whole page
      const root = target.parentElement === document.body ? target : blockAncestor(target);
      // Inside a skipped block or a formula, or below an opt-out (one on body or html does not count, see optedOut)
      const skipped = closestComposed(root, `${SKIP_TAGS.join(',')}, ${MATH}, ${OPT_OUT}`);
      if (skipped && skipped !== document.body && skipped !== document.documentElement) continue;
      roots.add(root);
    }
    dirty.clear();
    reused.clear();
    // A root inside another root is scanned with it
    for (const root of roots) {
      if (![...roots].some((other) => other !== root && containsComposed(other, root))) register(collect(root));
    }
    for (const el of orphans) if (!reused.has(el)) el.remove();
  }

  // ---------- Tooltips (title) ----------

  // Titles are translated when hovered, not up front: most are never seen, and a page can have hundreds
  const HOVER_DELAY = 300;
  let hoverTimer = 0;

  /** @param {PointerEvent} e */
  function onPointerOver(e) {
    // e.target is the host for an element in a shadow root; the path starts at the element (in open roots only)
    const origin = /** @type {Node | undefined} */ (e.composedPath()[0]);
    const target = origin && isElement(origin) ? closestComposed(origin, '[title]') : null;
    clearTimeout(hoverTimer);
    if (!target || seenAttrs.title.has(target) || target.closest(TRANS_TAG) || !inScope(target)) return;
    const opt = closestComposed(target, OPT_OUT);
    if (opt && opt !== document.body && opt !== document.documentElement) return;
    hoverTimer = setTimeout(() => {
      /** @type {AttrUnit} */
      const unit = { attr: 'title', target, text: '', payload: '' };
      register([unit]);
      // register queued nothing when the title is not worth translating; otherwise send it now, it is being looked at
      if (queue.includes(unit)) {
        queue.splice(queue.indexOf(unit), 1);
        queue.unshift(unit);
        pump();
      }
    }, HOVER_DELAY);
  }

  // ---------- Status ----------

  /** @type {HTMLElement | null} */
  let toastBox = null;
  let toastTimer = 0;

  /** @param {string} message */
  function toast(message, sticky = false) {
    if (!IS_TOP) return; // one status box for the whole tab, shown by the top frame
    clearTimeout(toastTimer);
    if (!toastBox) {
      if (!message) return;
      const host = document.createElement('dt-toast');
      // A closed shadow root keeps the page's styles out of the status box
      const shadow = host.attachShadow({ mode: 'closed' });
      shadow.innerHTML = `<style>
        div { position: fixed; right: 16px; bottom: 16px; z-index: 2147483647; max-width: 360px;
          padding: 8px 12px; border-radius: 8px; background: rgba(32, 33, 36, .92); color: #fff;
          font: 13px/1.5 system-ui, sans-serif; box-shadow: 0 2px 8px rgba(0, 0, 0, .3); }
        div:empty { display: none; }
      </style><div role="status"></div>`;
      toastBox = /** @type {HTMLElement} */ (shadow.querySelector('div'));
      // Outside body, so the MutationObserver on body does not see it
      document.documentElement.append(host);
    }
    const box = toastBox;
    box.textContent = message;
    if (message && !sticky) toastTimer = setTimeout(() => (box.textContent = ''), 2000);
  }

  // ---------- Shortcut ----------

  /** @param {boolean} value */
  function setHidden(value) {
    hidden = value;
    document.documentElement.toggleAttribute('data-dt-hidden', value);
    applyMode();
  }

  /** @type {Promise<void> | null} */
  let starting = null; // the first press's setup; later presses wait for it

  async function start() {
    const stored = await chrome.storage.sync.get({ mode: 'bilingual', scope: 'page' });
    mode = stored.mode === 'translated' ? 'translated' : 'bilingual';
    scope = stored.scope === 'main' ? 'main' : 'page';
    started = true;
    applyMode();
    register(collect(document.body));
    watchBody(document.body);
    document.addEventListener('pointerover', onPointerOver, true);
    if (!waiting.size) toast('没有找到需要翻译的内容');
  }

  function retryFailed() {
    queue.unshift(...failed.filter(current));
    failed = [];
  }

  async function toggle() {
    if (!starting) {
      starting = start();
      try {
        await starting;
      } catch (e) {
        // The next press starts over (the page may have no <body> yet)
        starting = null;
        started = false;
        throw e;
      }
    } else {
      await starting;
      if (failed.length && !hidden) {
        retryFailed();
        toast('');
      } else {
        setHidden(!hidden);
        toast(hidden ? '已隐藏译文' : '已显示译文');
      }
    }
    pump();
  }

  /** An iframe follows the top frame: started when it is and not hidden, hidden when it is. Failures in iframes have
   * no status box of their own, so every press that leaves the page shown retries them.
   * @param {PageState} state */
  async function sync({ hidden: topHidden }) {
    if (!starting) {
      if (!topHidden) await toggle();
      return;
    }
    await starting;
    if (hidden !== topHidden) setHidden(topHidden);
    if (!topHidden && failed.length) retryFailed();
    pump();
  }

  /** Whatever goes wrong, the service worker gets an answer: an unanswered message would leave it waiting for good.
   * @param {ControlMessage} msg
   * @returns {Promise<PageState>} */
  async function handle(msg) {
    try {
      if (msg.type === 'dt-toggle') await toggle();
      // Switching the display mode on a page not translated yet starts translating, in the new mode
      else if (msg.type === 'dt-start' && !starting) await toggle();
      else if (msg.type === 'dt-sync') await sync(msg);
    } catch {
      toast('现在无法翻译这个页面，请稍后再按快捷键重试');
    }
    return { started, hidden };
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!['dt-toggle', 'dt-start', 'dt-sync'].includes(msg?.type)) return false;
    handle(msg).then(sendResponse);
    return true; // the state is sent once toggle() has applied it
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'sync') return;
    // A new scope applies to what is found from now on; what is translated stays
    if (changes.scope) scope = changes.scope.newValue === 'main' ? 'main' : 'page';
    if (!changes.mode) return;
    mode = changes.mode.newValue === 'translated' ? 'translated' : 'bilingual';
    if (!started) return;
    applyMode();
    toast(mode === 'translated' ? '仅显示译文' : '双语对照');
  });

  // ---------- Sites translated automatically ----------

  // autostart.js runs just before this file on the sites in the options page's list (sites.js): start as if the
  // shortcut had been pressed
  if (window.__docTranslatorAuto && !starting) handle({ type: 'dt-toggle' });
})();

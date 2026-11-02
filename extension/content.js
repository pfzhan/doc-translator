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
  // Never translated, and not part of the surrounding sentence either
  const IGNORED_TAGS = new Set([
    'script', 'style', 'noscript', 'template', 'link', 'meta', 'wbr', 'hr',
    'img', 'picture', 'video', 'audio', 'canvas', 'iframe', 'object', 'embed',
    'svg', 'math', 'input', 'textarea', 'select', 'button', 'option', TRANS_TAG,
  ]);
  // Blocks kept as they are. Inline code inside a sentence stays part of the sentence
  const SKIP_TAGS = ['pre', 'code', 'kbd', 'samp'];
  // Tags that only occur inside a sentence. Hidden ones are dropped instead of splitting the sentence around them
  const PHRASING_TAGS = new Set([
    'a', 'abbr', 'b', 'bdi', 'bdo', 'cite', 'data', 'dfn', 'em', 'font', 'i', 'label', 'mark', 'q', 's',
    'small', 'span', 'strong', 'sub', 'sup', 'time', 'u', 'var',
  ]);
  // Inline formatting kept in translations. A paragraph containing one is sent as HTML with these tags only, each
  // numbered (i="N") and without attributes; other inline wrappers (span, font...) are sent as their text
  const RICH_TAGS = new Set(['a', 'b', 'strong', 'i', 'em', 'u', 's', 'code', 'sub', 'sup', 'mark']);
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

  let started = false;
  let hidden = false;
  let mode = 'bilingual'; // or 'translated'
  let scope = 'page'; // or 'main'
  let pageHasMain = false; // whether the page marks its main content; updated by collect
  let failed = []; // paragraphs of failed requests; non-empty means the next press retries
  let inflight = 0;
  // A paragraph is a run of inline content with the same parent: {nodes, text, parent, after, el}. The translation
  // el goes before after (null: at the end of parent). An attribute (placeholder, title) is {attr, target, text}.
  // A paragraph the page changed later is stale: a new one takes its place
  const unitOf = new WeakMap(); // node of a collected paragraph -> that paragraph
  const ownerOf = new WeakMap(); // dt-trans -> the paragraph it translates
  const churn = new WeakMap(); // parent element -> how many of its translations were replaced
  const seenAttrs = { placeholder: new WeakSet(), title: new WeakSet() }; // elements whose attribute was collected
  const shown = new Set(); // paragraphs and attributes whose translation is on the page
  const originals = new Map(); // text node emptied in translation-only mode -> its text
  const waiting = new Map(); // observed element -> its paragraphs not yet near the viewport
  const queue = []; // paragraphs near the viewport, not sent yet

  const visibility = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        visibility.unobserve(entry.target);
        const { width, height } = entry.boundingClientRect;
        // Screen-reader-only text is clipped to a 1px box: translating it would only spend quota
        if (width > 1 && height > 1) queue.push(...(waiting.get(entry.target) || []).filter(current));
        waiting.delete(entry.target);
      }
      pump();
    },
    { rootMargin: '100% 0px' }, // one screen above and below
  );

  // ---------- Finding paragraphs ----------

  function optedOut(el) {
    // translate="no" on <html> or <body> is usually aimed at the browser's own translator; the user asked for this
    // page explicitly, so only opt-outs below body count
    return el !== document.body && el !== document.documentElement && el.matches(OPT_OUT);
  }

  /** ignore: contributes nothing; inline: part of the current paragraph; block: ends it; skip: ends it, not entered */
  function classify(el) {
    if (IGNORED_TAGS.has(el.localName)) return 'ignore';
    // Hidden by translation-only mode: still the inline element it was when collected
    if (el.hasAttribute('data-dt-original')) return 'inline';
    const display = getComputedStyle(el).display;
    if (display === 'none' && PHRASING_TAGS.has(el.localName)) return 'ignore';
    const inline = display.startsWith('inline') || display === 'contents';
    // Text marked not to translate stays in its sentence and comes back unchanged (see richOf); editable content
    // and marked blocks are left out
    if (optedOut(el)) return inline && !el.matches(EDITABLE) ? 'inline' : 'skip';
    // Icon fonts write their icon names as aria-hidden text
    if (inline) return el.getAttribute('aria-hidden') === 'true' ? 'ignore' : 'inline';
    if (SKIP_TAGS.includes(el.localName)) return 'skip';
    // display:none blocks are still collected: they are translated once shown, when they enter the viewport
    return 'block';
  }

  function hasBlockChild(el) {
    for (const child of el.children) {
      const kind = classify(child);
      if (kind === 'block' || kind === 'skip') return true;
      if (kind === 'inline' && hasBlockChild(child)) return true;
    }
    return false;
  }

  /** The page's header, footer, navigation or sidebar that el is part of. A header or footer of an article or
   * section is part of that (its title, byline). */
  function landmarkOf(el) {
    for (let mark = el.closest(CHROME_SELECTOR); mark; mark = mark.parentElement?.closest(CHROME_SELECTOR)) {
      if (!/^(header|footer)$/.test(mark.localName) || !mark.parentElement?.closest('article, section')) return mark;
    }
    return null;
  }

  /** Whether el is part of what the user asked to translate (setting "scope"). */
  function inScope(el) {
    if (scope !== 'main') return true;
    const main = el.closest(MAIN_SELECTOR);
    if (main) {
      const aside = el.closest(CHROME_IN_MAIN);
      return !aside || !main.contains(aside);
    }
    // Pages that mark no main content: everything but the header, footer, navigation and sidebars
    return !pageHasMain && !landmarkOf(el);
  }

  /** Whether anything under el can be in scope; a wrapper outside it may still hold the main content. */
  const inScopeTree = (el) => inScope(el) || (pageHasMain && !!el.querySelector(MAIN_SELECTOR));

  /** Paragraphs under root (a block, or an element the page added). Shadow DOM is not entered. */
  function collect(root) {
    if (scope === 'main') pageHasMain = !!document.querySelector(MAIN_SELECTOR);
    const out = [];
    let run = [];
    let breaks = 0; // <br> in a row since the last text
    const flush = () => {
      // Line breaks at the end belong to the layout, not to the paragraph
      while (run.length && run[run.length - 1].localName === 'br') run.pop();
      if (run.length) {
        const last = run[run.length - 1];
        out.push({ nodes: run, parent: last.parentNode, after: last.nextSibling });
      }
      run = [];
      breaks = 0;
    };
    const visit = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        // Emptied by translation-only mode: still part of its paragraph
        if (run.length || node.data.trim() || originals.has(node)) run.push(node);
        if (node.data.trim()) breaks = 0;
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      if (node.localName === 'br') {
        // A blank line (<br><br>) separates paragraphs, as in poems and old forum posts
        if (++breaks === 2) flush();
        else if (run.length) run.push(node);
        return;
      }
      const kind = classify(node);
      if (kind === 'ignore') return;
      breaks = 0;
      // An inline element is taken whole, unless it wraps blocks (a card inside a link): then its children are
      // split like a block's
      if (kind === 'inline' && !hasBlockChild(node)) {
        run.push(node);
        return;
      }
      flush();
      if (kind !== 'skip' && inScopeTree(node)) node.childNodes.forEach(visit);
      flush();
    };
    if (inScopeTree(root)) root.childNodes.forEach(visit);
    flush();
    return out.filter((unit) => inScope(unit.parent)).concat(placeholders(root));
  }

  /** A text node's text, also while translation-only mode keeps it empty. */
  const dataOf = (node) => (node.data === '' && originals.has(node) ? originals.get(node) : node.data);

  /** The text of a paragraph, without what classify ignores (hidden labels, icon names). */
  function textOf(nodes) {
    let text = '';
    const walk = (node) => {
      if (node.nodeType === Node.TEXT_NODE) text += dataOf(node);
      else if (node.localName === 'br') text += ' ';
      else if (node.nodeType === Node.ELEMENT_NODE && classify(node) !== 'ignore') node.childNodes.forEach(walk);
    };
    nodes.forEach(walk);
    return text.replace(/\s+/g, ' ').trim();
  }

  function worth(text) {
    if (/^(https?:\/\/|www\.)\S+$/i.test(text)) return false;
    return (text.match(/\p{L}/gu) || []).length >= 2;
  }

  const escapeHTML = (s) => s.replace(/[&<>"]/g, (c) => `&${{ '&': 'amp', '<': 'lt', '>': 'gt', '"': 'quot' }[c]};`);

  /** The paragraph as HTML keeping only RICH_TAGS and text marked not to translate (as <span translate="no">, which
   * engines leave alone), with those elements in order of their i; null when it has none. */
  function richOf(nodes) {
    const refs = [];
    const walk = (node) => {
      if (node.nodeType === Node.TEXT_NODE) return escapeHTML(dataOf(node));
      if (node.localName === 'br') return ' ';
      if (node.nodeType !== Node.ELEMENT_NODE || classify(node) === 'ignore') return '';
      if (optedOut(node)) {
        refs.push(node);
        return `<span translate="no" i="${refs.length - 1}">${escapeHTML(textOf([node]))}</span>`;
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
   * so attributes such as href come from the page, never from the model. Text marked not to translate is copied
   * whole from the page, whatever the model put inside. Other tags keep only their text. */
  function buildRich(translated, refs) {
    // DOMParser documents are inert: no scripts run and no resources load
    const doc = new DOMParser().parseFromString(`<body>${translated}`, 'text/html');
    const used = new Set();
    const original = (el) => {
      const i = el.getAttribute('i');
      const ref = i === null ? null : refs[Number(i)];
      if (ref && !used.has(ref) && (optedOut(ref) ? el.localName === 'span' : ref.localName === el.localName)) {
        return ref;
      }
      if (!RICH_TAGS.has(el.localName)) return null;
      return refs.find((r) => r.localName === el.localName && !optedOut(r) && !used.has(r));
    };
    const build = (node, into) => {
      for (const child of node.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) {
          into.append(child.data);
        } else if (child.nodeType === Node.ELEMENT_NODE && !IGNORED_TAGS.has(child.localName)) {
          const ref = original(child);
          if (!ref) {
            build(child, into);
            continue;
          }
          used.add(ref);
          const kept = optedOut(ref);
          const copy = ref.cloneNode(kept);
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

  /** Input fields in root (or root itself) whose placeholder is collected along with the paragraphs. */
  function placeholders(root) {
    const selector = 'input[placeholder], textarea[placeholder]';
    const fields = [...root.querySelectorAll(selector), ...(root.matches(selector) ? [root] : [])];
    return fields
      .filter((el) => {
        const opt = el.closest(OPT_OUT); // one on body or html does not count, see optedOut
        return (!opt || opt === document.body || opt === document.documentElement) && inScope(el);
      })
      .map((target) => ({ attr: 'placeholder', target }));
  }

  /** Paragraphs and attributes still on the page. */
  const alive = (unit) => (unit.attr ? unit.target.isConnected : unit.nodes[0].isConnected);
  /** Still on the page and not replaced by a newer version. */
  const current = (unit) => !!unit && !unit.stale && alive(unit);
  /** A translation no current paragraph owns: left by an earlier version of one, or by one the page removed. */
  const free = (el) => el?.localName === TRANS_TAG && el.isConnected && !current(ownerOf.get(el));

  /** Whether a run found again is the paragraph collected before, unchanged. */
  const sameUnit = (old, unit) =>
    old.nodes.length === unit.nodes.length && old.nodes.every((node, i) => node === unit.nodes[i]) &&
    textOf(unit.nodes) === old.text;

  /** Translations a rescan may remove, unless a paragraph found again takes them over. */
  const reused = new Set();

  function register(units) {
    for (const unit of units) {
      if (unit.attr) registerAttr(unit);
      else registerParagraph(unit);
    }
  }

  function registerAttr(unit) {
    if (seenAttrs[unit.attr].has(unit.target)) return;
    seenAttrs[unit.attr].add(unit.target);
    unit.text = unit.payload = unit.target.getAttribute(unit.attr).trim();
    if (!worth(unit.text) || unit.text.length > MAX_TEXT) return;
    if (unit.attr === 'title') queue.push(unit); // hovered, so in view already
    else watch(unit, unit.target);
  }

  function registerParagraph(unit) {
    const olds = new Set(unit.nodes.map((node) => unitOf.get(node)).filter(current));
    if (olds.size === 1 && sameUnit([...olds][0], unit)) return;
    // Text was added to, changed in or removed from a paragraph collected before: this one takes its place, and
    // its translation once translated
    for (const old of olds) retire(old);
    unit.reuse = [...[...olds].map((old) => old.el), unit.after].find(free);
    unit.text = textOf(unit.nodes);
    // Not recorded when not worth translating: text added later may make it worth it
    const wanted = worth(unit.text);
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
    const observed = boxOf(unit.parent); // text nodes cannot be observed; their parent element is
    if (olds.size) setTimeout(() => current(unit) && watch(unit, observed), SETTLE_DELAY);
    else watch(unit, observed);
  }

  function watch(unit, observed) {
    if (!waiting.has(observed)) {
      waiting.set(observed, []);
      visibility.observe(observed);
    }
    waiting.get(observed).push(unit);
  }

  /** The nearest element that has a box: display:contents elements never intersect the viewport. */
  function boxOf(el) {
    while (el.parentElement && getComputedStyle(el).display === 'contents') el = el.parentElement;
    return el;
  }

  /** A paragraph is no longer current: its original is shown again, and a newer version may take its translation. */
  function retire(unit) {
    if (shown.has(unit) && !unit.attr) churn.set(unit.parent, (churn.get(unit.parent) || 0) + 1);
    unit.stale = true;
    shown.delete(unit);
    if (!unit.attr) setOriginalVisible(unit, true, true);
  }

  // ---------- Translating ----------

  function nextBatch() {
    const batch = [];
    let size = 0;
    while (queue.length && batch.length < BATCH_ITEMS) {
      const unit = queue[0];
      if (!current(unit)) {
        queue.shift(); // removed or changed by the page before its turn
        continue;
      }
      if (batch.length && size + unit.payload.length > BATCH_CHARS) break;
      batch.push(queue.shift());
      size += unit.payload.length;
    }
    return batch;
  }

  function pump() {
    while (started && !hidden && !failed.length && inflight < MAX_INFLIGHT && queue.length) {
      const batch = nextBatch();
      if (batch.length) send(batch);
    }
  }

  /** A reply the content script can use: one string or null per paragraph sent. */
  function usable(result, batch) {
    const list = result?.translations;
    return Array.isArray(list) && list.length === batch.length && typeof result.lang === 'string' &&
      list.every((t) => t === null || typeof t === 'string');
  }

  async function send(batch) {
    inflight++;
    toast('翻译中…', true);
    let result;
    try {
      result = await chrome.runtime.sendMessage({
        type: 'dt-translate',
        texts: batch.map((u) => u.payload),
        html: batch.map((u) => !!u.rich),
        title: document.title,
      });
    } catch {
      // The extension was reloaded or removed; the script left in this page can no longer reach it
      result = { error: '扩展已更新，请刷新页面后重试' };
    }
    inflight--;
    if (!result?.error && !usable(result, batch)) result = { error: '翻译服务返回的结果无法识别' };
    if (result.error) {
      failed.push(...batch);
      toast(`翻译失败：${result.error}。按快捷键重试`, true);
      return;
    }
    batch.forEach((unit, i) => show(unit, result.translations[i], result.lang));
    pump();
    if (!inflight && !failed.length && !hidden) toast('');
  }

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
    // In a flex or grid row, or inside an inline box (a tag chip, a button-like link), a block below the original
    // would become a squeezed item beside it or break the line; such translations follow the original inline
    const display = getComputedStyle(unit.parent).display;
    el.toggleAttribute('data-dt-inline', /flex|grid/.test(display) || display.startsWith('inline'));
    el.setAttribute('data-dt-backdrop', darkBehind(unit.parent) ? 'dark' : 'light');
    // The paragraph's text can have another size than its parent (font-size: 0 on a parent hides the gaps between
    // inline blocks): the translation takes the size of the text
    const textEl = unit.nodes.find((n) => n.nodeType === Node.ELEMENT_NODE);
    const size = textEl && getComputedStyle(textEl).fontSize;
    el.style.fontSize = size && size !== getComputedStyle(unit.parent).fontSize ? size : '';
    if (unit.rich) el.replaceChildren(buildRich(translation, unit.rich.refs));
    else el.textContent = translation;
    if (el.parentNode !== unit.parent) unit.parent.insertBefore(el, after);
    unit.el = el;
    ownerOf.set(el, unit);
    shown.add(unit);
    setOriginalVisible(unit, originalVisible());
  }

  // ---------- Colors ----------

  // Translations are tinted so they can be told from the page's own text. Whether the tint is for a light or a dark
  // background is decided by what is really behind each one: the browser's dark mode says nothing about a site that
  // keeps its own colors, and a light page can still have a dark header
  const backdrops = new WeakMap(); // element -> whether its background is dark
  const painter = document.createElement('canvas').getContext('2d', { willReadFrequently: true });

  /** [r, g, b, a] in 0-255 of a CSS color in any syntax (rgb, oklch, lab, color()), by painting one pixel. A color the
   * canvas cannot read comes out transparent, so the parent's background decides instead. */
  function rgbaOf(color) {
    painter.clearRect(0, 0, 1, 1);
    painter.fillStyle = 'transparent';
    painter.fillStyle = color;
    painter.fillRect(0, 0, 1, 1);
    return painter.getImageData(0, 0, 1, 1).data;
  }

  function darkBehind(el) {
    if (backdrops.has(el)) return backdrops.get(el);
    let dark;
    const [r8, g8, b8, a8] = rgbaOf(getComputedStyle(el).backgroundColor);
    if (a8 >= 128) {
      // Relative luminance of sRGB (WCAG)
      const [r, g, b] = [r8, g8, b8].map((c) => {
        const v = c / 255;
        return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
      });
      dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.18;
    } else if (el.parentElement) {
      dark = darkBehind(el.parentElement);
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
   * paragraph is being dropped, so nodes no longer on the page are forgotten too. */
  function setOriginalVisible(unit, visible, forget = false) {
    for (const node of unit.nodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        if (!visible && !originals.has(node)) {
          originals.set(node, node.data);
          ourWrites.add(node);
          node.data = '';
        } else if (visible && originals.has(node)) {
          // Text the page wrote since is newer than ours
          if (node.data === '') {
            ourWrites.add(node);
            node.data = originals.get(node);
          }
          originals.delete(node);
        }
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        const keep = node.matches(KEEP_SELECTOR) || node.querySelector(KEEP_SELECTOR);
        node.toggleAttribute('data-dt-original', !visible && !keep);
      }
    }
    if (forget) unit.nodes.forEach((node) => originals.delete(node));
  }

  /** Attributes cannot show two texts apart: bilingual mode joins them (tooltips on two lines). */
  function setAttrVisible(unit, originalShown) {
    const { target, attr } = unit;
    if (target.getAttribute(attr) !== unit.current) {
      shown.delete(unit); // the page set its own value since; it is the page's now
      return;
    }
    if (!originalShown) unit.current = unit.translation;
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
    const orphans = [];
    for (const unit of shown) {
      if (alive(unit) && (unit.attr || unit.el.isConnected)) continue;
      retire(unit);
      if (unit.attr) continue;
      if (alive(unit)) dirty.add(unit.parent);
      else if (unit.el.isConnected) orphans.push(unit.el);
    }
    for (const [el, units] of waiting) {
      if (el.isConnected && units.some(current)) continue;
      visibility.unobserve(el);
      waiting.delete(el);
    }
    for (const node of originals.keys()) if (!node.isConnected) originals.delete(node);
    if (dirty.size) scheduleRescan();
    return orphans;
  }

  // ---------- Content added or changed later (infinite scrolling, single-page apps) ----------

  const dirty = new Set();
  const ourWrites = new Set(); // text nodes this script just emptied or restored: their mutation records are ours
  let rescanTimer = 0;
  let observedBody = null;

  const scheduleRescan = () => rescanTimer || (rescanTimer = setTimeout(rescan, RESCAN_DELAY));

  const mutations = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'characterData') {
        const parent = record.target.parentElement;
        if (!ourWrites.has(record.target) && parent && !parent.closest(TRANS_TAG)) dirty.add(parent);
        continue;
      }
      // Our own insertions and their text
      if (record.target.closest?.(TRANS_TAG)) continue;
      const added = [...record.addedNodes].filter((n) => n.localName !== TRANS_TAG && n.localName !== 'dt-toast');
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

  function watchBody() {
    observedBody = document.body;
    mutations.disconnect();
    mutations.observe(document.body, { childList: true, subtree: true, characterData: true });
    // Navigation that replaces <body> (Turbo and the like) is seen on its parent
    mutations.observe(document.documentElement, { childList: true });
  }

  function blockAncestor(el) {
    while (el !== document.body && el.parentElement && classify(el) !== 'block') el = el.parentElement;
    return el;
  }

  function rescan() {
    rescanTimer = 0;
    if (document.body && observedBody !== document.body) watchBody();
    const orphans = prune();
    const roots = new Set();
    for (const target of dirty) {
      if (target.nodeType !== Node.ELEMENT_NODE || !target.isConnected || target === document.documentElement) continue;
      // An element added to body is scanned on its own: scanning body again would scan the whole page
      const root = target.parentElement === document.body ? target : blockAncestor(target);
      // Inside a skipped block, or below an opt-out (one on body or html does not count, see optedOut)
      const skipped = root.closest(`${SKIP_TAGS.join(',')}, ${OPT_OUT}`);
      if (skipped && skipped !== document.body && skipped !== document.documentElement) continue;
      roots.add(root);
    }
    dirty.clear();
    reused.clear();
    // A root inside another root is scanned with it
    for (const root of roots) {
      if (![...roots].some((other) => other !== root && other.contains(root))) register(collect(root));
    }
    for (const el of orphans) if (!reused.has(el)) el.remove();
  }

  // ---------- Tooltips (title) ----------

  // Titles are translated when hovered, not up front: most are never seen, and a page can have hundreds
  const HOVER_DELAY = 300;
  let hoverTimer = 0;

  function onPointerOver(e) {
    const target = e.target.closest?.('[title]');
    clearTimeout(hoverTimer);
    if (!target || seenAttrs.title.has(target) || target.closest(TRANS_TAG) || !inScope(target)) return;
    const opt = target.closest(OPT_OUT);
    if (opt && opt !== document.body && opt !== document.documentElement) return;
    hoverTimer = setTimeout(() => {
      const unit = { attr: 'title', target };
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

  let toastBox = null;
  let toastTimer = 0;

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
      toastBox = shadow.querySelector('div');
      // Outside body, so the MutationObserver on body does not see it
      document.documentElement.append(host);
    }
    toastBox.textContent = message;
    if (message && !sticky) toastTimer = setTimeout(() => (toastBox.textContent = ''), 2000);
  }

  // ---------- Shortcut ----------

  function setHidden(value) {
    hidden = value;
    document.documentElement.toggleAttribute('data-dt-hidden', value);
    applyMode();
  }

  let starting = null; // the first press's setup; later presses wait for it

  async function start() {
    ({ mode, scope } = await chrome.storage.sync.get({ mode: 'bilingual', scope: 'page' }));
    started = true;
    applyMode();
    register(collect(document.body));
    watchBody();
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
      await starting;
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
   * no status box of their own, so every press that leaves the page shown retries them. */
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

  async function handle(msg) {
    if (msg.type === 'dt-toggle') await toggle();
    // Switching the display mode on a page not translated yet starts translating, in the new mode
    else if (msg.type === 'dt-start' && !starting) await toggle();
    else if (msg.type === 'dt-sync') await sync(msg);
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
  if (window.__docTranslatorAuto && !starting) toggle();
})();

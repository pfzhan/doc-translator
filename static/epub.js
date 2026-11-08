"use strict";

// EPUB online preview: chapters are fetched from /api/jobs/{id}/epub/{file}/{member}, and images and styles follow
// by relative path

// The script's names are kept in a block: tsconfig.static.json checks it together with app.js as if the two shared
// one global scope, though each page loads only its own
{
  /**
   * The book's outline (book_outline in app/formats/epub.py): its title, the members in reading order (spine) and
   * the table of contents.
   * @typedef {{ label: string, path: string, anchor: string, depth: number }} TocEntry
   * @typedef {{ title: string, spine: string[], toc: TocEntry[] }} BookOutline
   */

  /**
   * The element with this id, which epub.html always has.
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

  const params = new URLSearchParams(location.search);
  const jobId = params.get("job") || "";
  const fileName = params.get("file") || "";
  const base = `/api/jobs/${encodeURIComponent(jobId)}/epub/${encodeURIComponent(fileName)}/`;
  const frame = byId("book-frame", HTMLIFrameElement);
  const toc = $("book-toc");
  /** @type {{ spine: string[], links: HTMLAnchorElement[], current: string }} */
  const state = { spine: [], links: [], current: "" };

  /**
   * @param {string} path
   * @param {string} [anchor]
   */
  function memberUrl(path, anchor) {
    return base + path.split("/").map(encodeURIComponent).join("/") + (anchor ? `#${anchor}` : "");
  }

  /** @param {string} url */
  function memberOf(url) {
    const path = new URL(url, location.href).pathname;
    return path.startsWith(base) ? path.slice(base.length).split("/").map(decodeURIComponent).join("/") : "";
  }

  /**
   * @param {string} path
   * @param {string} [anchor]
   */
  function open(path, anchor) {
    frame.src = memberUrl(path, anchor);
    if (window.matchMedia("(max-width: 720px)").matches) setToc(false);
  }

  /** @param {number} delta */
  function step(delta) {
    const index = state.spine.indexOf(state.current) + delta;
    if (index >= 0 && index < state.spine.length) open(state.spine[index]);
  }

  /** @param {boolean} shown */
  function setToc(shown) {
    document.body.classList.toggle("toc-open", shown);
    $("book-toc-toggle").setAttribute("aria-expanded", String(shown));
  }

  // Follow links clicked inside the book: highlight the current chapter and keep the position in the address bar,
  // so a reload returns to this chapter
  function onLoad() {
    const win = frame.contentWindow;
    if (!win) return;
    let path = "";
    try {
      path = memberOf(win.location.href);
    } catch {
      return;
    }
    if (!path) return;
    state.current = path;
    const index = state.spine.indexOf(path);
    byId("book-prev", HTMLButtonElement).disabled = index <= 0;
    byId("book-next", HTMLButtonElement).disabled = index < 0 || index >= state.spine.length - 1;
    $("book-meta").textContent =
      index >= 0 ? `${fileName} · 第 ${index + 1} / ${state.spine.length} 部分` : fileName;
    // Highlight the section when a section was clicked, otherwise the first entry of the chapter
    const anchor = decodeURIComponent(win.location.hash.slice(1));
    const here = state.links.filter((link) => link.dataset.path === path);
    const active = here.find((link) => link.dataset.anchor === anchor) || here[0] || null;
    for (const link of state.links) link.classList.remove("active");
    if (active) {
      active.classList.add("active");
      active.scrollIntoView({ block: "nearest" });
    }
    history.replaceState(null, "", `${location.pathname}${location.search}#${encodeURIComponent(path)}`);
    // Without margins from the book the text touches the window edge, so some padding is added, and images stay
    // within the page
    const doc = frame.contentDocument;
    if (doc && doc.head && !doc.getElementById("book-view-style")) {
      const style = doc.createElementNS("http://www.w3.org/1999/xhtml", "style");
      style.id = "book-view-style";
      style.textContent = "html{padding:0 16px}body{max-width:46em;margin-left:auto;margin-right:auto}img,svg{max-width:100%;height:auto}";
      doc.head.append(style);
    }
  }

  /** @param {string} message */
  function showError(message) {
    const box = $("book-error");
    box.textContent = message;
    box.hidden = false;
    frame.hidden = true;
    $("book-name").textContent = fileName || "EPUB 预览";
  }

  async function init() {
    if (!jobId || !fileName) {
      showError("链接里缺少任务或文件名。");
      return;
    }
    /** @type {BookOutline} */
    let outline;
    try {
      const resp = await fetch(base.slice(0, -1));
      if (!resp.ok) {
        /** @type {unknown} */
        const body = await resp.json().catch(() => ({}));
        const detail = typeof body === "object" && body !== null && "detail" in body ? body.detail : undefined;
        // As new Error itself would, a detail that is not a string is turned into one
        throw new Error(String(detail || `HTTP ${resp.status}`));
      }
      outline = /** @type {BookOutline} */ (await resp.json());
    } catch (err) {
      showError(`打不开这本书：${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    state.spine = outline.spine;
    const title = outline.title || fileName;
    document.title = `${title} · 预览`;
    $("book-name").textContent = title;
    // A book without a table of contents lists file names in reading order
    /** @type {TocEntry[]} */
    const entries = outline.toc.length
      ? outline.toc
      : outline.spine.map((path) => ({ label: path.split("/").pop() ?? "", path, anchor: "", depth: 0 }));
    for (const entry of entries) {
      const link = document.createElement("a");
      link.href = memberUrl(entry.path, entry.anchor);
      link.textContent = entry.label || entry.path.split("/").pop() || "";
      link.dataset.path = entry.path;
      link.dataset.anchor = entry.anchor;
      link.style.paddingLeft = `${12 + entry.depth * 14}px`;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        open(entry.path, entry.anchor);
      });
      toc.append(link);
      state.links.push(link);
    }
    frame.addEventListener("load", onLoad);
    const saved = decodeURIComponent(location.hash.slice(1));
    const start = state.spine.includes(saved) ? saved : state.spine[0];
    if (start) open(start);
    else showError("书里没有可显示的章节。");
  }

  $("book-prev").addEventListener("click", () => step(-1));
  $("book-next").addEventListener("click", () => step(1));
  $("book-toc-toggle").addEventListener("click", () =>
    setToc(!document.body.classList.contains("toc-open")));
  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === "ArrowLeft") step(-1);
    if (event.key === "ArrowRight") step(1);
  });
  init();
}

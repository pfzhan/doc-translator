"use strict";

// EPUB online preview: chapters are fetched from /api/jobs/{id}/epub/{file}/{member}, and images and styles follow
// by relative path
const params = new URLSearchParams(location.search);
const jobId = params.get("job") || "";
const fileName = params.get("file") || "";
const base = `/api/jobs/${encodeURIComponent(jobId)}/epub/${encodeURIComponent(fileName)}/`;
const frame = document.getElementById("book-frame");
const toc = document.getElementById("book-toc");
const state = { spine: [], links: [], current: "" };

function memberUrl(path, anchor) {
  return base + path.split("/").map(encodeURIComponent).join("/") + (anchor ? `#${anchor}` : "");
}

function memberOf(url) {
  const path = new URL(url, location.href).pathname;
  return path.startsWith(base) ? path.slice(base.length).split("/").map(decodeURIComponent).join("/") : "";
}

function open(path, anchor) {
  frame.src = memberUrl(path, anchor);
  if (window.matchMedia("(max-width: 720px)").matches) setToc(false);
}

function step(delta) {
  const index = state.spine.indexOf(state.current) + delta;
  if (index >= 0 && index < state.spine.length) open(state.spine[index]);
}

function setToc(shown) {
  document.body.classList.toggle("toc-open", shown);
  document.getElementById("book-toc-toggle").setAttribute("aria-expanded", String(shown));
}

// Follow links clicked inside the book: highlight the current chapter and keep the position in the address bar, so
// a reload returns to this chapter
function onLoad() {
  let path = "";
  try {
    path = memberOf(frame.contentWindow.location.href);
  } catch {
    return;
  }
  if (!path) return;
  state.current = path;
  const index = state.spine.indexOf(path);
  document.getElementById("book-prev").disabled = index <= 0;
  document.getElementById("book-next").disabled = index < 0 || index >= state.spine.length - 1;
  document.getElementById("book-meta").textContent =
    index >= 0 ? `${fileName} · 第 ${index + 1} / ${state.spine.length} 部分` : fileName;
  // Highlight the section when a section was clicked, otherwise the first entry of the chapter
  const anchor = decodeURIComponent(frame.contentWindow.location.hash.slice(1));
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

function showError(message) {
  const box = document.getElementById("book-error");
  box.textContent = message;
  box.hidden = false;
  frame.hidden = true;
  document.getElementById("book-name").textContent = fileName || "EPUB 预览";
}

async function init() {
  if (!jobId || !fileName) {
    showError("链接里缺少任务或文件名。");
    return;
  }
  let outline;
  try {
    const resp = await fetch(base.slice(0, -1));
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`);
    outline = await resp.json();
  } catch (err) {
    showError(`打不开这本书：${err.message}`);
    return;
  }
  state.spine = outline.spine;
  const title = outline.title || fileName;
  document.title = `${title} · 预览`;
  document.getElementById("book-name").textContent = title;
  // A book without a table of contents lists file names in reading order
  const entries = outline.toc.length
    ? outline.toc
    : outline.spine.map((path) => ({ label: path.split("/").pop(), path, anchor: "", depth: 0 }));
  for (const entry of entries) {
    const link = document.createElement("a");
    link.href = memberUrl(entry.path, entry.anchor);
    link.textContent = entry.label || entry.path.split("/").pop();
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

document.getElementById("book-prev").addEventListener("click", () => step(-1));
document.getElementById("book-next").addEventListener("click", () => step(1));
document.getElementById("book-toc-toggle").addEventListener("click", () =>
  setToc(!document.body.classList.contains("toc-open")));
document.addEventListener("keydown", (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
});
init();

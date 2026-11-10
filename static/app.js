/**
 * The element with this id, which index.html always has.
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
 * The first element under root matching the selector, which the page's own markup always has.
 * @param {ParentNode} root
 * @param {string} selector
 * @returns {HTMLElement}
 */
function query(root, selector) {
  const found = root.querySelector(selector);
  if (!(found instanceof HTMLElement)) throw new Error(`${selector} is missing`);
  return found;
}

/**
 * The nearest element from start up that matches the selector, which the page's own markup always has.
 * @param {Element} start
 * @param {string} selector
 * @returns {HTMLElement}
 */
function closest(start, selector) {
  const found = start.closest(selector);
  if (!(found instanceof HTMLElement)) throw new Error(`${selector} around #${start.id} is missing`);
  return found;
}

/**
 * All elements under root matching the selector, each of a kind the page's markup guarantees.
 * @template {Element} T
 * @param {ParentNode} root
 * @param {string} selector
 * @param {{ new (): T, name: string }} kind
 * @returns {T[]}
 */
function queryAll(root, selector, kind) {
  return [...root.querySelectorAll(selector)].map((found) => {
    if (!(found instanceof kind)) throw new Error(`${selector} matched a ${found.tagName}, not a ${kind.name}`);
    return found;
  });
}

/**
 * The control of this name in the form, which the page's markup always has.
 * @template {Element} T
 * @param {HTMLFormElement} form
 * @param {string} name
 * @param {{ new (): T, name: string }} kind
 * @returns {T}
 */
function field(form, name, kind) {
  const found = form.elements.namedItem(name);
  if (!(found instanceof kind)) throw new Error(`#${form.id} has no ${kind.name} named ${name}`);
  return found;
}

/**
 * The same for a control whose kind does not matter, only its value: a single field, or a group of radio buttons
 * (whose value is the checked one's).
 * @param {HTMLFormElement} form
 * @param {string} name
 * @returns {HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | RadioNodeList}
 */
function control(form, name) {
  const found = form.elements.namedItem(name);
  if (
    found instanceof HTMLInputElement || found instanceof HTMLSelectElement || found instanceof HTMLTextAreaElement ||
    found instanceof RadioNodeList
  ) {
    return found;
  }
  throw new Error(`#${form.id} has no field named ${name}`);
}

/**
 * The text of a caught error. What the page throws and catches is always an Error; anything else is shown as is.
 * @param {unknown} err
 * @returns {string}
 */
function messageOf(err) {
  return err instanceof Error ? err.message : String(err);
}

// ---------- The server's answers ----------
// The JSON shapes this page reads, from app/history.py, app/services.py, app/settings.py and app/routes/*.py. Only
// the fields the page uses are listed. The server is this app's own, so its answers are trusted: each call of api()
// states the shape its route returns

/**
 * A translation record (Job.to_dict in app/history.py).
 * @typedef {"queued" | "running" | "done" | "error" | "interrupted" | "paused" | "idle"} JobStatus
 * @typedef {{ name: string, url: string }} JobOutput
 * @typedef {{ name: string, done: number, total: number, unit: string }} JobStage
 * @typedef {{
 *   id: string, filename: string, status: JobStatus, done: number, total: number, error: string, created: number,
 *   finished: number, source_lang: string, target_lang: string, mode: string, service_id: string,
 *   service_name: string, glossary_ids: string[], queue_position: number, model: string, size: number,
 *   detected_lang: string, skipped: number, failed: number, prompt_tokens: number, completion_tokens: number,
 *   reasoning_tokens: number, cached_tokens: number, cache_write_tokens: number, tokens_reported: boolean,
 *   requests: RequestStats | null, same_lang: boolean, preview: boolean, stage: JobStage | null,
 *   outputs: JobOutput[]
 * }} Job
 * @typedef {{
 *   count: number, retries: number, failed: number, mean_latency: number, p90_latency: number,
 *   mean_gate_wait: number, reasons: Record<string, number>
 * }} RequestStats
 * @typedef {{ jobs: Job[], disk_usage: number, retention_days: number }} JobList
 */

/**
 * The reader's preview of a record (GET /api/jobs/{id}/preview). segments, the paragraphs in document order, come
 * only with the first fetch (since=-1). An update is [paragraph index, translation, 1 when skipped else 0].
 * @typedef {{ s: string, k: string }} Segment
 * @typedef {[number, string, number]} PreviewUpdate
 * @typedef {{ ready: false, version: number } | {
 *   ready: true, version: number, updates: PreviewUpdate[], segments?: Segment[]
 * }} Preview
 */

/**
 * What clearing a record's cache would remove (GET /api/jobs/{id}/cache), and the unused cache (/api/cache/unused).
 * @typedef {{
 *   entries: number, outputs: number, generated_bytes: number, boxes_bytes: number, shared: number
 * }} JobCache
 * @typedef {{ removed: number, untracked: number }} CacheSweep
 */

/**
 * A translation service (public() in app/services.py), without its key: one configured here, or the one currently
 * active in CC Switch, whose connection details are read from there (_ccswitch_services).
 * @typedef {{
 *   id: string, provider: string, builtin: boolean, name: string, enabled: boolean, base_url: string, model: string,
 *   concurrency: number, max_items: number, max_chars: number, temperature: number | null, rpm: number | null,
 *   reasoning: string, prompt: string, user_prompt: string, glossary: string, auto_glossary: boolean,
 *   api_key_hint: string
 * }} ServiceFields
 * @typedef {ServiceFields & { source?: undefined, available?: undefined }} LocalService
 * @typedef {ServiceFields & {
 *   source: "ccswitch", ccswitch_model: string, available: boolean, unavailable_reason: string
 * }} CCSwitchService
 * @typedef {LocalService | CCSwitchService} Service
 * @typedef {{ db: string, found: boolean }} CCSwitchInfo
 * @typedef {{ default: string, services: Service[], ccswitch: CCSwitchInfo }} ServiceList
 */

/**
 * A kind of translation service (PROVIDERS in app/services.py).
 * @typedef {{
 *   label: string, desc: string, base_url: string, needs_key: boolean, llm: boolean, key_url?: string,
 *   models: string[]
 * }} Provider
 */

/**
 * @typedef {{ system: string, user: string }} DefaultPrompts
 * @typedef {{ source: string, result: string, ms: number }} ServiceTest
 * @typedef {{ code: string, name: string, native: string, google: boolean }} Language
 * @typedef {{ id: string, name: string, entries: string, count: number }} Glossary
 * @typedef {{ entries: string, count: number }} GlossaryImport
 * @typedef {{ retention_days: number, max_jobs: number }} Settings
 */

/**
 * The layout model or the formula detection model (_model_status in app/routes/models.py). size, workers, threads
 * and auto belong to the formula detection model only.
 * @typedef {{ exists: boolean, size: number }} FileStatus
 * @typedef {{ running: boolean, done: number, total: number, error: string, source: string }} DownloadState
 * @typedef {{ running: boolean, done: number, total: number, error: string }} QuantizeState
 * @typedef {{
 *   path: string, file: FileStatus, available: boolean, install_hint: string, state: QuantizeState
 * }} Int8Status
 * @typedef {{ workers: number, threads: number }} ModelParallel
 * @typedef {{
 *   enabled: boolean, path: string, size?: number, precision?: string, workers?: number, threads?: number,
 *   int8: Int8Status, in_use: string, auto?: ModelParallel, resolved: string, default_path: string,
 *   runtime: string, install_hint: string, file: FileStatus, download: DownloadState,
 *   sources: { id: string, label: string }[]
 * }} ModelStatus
 * @typedef {{ load_ms: number, page_ms: number, labels: string[] }} ModelTest
 */

const SETTINGS_KEY = "doc-translator-settings";

/**
 * @type {{
 *   providers: Record<string, Provider>, services: Service[], glossaries: Glossary[], glossaryId: string | null,
 *   defaultId: string, selectedId: string | null, jobs: Job[], diskUsage: number, retentionDays: number,
 *   maxJobs: number, readerView: string, view: string, selected: Set<string>, historyPage: number,
 *   pageSize: number, pageJobs: Job[], ccswitch?: CCSwitchInfo, serviceTouched?: boolean, languages?: Language[]
 * }}
 */
const state = {
  providers: {}, services: [], glossaries: [], glossaryId: null, defaultId: "google", selectedId: null,
  jobs: [], diskUsage: 0, retentionDays: 30, maxJobs: 2, readerView: "both", view: "translate",
  selected: new Set(),  // ids of records ticked on the records page, for batch deletion
  historyPage: 1,  // current page of the records page (from 1)
  pageSize: 20,  // records per page, adjustable on the page (one of PAGE_SIZES)
  pageJobs: [],  // records on the current page of the records page
};

/** A failed request, with the HTTP status of the answer. */
class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   */
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

/**
 * Options of api(): those of fetch, plus json, a body sent as JSON (by POST unless another method is given).
 * @typedef {RequestInit & { json?: unknown }} ApiOptions
 */

/**
 * Call the server. The answer's shape is not checked here; callers state it (see the typedefs above).
 * @param {string} url
 * @param {ApiOptions} [options]
 * @returns {Promise<unknown>}
 */
async function api(url, options = {}) {
  const init = { ...options };
  if (init.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  const res = await fetch(url, init);
  // Opened from another machine and signed out since (the session ended, or the server got a new ACCESS_TOKEN)
  if (res.status === 401) location.assign(`/login.html${location.hash}`);
  /** @type {unknown} */
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data === "object" && data !== null && "detail" in data ? data.detail : undefined;
    throw new ApiError(typeof detail === "string" ? detail : `请求失败 (${res.status})`, res.status);
  }
  return data;
}

// ---------- Views ----------
// Pages: translate / history (translation records) / services (translation services) / glossaries / layout (layout
// models);
// the reader is a full-screen layer on top

const VIEWS = ["translate", "history", "services", "glossaries", "layout"];

/** @param {string | undefined} name a page in VIEWS; anything else shows the translate page */
function showView(name) {
  if (name === undefined || !VIEWS.includes(name)) name = "translate";
  for (const tab of queryAll(document, ".tab", HTMLElement)) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === name));
  }
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== name;
  state.view = name;
  if (name === "history") refreshJobs();
  if (name === "layout") loadLayoutModel();
  try {
    localStorage.setItem("doc-translator-view", name);
  } catch {
    /* ignore */
  }
}
for (const tab of queryAll(document, ".tab", HTMLElement)) {
  tab.addEventListener("click", () => {
    if (reader.jobId) closeReader();
    showView(tab.dataset.view);
  });
}
document.addEventListener("click", (e) => {
  const goto = e.target instanceof Element ? e.target.closest("[data-goto]") : null;
  if (goto instanceof HTMLElement) showView(goto.dataset.goto);
});

/** @type {number | undefined} */
let toastTimer;
/** @param {string} text */
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 2600);
}

/**
 * Immediate feedback for async buttons: disabled with a busy label on click, restored when done (success or failure)
 * @param {HTMLButtonElement} btn
 * @param {string} text
 * @param {() => Promise<unknown>} fn
 */
async function busy(btn, text, fn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = text;
  try {
    await fn();
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}


// ---------- Service data ----------

async function loadServices() {
  const data = /** @type {ServiceList} */ (await api("/api/services"));
  state.services = data.services;
  state.defaultId = data.default;
  state.ccswitch = data.ccswitch;
  if (!state.services.some((s) => s.id === state.selectedId)) state.selectedId = state.defaultId;
  renderServiceList();
  renderServiceSelect();
  renderDetail();
}

// ---------- Glossaries ----------

async function loadGlossaries() {
  state.glossaries = /** @type {Glossary[]} */ (await api("/api/glossaries"));
  if (!state.glossaries.some((g) => g.id === state.glossaryId)) state.glossaryId = state.glossaries[0]?.id ?? null;
  renderGlossaryList();
  renderGlossaryDetail();
  renderGlossaryPick();
}

function renderGlossaryList() {
  const list = $("glossary-list");
  list.replaceChildren();
  for (const g of state.glossaries) {
    const li = el("li", "service-item");
    li.classList.toggle("active", g.id === state.glossaryId);
    const pick = el("button", "service-pick");
    pick.type = "button";
    pick.append(el("span", "service-name", g.name), el("span", "tag", `${g.count} 条`));
    pick.addEventListener("click", () => {
      state.glossaryId = g.id;
      renderGlossaryList();
      renderGlossaryDetail();
    });
    li.append(pick);
    list.append(li);
  }
}

/** The fields of the glossary form. */
function glossaryFields() {
  const form = byId("glossary-form", HTMLFormElement);
  return { name: field(form, "name", HTMLInputElement), entries: field(form, "entries", HTMLTextAreaElement) };
}

function renderGlossaryDetail() {
  const g = state.glossaries.find((x) => x.id === state.glossaryId);
  const form = $("glossary-form");
  form.hidden = !g;
  $("glossary-empty").hidden = !!g;
  $("glossary-status").textContent = "";
  if (!g) return;
  const f = glossaryFields();
  f.name.value = g.name;
  f.entries.value = g.entries;
  byId("glossary-export", HTMLAnchorElement).href = `/api/glossaries/${g.id}/export`;
}

// The translate page lists the glossaries as checkboxes; a choice is kept while the list refreshes
function renderGlossaryPick() {
  const box = $("glossary-checks");
  const kept = new Set(queryAll(box, "input:checked", HTMLInputElement).map((i) => i.value));
  box.replaceChildren();
  for (const g of state.glossaries) {
    const label = el("label", "check");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "glossary_ids";
    input.value = g.id;
    input.checked = kept.has(g.id);
    label.append(input, el("span", "", `${g.name}（${g.count} 条）`));
    box.append(label);
  }
  $("glossary-pick").hidden = !state.glossaries.length;
}

$("glossary-add").addEventListener("click", async () => {
  try {
    const g = /** @type {Glossary} */ (
      await api("/api/glossaries", { json: { name: `术语表 ${state.glossaries.length + 1}`, entries: "" } })
    );
    state.glossaryId = g.id;
    await loadGlossaries();
    glossaryFields().name.select();
  } catch (err) {
    toast(messageOf(err));
  }
});

$("glossary-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = glossaryFields();
  await busy(byId("glossary-save", HTMLButtonElement), "保存中…", async () => {
    try {
      await api(`/api/glossaries/${state.glossaryId}`, {
        method: "PUT",
        json: { name: f.name.value, entries: f.entries.value },
      });
      await loadGlossaries();
      $("glossary-status").textContent = "已保存";
    } catch (err) {
      $("glossary-status").textContent = `保存失败：${messageOf(err)}`;
    }
  });
});

$("glossary-delete").addEventListener("click", async () => {
  const g = state.glossaries.find((x) => x.id === state.glossaryId);
  if (!g || !confirm(`确定删除「${g.name}」吗？已用它翻译的记录继续翻译时不再带这些术语。`)) return;
  try {
    await api(`/api/glossaries/${g.id}`, { method: "DELETE" });
    await loadGlossaries();
  } catch (err) {
    toast(messageOf(err));
  }
});

$("glossary-import").addEventListener("click", () => $("glossary-file").click());
$("glossary-file").addEventListener("change", async () => {
  const input = byId("glossary-file", HTMLInputElement);
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  try {
    const json = { text: await file.text() };
    const got = /** @type {GlossaryImport} */ (await api("/api/glossaries/import", { json }));
    const box = glossaryFields().entries;
    box.value = [box.value.trimEnd(), got.entries].filter(Boolean).join("\n");
    $("glossary-status").textContent = `读入 ${got.count} 条，保存后生效`;
  } catch (err) {
    $("glossary-status").textContent = `导入失败：${messageOf(err)}`;
  }
});

// 24x24 vector icons, colored by currentColor
/** @type {Record<string, string>} */
const ICONS = {
  google: '<path fill="currentColor" d="M21.6 12.23c0-.68-.06-1.33-.17-1.96H12v3.7h5.38a4.6 4.6 0 0 1-2 3.02v2.5h3.23c1.89-1.74 2.99-4.3 2.99-7.26z"/><path fill="currentColor" opacity=".75" d="M12 22c2.7 0 4.96-.9 6.61-2.43l-3.23-2.5c-.9.6-2.04.96-3.38.96-2.6 0-4.8-1.76-5.59-4.12H3.07v2.58A10 10 0 0 0 12 22z"/><path fill="currentColor" opacity=".55" d="M6.41 13.91a6 6 0 0 1 0-3.82V7.51H3.07a10 10 0 0 0 0 8.98z"/><path fill="currentColor" opacity=".9" d="M12 5.96c1.47 0 2.79.5 3.83 1.5l2.87-2.87A9.6 9.6 0 0 0 12 2 10 10 0 0 0 3.07 7.51l3.34 2.58C7.2 7.72 9.4 5.96 12 5.96z"/>',
  openai: '<path fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" d="M12 3.2l6.4 3.7v7.4L12 18l-6.4-3.7V6.9zM12 3.2v7.4m0 0l6.4-3.7M12 10.6L5.6 6.9m6.4 3.7v7.4m0-7.4l6.4 3.7M12 10.6l-6.4 3.7"/>',
  claude: '<path fill="currentColor" d="M12 2.5l1.5 6.2 4.9-4.1-2.6 5.8 6.2-1.4-5.4 3.4 5.4 3.4-6.2-1.4 2.6 5.8-4.9-4.1L12 21.5l-1.5-6.2-4.9 4.1 2.6-5.8-6.2 1.4 5.4-3.4-5.4-3.4 6.2 1.4-2.6-5.8 4.9 4.1z"/>',
  gemini: '<path fill="currentColor" d="M12 2c.6 5.4 4.6 9.4 10 10-5.4.6-9.4 4.6-10 10-.6-5.4-4.6-9.4-10-10 5.4-.6 9.4-4.6 10-10z"/>',
  grok: '<circle cx="12" cy="12" r="8.2" fill="none" stroke="currentColor" stroke-width="2"/><path stroke="currentColor" stroke-width="2.4" stroke-linecap="round" d="M6.2 19.2L18.4 5.6"/>',
  custom: '<circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6l2.1 2.1M18.4 5.6l-2.1 2.1m-8.6 8.6l-2.1 2.1"/>',
};

/** @param {string} provider */
function iconFor(provider) {
  const span = document.createElement("span");
  span.className = `icon icon-${provider}`;
  span.innerHTML = `<svg viewBox="0 0 24 24" width="60%" height="60%">${ICONS[provider] || ICONS.custom}</svg>`;
  span.setAttribute("aria-hidden", "true");
  return span;
}

function renderServiceSelect() {
  const select = byId("service_id", HTMLSelectElement);
  // Select the default service when the page opens; a later manual choice is kept (not reset when the service list
  // refreshes)
  const previous = state.serviceTouched ? select.value : state.defaultId;
  select.replaceChildren();
  for (const s of state.services.filter((s) => s.enabled)) {
    const label = s.id === state.defaultId ? `${s.name}（默认）` : s.name;
    select.add(new Option(label, s.id));
  }
  const ids = [...select.options].map((o) => o.value);
  select.value = ids.includes(previous) ? previous : state.defaultId;
  if (state.languages) checkLangPair();
}

/**
 * @param {string} text
 * @param {HTMLElement | null} extra
 */
function groupHeader(text, extra) {
  const li = document.createElement("li");
  li.className = "service-group";
  const span = document.createElement("span");
  span.textContent = text;
  li.append(span);
  if (extra) li.append(extra);
  return li;
}

function renderServiceList() {
  const list = $("service-list");
  list.replaceChildren();
  const local = state.services.filter((s) => s.source !== "ccswitch");
  const cc = state.services.filter((s) => s.source === "ccswitch");
  const refresh = document.createElement("button");
  refresh.type = "button";
  refresh.className = "link small";
  refresh.textContent = "刷新";
  refresh.title = "在 CC Switch 中切换供应商后，点击此处重新读取";
  refresh.addEventListener("click", () =>
    busy(refresh, "刷新中…", async () => {
      try {
        await loadServices();
        toast("服务列表已刷新");
      } catch (err) {
        toast(`刷新失败：${messageOf(err)}`);
      }
    })
  );
  /** @type {[string, Service[], HTMLElement | null][]} */
  const sections = [["本地配置", local, null]];
  if (state.ccswitch?.found || cc.length) sections.push(["CC Switch 当前生效", cc, refresh]);
  for (const [title, items, extra] of sections) {
    list.append(groupHeader(title, extra));
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "hint service-empty";
      empty.textContent = "未读取到配置";
      list.append(empty);
    }
    for (const s of items) list.append(serviceItem(s));
  }
}

/** @param {Service} s */
function serviceItem(s) {
  const li = document.createElement("li");
  li.className = "service-item";
  li.classList.toggle("active", s.id === state.selectedId);

  const pick = document.createElement("button");
  pick.type = "button";
  pick.className = "service-pick";
  pick.append(iconFor(s.provider));
  const name = document.createElement("span");
  name.className = "service-name";
  name.textContent = s.name;
  pick.append(name);
  pick.addEventListener("click", () => {
    state.selectedId = s.id;
    renderServiceList();
    renderDetail();
  });
  li.append(pick);

  if (s.id === state.defaultId) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "当前默认";
    li.append(tag);
  } else if (s.enabled) {
    const setDefault = document.createElement("button");
    setDefault.type = "button";
    setDefault.className = "link small set-default";
    setDefault.textContent = "设为默认";
    setDefault.addEventListener("click", async () => {
      try {
        await api(`/api/services/${s.id}/default`, { method: "POST" });
        await loadServices();
      } catch (err) {
        alert(messageOf(err));
      }
    });
    li.append(setDefault);
  }

  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.className = "switch";
  toggle.checked = s.enabled;
  toggle.disabled = s.id === state.defaultId || s.available === false;
  if (s.available === false) li.title = s.unavailable_reason;
  else if (s.id === state.defaultId) toggle.title = "默认服务不能关闭";
  toggle.setAttribute("aria-label", `启用 ${s.name}`);
  toggle.addEventListener("change", async () => {
    try {
      await api(`/api/services/${s.id}`, { method: "PUT", json: { enabled: toggle.checked } });
      await loadServices();
    } catch (err) {
      toggle.checked = !toggle.checked;
      alert(messageOf(err));
    }
  });
  li.append(toggle);
  return li;
}

// ---------- Service details ----------

/**
 * Copy a CC Switch service to a local one: connection details are fixed and no longer follow CC Switch
 * @param {Service} s
 * @param {HTMLButtonElement} [btn]
 */
async function cloneService(s, btn) {
  const run = async () => {
    try {
      const svc = /** @type {Service} */ (await api(`/api/services/${s.id}/clone`, { method: "POST" }));
      state.selectedId = svc.id;
      await loadServices();
      toast("已复制为本地配置，连接信息不再随 CC Switch 变化");
    } catch (err) {
      toast(`复制失败：${messageOf(err)}`);
    }
  };
  if (btn) await busy(btn, "复制中…", run);
  else await run();
}

const svcForm = byId("svc-form", HTMLFormElement);

/** The fields of the service form, by name. */
function svcFields() {
  /** @param {string} name */
  const input = (name) => field(svcForm, name, HTMLInputElement);
  /** @param {string} name */
  const textarea = (name) => field(svcForm, name, HTMLTextAreaElement);
  return {
    api_key: input("api_key"), name: input("name"), base_url: input("base_url"), model: input("model"),
    concurrency: input("concurrency"), max_items: input("max_items"), max_chars: input("max_chars"),
    temperature: input("temperature"), rpm: input("rpm"), reasoning: field(svcForm, "reasoning", HTMLSelectElement),
    prompt: textarea("prompt"), user_prompt: textarea("user_prompt"),
    glossary: textarea("glossary"), auto_glossary: input("auto_glossary"),
  };
}

function selected() {
  return state.services.find((s) => s.id === state.selectedId);
}

/**
 * @param {string[]} models
 * @param {string} current
 */
function setModelOptions(models, current) {
  const select = byId("model-select", HTMLSelectElement);
  select.replaceChildren();
  const all = [...new Set([...(current ? [current] : []), ...models])];
  for (const m of all) select.add(new Option(m, m));
  select.value = current || all[0] || "";
}

/** @param {boolean} on */
function useCustomModel(on) {
  byId("custom-model", HTMLInputElement).checked = on;
  $("model-select").hidden = on;
  $("model-input").hidden = !on;
}

function renderDetail() {
  const s = selected();
  if (!s) return;
  /** @type {Partial<Provider>} */
  const meta = state.providers[s.provider] || {};
  $("svc-icon").replaceWith(Object.assign(iconFor(s.provider), { id: "svc-icon" }));
  $("svc-name-title").textContent = s.name;
  $("svc-desc").textContent = meta.desc || "";
  $("test-result").hidden = true;
  $("save-status").textContent = "";

  const fromCC = s.source === "ccswitch";
  const editable = meta.llm && (!fromCC || s.available);
  $("svc-fields").hidden = !editable;
  $("save-btn").hidden = !editable;
  $("delete-btn").hidden = s.builtin;
  $("test-btn").hidden = fromCC && !s.available;
  // A CC Switch service can be copied to a local configuration (connection details fixed, no longer following
  // CC Switch); an unavailable one has no key to copy
  $("clone-btn").hidden = !fromCC;
  byId("clone-btn", HTMLButtonElement).disabled = fromCC && !s.available;
  // URL, key and name of a CC Switch service come from CC Switch and are read-only here
  for (const id of ["api_key", "base_url", "svc-name-input"]) byId(id, HTMLInputElement).readOnly = fromCC;
  closest($("show-key"), "label").hidden = fromCC;
  $("cc-notice").hidden = !fromCC;
  if (fromCC) {
    $("cc-notice").textContent = s.available
      ? `来自 CC Switch 当前生效的「${s.name.split(" · ").pop()}」。地址和 API Key 在每次翻译时从 CC Switch 读取；在 CC Switch 中切换供应商后，点击列表中的「刷新」。模型和高级设置可在此处单独修改。`
      : `CC Switch 当前生效的「${s.name.split(" · ").pop()}」无法用于翻译：${s.unavailable_reason}。请在 CC Switch 中切换到配置了 API Key 的供应商，然后点击「刷新」。`;
  }
  if (!editable) return;

  const notice = $("key-notice");
  notice.replaceChildren();
  if (meta.needs_key && !s.api_key_hint && !fromCC) {
    const strong = document.createElement("strong");
    strong.textContent = s.name;
    notice.append(strong, " 需要填写 API Key 后才可用");
    if (meta.key_url) {
      const a = document.createElement("a");
      a.href = meta.key_url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = "申请 API Key";
      notice.append("，", a);
    }
  }
  notice.hidden = !notice.childNodes.length;

  const f = svcFields();
  f.api_key.value = "";
  f.api_key.placeholder = fromCC
    ? `来自 CC Switch（${s.api_key_hint}）`
    : s.api_key_hint ? `已保存（${s.api_key_hint}），留空则不修改` : meta.needs_key ? "必填" : "可选";
  f.api_key.type = "password";
  byId("show-key", HTMLInputElement).checked = false;
  f.name.value = s.name;
  f.base_url.value = s.base_url;
  f.base_url.placeholder = meta.base_url || "例如 https://api.deepseek.com/v1";
  f.base_url.required = !meta.base_url;
  f.concurrency.value = String(s.concurrency);
  f.max_items.value = String(s.max_items);
  f.max_chars.value = String(s.max_chars);
  f.temperature.value = String(s.temperature ?? "");
  f.rpm.value = String(s.rpm ?? "");
  f.reasoning.value = s.reasoning || "";
  f.prompt.value = s.prompt;
  f.user_prompt.value = s.user_prompt || "";
  f.glossary.value = s.glossary || "";
  f.auto_glossary.checked = s.auto_glossary !== false;
  showDefaultPromptPlaceholders();

  // Use the fetched model list if there is one, otherwise the preset models
  const fetched = fetchedModels[s.id];
  const models = fetched || meta.models || [];
  setModelOptions(models, s.model);
  f.model.value = s.model;
  $("fetch-models").textContent = fetched ? "重新获取" : "获取模型列表";
  byId("fetch-models", HTMLButtonElement).disabled = false;
  setFetchStatus(fetched ? `已获取 ${fetched.length} 个模型` : "");
  // Default to the text input when there are no models to choose from (such as the OpenAI-compatible API) or the
  // current model is not in the list (a custom model name of a relay service)
  useCustomModel(!models.length || (fromCC && !models.includes(s.model)));
  if (fromCC) {
    const input = byId("model-input", HTMLInputElement);
    input.placeholder = `留空时使用 CC Switch 中配置的模型（${s.ccswitch_model}）`;
  }
}

// The default prompt depends on the target language and is shown as the placeholder of an empty input
/** @type {Record<string, DefaultPrompts>} */
const defaultPromptCache = {};

async function getDefaultPrompts() {
  const to = byId("target_lang", HTMLSelectElement).value || "zh-CN";
  const from = byId("source_lang", HTMLSelectElement).value || "auto";
  const key = `${from}2${to}`;
  if (!defaultPromptCache[key]) {
    const qs = new URLSearchParams({ target_lang: to, source_lang: from });
    defaultPromptCache[key] = /** @type {DefaultPrompts} */ (await api(`/api/prompts/default?${qs}`));
  }
  return defaultPromptCache[key];
}

async function showDefaultPromptPlaceholders() {
  try {
    const d = await getDefaultPrompts();
    const f = svcFields();
    f.prompt.placeholder = d.system;
    f.user_prompt.placeholder = d.user;
  } catch {
    /* failing to get the default does not matter */
  }
}

$("load-default-prompt").addEventListener("click", async () => {
  const f = svcFields();
  if ((f.prompt.value || f.user_prompt.value) && !confirm("使用默认提示词替换当前内容？")) return;
  const d = await getDefaultPrompts();
  f.prompt.value = d.system;
  f.user_prompt.value = d.user;
});

for (const id of ["target_lang", "source_lang"]) {
  $(id).addEventListener("change", showDefaultPromptPlaceholders);
}

$("show-key").addEventListener("change", () => {
  svcFields().api_key.type = byId("show-key", HTMLInputElement).checked ? "text" : "password";
});
$("custom-model").addEventListener("change", () => {
  const custom = byId("custom-model", HTMLInputElement).checked;
  const select = byId("model-select", HTMLSelectElement);
  const input = byId("model-input", HTMLInputElement);
  if (custom) input.value = select.value;
  else setModelOptions([...select.options].map((o) => o.value), input.value);
  useCustomModel(custom);
});

/** The model chosen in the service form: typed in, or picked from the list. */
function chosenModel() {
  return byId("custom-model", HTMLInputElement).checked
    ? byId("model-input", HTMLInputElement).value
    : byId("model-select", HTMLSelectElement).value;
}

// The service form as sent to the server (to save, test or list models), for the selected service
function formPayload() {
  const f = svcFields();
  const s = selected();
  // The services page always has one selected once the list has loaded
  if (!s) throw new Error("没有选中的翻译服务");
  const model = chosenModel();
  return {
    id: s.id,
    provider: s.provider,
    name: f.name.value,
    api_key: f.api_key.value,
    base_url: f.base_url.value,
    model,
    concurrency: f.concurrency.value,
    max_items: f.max_items.value,
    max_chars: f.max_chars.value,
    temperature: f.temperature.value,
    rpm: f.rpm.value,
    reasoning: f.reasoning.value,
    prompt: f.prompt.value,
    user_prompt: f.user_prompt.value,
    glossary: f.glossary.value,
    auto_glossary: f.auto_glossary.checked,
    target_lang: byId("target_lang", HTMLSelectElement).value || "zh-CN",
    source_lang: byId("source_lang", HTMLSelectElement).value || "auto",
  };
}

svcForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formPayload();
  if (!payload.model && selected()?.source !== "ccswitch") {
    $("save-status").textContent = "请先选择或输入模型";
    return;
  }
  await busy(byId("save-btn", HTMLButtonElement), "保存中…", async () => {
    try {
      await api(`/api/services/${payload.id}`, { method: "PUT", json: payload });
      await loadServices();
      $("save-status").textContent = "已保存";
    } catch (err) {
      $("save-status").textContent = `保存失败：${messageOf(err)}`;
    }
  });
});

$("delete-btn").addEventListener("click", async () => {
  const s = selected();
  if (!s || !confirm(`确定删除「${s.name}」吗？`)) return;
  await busy(byId("delete-btn", HTMLButtonElement), "删除中…", async () => {
    try {
      await api(`/api/services/${s.id}`, { method: "DELETE" });
      state.selectedId = state.defaultId;
      await loadServices();
    } catch (err) {
      alert(messageOf(err));
    }
  });
});

$("clone-btn").addEventListener("click", () => {
  const s = selected();
  if (s) cloneService(s, byId("clone-btn", HTMLButtonElement));
});

$("test-btn").addEventListener("click", async () => {
  const box = $("test-result");
  const s = selected();
  box.hidden = false;
  box.className = "test-result";
  box.textContent = "测试中…";
  await busy(byId("test-btn", HTMLButtonElement), "测试中…", async () => {
    try {
      // As formPayload, which fails the same way when no service is selected
      if (!s) throw new Error("没有选中的翻译服务");
      const payload = state.providers[s.provider]?.llm ? formPayload() : { id: s.id, provider: s.provider };
      const r = /** @type {ServiceTest} */ (await api("/api/services/test", { json: payload }));
      box.classList.add("ok");
      box.textContent = `测试成功（${r.ms} ms）：${r.source} → ${r.result}`;
    } catch (err) {
      box.classList.add("fail");
      box.textContent = `测试失败：${messageOf(err)}`;
    }
  });
});

// Fetched model lists are cached per service, so switching away and back needs no new fetch
/** @type {Record<string, string[]>} */
const fetchedModels = {};

/**
 * @param {string} text
 * @param {boolean} [isError]
 */
function setFetchStatus(text, isError = false) {
  const el = $("fetch-models-status");
  el.textContent = text;
  el.classList.toggle("warn", isError);
}

$("fetch-models").addEventListener("click", async () => {
  const btn = byId("fetch-models", HTMLButtonElement);
  const sid = state.selectedId;
  btn.disabled = true;
  setFetchStatus("获取中…");
  try {
    const payload = formPayload();
    payload.model = payload.model || "placeholder"; // no model is selected yet while fetching the list
    const { models } = /** @type {{ models: string[] }} */ (await api("/api/services/models", { json: payload }));
    if (!models.length) throw new Error("接口未返回任何模型");
    fetchedModels[String(sid)] = models;
    if (state.selectedId !== sid) return; // switched to another service while fetching
    const current = chosenModel();
    // Use only the list returned by the API, without the preset models; when the current model is not in it (the
    // provider does not support it), select the first one instead of keeping an invalid choice
    const fallback = models.includes(current) ? current : models[0];
    setModelOptions(models, fallback);
    useCustomModel(false);
    const note = current && current !== fallback ? `，原模型 ${current} 不在列表中，已改选 ${fallback}` : "";
    setFetchStatus(`已获取 ${models.length} 个模型${note} · ${new Date().toLocaleTimeString()}`);
  } catch (err) {
    if (state.selectedId === sid) setFetchStatus(`获取失败：${messageOf(err)}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = fetchedModels[String(sid)] ? "重新获取" : "获取模型列表";
  }
});

$("add-btn").addEventListener("click", async () => {
  await busy(byId("add-btn", HTMLButtonElement), "添加中…", async () => {
    try {
      const provider = byId("add-provider", HTMLSelectElement).value;
      const svc = /** @type {Service} */ (await api("/api/services", { json: { provider } }));
      state.selectedId = svc.id;
      await loadServices();
      svcFields().api_key.focus();
    } catch (err) {
      alert(messageOf(err));
    }
  });
});

// ---------- Translate page ----------

const form = byId("form", HTMLFormElement);
const fileInput = byId("file", HTMLInputElement);
const drop = $("drop");
// The translation service is not remembered: every page load starts with the default service
const SAVED_FIELDS = ["source_lang", "target_lang", "mode"];

/**
 * The form fields saveSettings stored, by name.
 * @returns {Record<string, string>}
 */
function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "null") || {};
  } catch {
    return {};
  }
}

function saveSettings() {
  /** @type {Record<string, string>} */
  const data = {};
  for (const name of SAVED_FIELDS) data[name] = control(form, name).value;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
  } catch {
    /* may be unavailable in private mode; ignore */
  }
}


// ---------- Languages ----------

/** @param {string} code */
function langInfo(code) {
  return (state.languages || []).find((l) => l.code === code);
}

/** @param {string} code */
function langLabel(code) {
  const l = langInfo(code);
  if (!l) return code;
  if (code === "auto") return "自动检测";
  // Native name + English name, such as "日本語 Japanese"; only one when both are the same
  return l.native && l.native !== l.name ? `${l.native} ${l.name}` : l.name;
}

// As in the extension, the default target language follows the browser language (navigator.language)
function defaultTargetLang() {
  const codes = (state.languages || []).map((l) => l.code);
  for (const raw of navigator.languages || [navigator.language || "zh-CN"]) {
    let code = raw;
    if (/^zh-(TW|HK|MO|Hant)/i.test(raw)) code = /HK|MO/i.test(raw) ? "zh-HK" : "zh-TW";
    else if (/^zh/i.test(raw)) code = "zh-CN";
    else if (/^pt-BR$/i.test(raw)) code = "pt-br";
    if (codes.includes(code)) return code;
    const base = raw.split("-")[0];
    if (codes.includes(base)) return base;
  }
  return "zh-CN";
}

function selectedService() {
  const id = byId("service_id", HTMLSelectElement).value;
  return state.services.find((s) => s.id === id);
}

function checkLangPair() {
  const from = byId("source_lang", HTMLSelectElement).value;
  const to = byId("target_lang", HTMLSelectElement).value;
  const warn = $("lang-warning");
  /** @type {string[]} */
  const msgs = [];
  if (from !== "auto" && from === to) msgs.push("源语言与目标语言相同，无需翻译。");
  const svc = selectedService();
  if (svc && svc.provider === "google") {
    const bad = [from, to].filter((c) => c !== "auto" && !langInfo(c)?.google);
    if (bad.length) msgs.push(`谷歌翻译不支持${bad.map(langLabel).join("、")}，请换用 AI 翻译服务。`);
  }
  warn.textContent = msgs.join(" ");
  warn.hidden = !msgs.length;
  byId("submit", HTMLButtonElement).disabled = msgs.length > 0;
}

for (const id of ["source_lang", "target_lang", "service_id"]) {
  $(id).addEventListener("change", checkLangPair);
}
$("service_id").addEventListener("change", () => (state.serviceTouched = true));

$("swap-lang").addEventListener("click", () => {
  const src = byId("source_lang", HTMLSelectElement);
  const dst = byId("target_lang", HTMLSelectElement);
  // A source language of "自动检测" cannot be swapped into the target, so the last detected language is used:
  // the language detected in the most recent record
  const lastDetected = (state.jobs || []).find((j) => j.detected_lang)?.detected_lang;
  const from = src.value === "auto" ? lastDetected : src.value;
  if (!from) {
    toast("源语言为自动检测，无法交换。请先手动选择源语言，或完成一次翻译后再交换");
    return;
  }
  src.value = dst.value;
  dst.value = from;
  checkLangPair();
  showDefaultPromptPlaceholders();
});

/** @param {number} bytes */
function formatSize(bytes) {
  if (!bytes) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function showFile() {
  const files = [...(fileInput.files ?? [])];
  drop.classList.toggle("has-file", files.length > 0);
  const size = formatSize(files.reduce((sum, f) => sum + f.size, 0));
  if (!files.length) {
    $("drop-text").textContent = "选择文件，或将文件拖放到此处";
    $("drop-sub").textContent = "可选多个文件，每个最大 200 MB";
  } else if (files.length === 1) {
    $("drop-text").textContent = files[0].name;
    $("drop-sub").textContent = `${size} · 点击更换`;
  } else {
    $("drop-text").textContent = `${files.length} 个文件`;
    $("drop-sub").textContent = `共 ${size} · 点击更换`;
  }
  drop.title = files.map((f) => f.name).join("\n");
}
fileInput.addEventListener("change", showFile);
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer?.files.length) {
    fileInput.files = e.dataTransfer.files;
    showFile();
  }
});

/** @param {string} text */
function showSubmitError(text) {
  const el = $("submit-error");
  el.textContent = text || "";
  el.hidden = !text;
}

// Several files are uploaded one by one, each as its own job with the same settings. A file that fails does not stop
// the others; the failures are listed below the button
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = [...(fileInput.files ?? [])];
  if (!files.length) return;
  // Asked here, on a click, as browsers require; the answer is remembered by the browser. Safari before 15 takes a
  // callback and returns nothing
  if (canNotify() && Notification.permission === "default") Notification.requestPermission()?.catch(() => {});
  saveSettings();
  showSubmitError("");
  const btn = byId("submit", HTMLButtonElement);
  btn.disabled = true;
  /** @type {Job[]} */
  const jobs = [];
  /** @type {File[]} */
  const failed = [];
  /** @type {string[]} */
  const errors = [];
  // The settings are read once, so changing the form during the upload does not affect the remaining files
  const fields = [...new FormData(form)].filter(([name]) => name !== "file");
  try {
    for (const [i, file] of files.entries()) {
      btn.textContent = files.length > 1 ? `上传中 ${i + 1} / ${files.length}…` : "上传中…";
      const body = new FormData();
      for (const [name, value] of fields) body.append(name, value);
      body.append("file", file);
      try {
        const job = /** @type {Job} */ (await api("/api/jobs", { method: "POST", body }));
        jobs.push(job);
        upsertJob(job);
      } catch (err) {
        failed.push(file);
        errors.push(files.length > 1 ? `${file.name}：${messageOf(err)}` : messageOf(err));
      }
    }
    // Files that failed stay selected, so they can be submitted again after the cause is fixed
    const keep = new DataTransfer();
    for (const file of failed) keep.items.add(file);
    fileInput.files = keep.files;
    showFile();
    showSubmitError(errors.join("\n"));
    // Stay on this page: the new jobs appear under "最近翻译" below, and "查看进度" opens them in the reader
    if (jobs.length === 1) {
      toast(startedText(jobs[0], "已开始翻译"));
    } else if (jobs.length) {
      const queued = jobs.filter((j) => j.queue_position).length;
      toast(`已提交 ${jobs.length} 个文件${queued ? `，其中 ${queued} 个在排队` : ""}`);
    }
  } finally {
    btn.textContent = "开始翻译";
    btn.disabled = false;
    checkLangPair();
  }
});

// ---------- Translation records ----------
// Records are stored on the server's disk: they survive page reloads, switching books and server restarts, and are
// deleted only manually or after the retention period

/** @type {Record<JobStatus, string>} */
const STATUS_TEXT = {
  queued: "排队中", running: "翻译中", done: "已完成", error: "失败", interrupted: "已中断", paused: "已暂停", idle: "未翻译",
};
// States that allow continuing ("继续翻译"): translated paragraphs are in the cache and reused when continuing
/** @type {Set<string>} */
const RESUMABLE = new Set(["paused", "interrupted"]);
/** @type {Set<string>} */
const ACTIVE = new Set(["queued", "running"]);

// Source and translated text of ebook rich-text paragraphs carry inline tags (<b>, <a> and so on), stripped for the
// preview with plain string processing (not through the DOM: translations come from a model, and attributes such
// as onerror must not fire)
/** @param {string} s */
function htmlToText(s) {
  if (!/<[a-zA-Z/]/.test(s) && !/&(?:#\d+|#x[0-9a-f]+|[a-z]+);/i.test(s)) return s;
  /** @type {Record<string, string>} */
  const named = { nbsp: "\u00a0", apos: "'", amp: "&", lt: "<", gt: ">", quot: '"' };
  return s
    .replace(/<br\b[^>]*>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCharCode(parseInt(n, 16)))
    .replace(/&([a-z]+);/gi, (m, name) => named[name.toLowerCase()] ?? m);
}

/** @param {Job | null} job */
function isEbookJob(job) {
  return /\.(epub|mobi|azw3?|azw)$/i.test(job?.filename || "");
}

// For display in the reader: strip inline tags from ebooks and web pages, tags and content placeholders ({vN}) from
// Word documents, and formula placeholders, script sentinels and formula image sentinels from PDFs
/**
 * @param {Job | null} job
 * @param {string} s
 */
function previewText(job, s) {
  if (isEbookJob(job) || /\.(html?|xhtml)$/i.test(job?.filename || "")) return htmlToText(s);
  if (/\.docx$/i.test(job?.filename || "")) return htmlToText(s).replace(/\s*\{\s*v\s*\d+\s*\}\s*/g, " ").trim();
  if (/\.pdf$/i.test(job?.filename || "")) {
    // and the bold, italic and size markers of the text sent for translation, and the page break marker {|}
    return s.replace(/\s*\{\|\}\s*/g, " ").replace(/\{\/?[bi]\}|\{z\d+\}|\{\/z\}/g, "").replace(/\{\s*v\s*\d+\s*\}/g, "").replace(/\x01i\x02\d+\x01\/i\x02/g, "").replace(/\x01\/?[sb]\x02/g, "");
  }
  return s;
}

/** @param {string} name */
function fmtOf(name) {
  return (name.split(".").pop() || "").toLowerCase().replace("markdown", "md");
}

// Same rule as output_variant in app/formats/__init__.py: the marker comes right before the extension
/**
 * @param {string} name
 * @returns {"bilingual" | "translated" | ""}
 */
function outputVariant(name) {
  const stem = name.slice(0, Math.max(0, name.lastIndexOf(".")));
  const token = stem.slice(stem.lastIndexOf(".") + 1);
  return token === "bilingual" || token === "translated" ? token : "";
}

// ---------- Drop-down menus ----------
// Opens below the button (above it near the bottom of the window); arrow keys move between items, and Esc, Tab, a
// click outside or scrolling close it. A window resize only repositions it: on phones, the address bar collapsing
// or expanding also triggers resize. While records are running, the list and the reader redraw periodically; with
// a menu open the redraw is deferred until it closes (see deferWhileMenuOpen)
/** @type {{ panel: HTMLElement | null, anchor: HTMLElement | null, pending: Map<string, () => void> }} */
const menu = { panel: null, anchor: null, pending: new Map() };

/**
 * @param {string} key
 * @param {() => void} render
 */
function deferWhileMenuOpen(key, render) {
  if (!menu.panel) return false;
  menu.pending.set(key, render);
  return true;
}

function closeMenu({ focus = false } = {}) {
  const { panel, anchor } = menu;
  if (!panel || !anchor) return;
  menu.panel = menu.anchor = null;
  panel.remove();
  anchor.setAttribute("aria-expanded", "false");
  document.removeEventListener("pointerdown", onMenuPointer, true);
  document.removeEventListener("scroll", onMenuScroll, true);
  window.removeEventListener("resize", placeMenu);
  if (focus && anchor.isConnected) anchor.focus();
  const pending = [...menu.pending.values()];
  menu.pending.clear();
  for (const render of pending) render();
}

// The listeners are added only while a menu is open
/** @param {Event} e */
function onMenuPointer(e) {
  const target = e.target instanceof Node ? e.target : null;
  if (!menu.panel?.contains(target) && !menu.anchor?.contains(target)) closeMenu();
}

/** @param {Event} e */
function onMenuScroll(e) {
  if (!menu.panel?.contains(e.target instanceof Node ? e.target : null)) closeMenu();
}

/**
 * Menu items: { label, hint, href, download, newTab, danger, disabled, onSelect }, group headings { heading },
 * separator "sep"
 * @typedef {{
 *   label: string, hint?: string, href?: string, download?: string, newTab?: boolean, danger?: boolean,
 *   disabled?: boolean, onSelect?: () => void
 * }} MenuAction
 * @typedef {MenuAction | { heading: string } | "sep"} MenuItem
 */

/** @param {MenuItem} item */
function menuItem(item) {
  if (item === "sep") return el("div", "menu-sep");
  if ("heading" in item) return el("div", "menu-heading", item.heading);
  const cls = `menu-item${item.danger ? " danger" : ""}`;
  // A link or a button, set up as each needs
  /** @type {HTMLAnchorElement | HTMLButtonElement} */
  let node;
  if (item.href) {
    const a = el("a", cls);
    a.href = item.href;
    if (item.download) a.download = item.download;
    if (item.newTab) {
      a.target = "_blank";
      a.rel = "noopener";
    }
    node = a;
  } else {
    const b = el("button", cls);
    b.type = "button";
    b.disabled = Boolean(item.disabled);
    node = b;
  }
  node.setAttribute("role", "menuitem");
  node.tabIndex = -1;
  node.append(el("span", "", item.label));
  if (item.hint) node.append(el("span", "menu-hint", item.hint));
  node.addEventListener("click", () => {
    closeMenu();
    item.onSelect?.();
  });
  return node;
}

// Aligned with the button's right edge, and placed above when there is no room below
function placeMenu() {
  const { panel, anchor } = menu;
  if (!panel || !anchor) return;
  const rect = anchor.getBoundingClientRect();
  const { offsetWidth: w, offsetHeight: h } = panel;
  const below = rect.bottom + 4 + h <= window.innerHeight - 8;
  panel.style.top = `${below ? rect.bottom + 4 : Math.max(8, rect.top - 4 - h)}px`;
  panel.style.left = `${Math.min(Math.max(8, rect.right - w), window.innerWidth - w - 8)}px`;
}

/**
 * @param {HTMLElement} anchor
 * @param {MenuItem[]} items
 * @param {{ keyboard?: boolean }} [options] keyboard: opened from the keyboard, so the first item takes the focus
 */
function openMenu(anchor, items, { keyboard = false } = {}) {
  closeMenu();
  const panel = el("div", "menu");
  panel.setAttribute("role", "menu");
  panel.append(...items.map(menuItem));
  document.body.append(panel);
  menu.panel = panel;
  menu.anchor = anchor;
  placeMenu();
  anchor.setAttribute("aria-expanded", "true");
  document.addEventListener("pointerdown", onMenuPointer, true);
  document.addEventListener("scroll", onMenuScroll, true);
  window.addEventListener("resize", placeMenu);
  const entries = () => queryAll(panel, ".menu-item:not(:disabled)", HTMLElement);
  panel.addEventListener("keydown", (e) => {
    const list = entries();
    const at = list.findIndex((item) => item === document.activeElement);
    /** @type {Record<string, number | undefined>} */
    const moves = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: list.length - 1 };
    const move = moves[e.key];
    if (move !== undefined) {
      e.preventDefault();
      list[(move + list.length) % list.length]?.focus();
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeMenu({ focus: true });
    } else if (e.key === "Tab") {
      closeMenu();
    }
  });
  (keyboard ? entries()[0] : panel)?.focus?.();
}

/**
 * A button with a drop-down menu. items is computed on opening, so it sees the latest state of the record
 * @param {string} label
 * @param {string} cls
 * @param {() => MenuItem[]} items
 * @param {string} [ariaLabel]
 */
function menuButton(label, cls, items, ariaLabel) {
  const btn = el("button", cls, label);
  btn.type = "button";
  btn.setAttribute("aria-haspopup", "menu");
  btn.setAttribute("aria-expanded", "false");
  if (ariaLabel) btn.setAttribute("aria-label", ariaLabel);
  btn.addEventListener("click", (e) => {
    if (menu.anchor === btn) closeMenu();
    else openMenu(btn, items(), { keyboard: e.detail === 0 });
  });
  btn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" && menu.anchor !== btn) {
      e.preventDefault();
      openMenu(btn, items(), { keyboard: true });
    }
  });
  return btn;
}

// ---------- Downloads ----------

/** @type {Record<string, string>} */
const VARIANT_NAMES = { bilingual: "双语", translated: "译文" };

// The translation file of the current mode, which the download button downloads directly
/** @param {Job} job */
function currentOutput(job) {
  const current = job.mode === "bilingual" ? "bilingual" : "translated";
  return job.outputs.find((o) => outputVariant(o.name) === current) || job.outputs[0] || null;
}

// When one version has several formats (EPUB + MOBI for a MOBI job), the label includes the format name
/**
 * @param {Job} job
 * @param {JobOutput} o
 */
function outputLabel(job, o) {
  const variant = outputVariant(o.name);
  const formats = job.outputs.filter((x) => outputVariant(x.name) === variant).length;
  return `${VARIANT_NAMES[variant] || ""}${formats > 1 ? ` ${fmtOf(o.name).toUpperCase()}` : ""}`;
}

// PDFs use the browser's built-in viewer and EPUBs epub.html, chapter by chapter; other formats have no preview
/**
 * @param {Job} job
 * @param {JobOutput} o
 */
function previewHref(job, o) {
  const fmt = fmtOf(o.name);
  if (fmt === "pdf") return `${o.url}?inline=1`;
  if (fmt === "epub") return `epub.html?${new URLSearchParams({ job: job.id, file: o.name })}`;
  return null;
}

// Open the translation file of the current version in a new tab
/**
 * @param {Job} job
 * @param {string} cls
 */
function previewButton(job, cls) {
  const o = currentOutput(job);
  const href = o && previewHref(job, o);
  if (!href) return null;
  const a = el("a", cls, "预览");
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener";
  a.title = `在新标签页打开 ${o.name}`;
  return a;
}

// Download is a split button: the left part downloads the current version, the arrow on the right lists the other
// files and the other version. Without other options there is only the left button. Returns null when there is
// nothing to download
/**
 * @param {Job} job
 * @param {string} cls
 */
function downloadButton(job, cls) {
  const main = currentOutput(job);
  if (!main) return null;
  const a = el("a", cls, `下载${outputLabel(job, main)}`);
  a.href = main.url;
  a.download = main.name;
  a.title = main.name;
  if (!otherDownloads(job, main).length) return a;
  const group = el("div", "split");
  group.append(a, menuButton("", `${cls} split-caret`, () => otherDownloads(job, main), "更多下载选项"));
  return group;
}

/**
 * @param {Job} job
 * @param {JobOutput} main
 */
function otherDownloads(job, main) {
  /** @type {MenuItem[]} */
  const items = [];
  for (const o of job.outputs) {
    if (o === main || o.name === main.name) continue;
    const label = outputLabel(job, o);
    items.push({ label: `下载${label}`, href: o.url, download: o.name });
    const href = previewHref(job, o);
    if (href) items.push({ label: `预览${label}`, hint: "新标签页", href, newTab: true });
  }
  const current = job.mode === "bilingual" ? "bilingual" : "translated";
  const other = current === "bilingual" ? "translated" : "bilingual";
  if (job.status === "done" && !job.outputs.some((o) => outputVariant(o.name) === other)) {
    const busy = variantInflight.has(job.id);
    items.push({
      label: busy ? `正在生成${VARIANT_NAMES[other]}…` : `生成并下载${VARIANT_NAMES[other]}`,
      hint: busy ? "" : "约需几秒", disabled: busy, onSelect: () => generateVariant(job, other),
    });
  }
  return items;
}

// The other version is generated in the background, and the download starts when it is ready
/** @type {Set<string>} */
const variantInflight = new Set();

/**
 * @param {Job} job
 * @param {"bilingual" | "translated"} marker
 */
async function generateVariant(job, marker) {
  if (variantInflight.has(job.id)) return;
  variantInflight.add(job.id);
  toast(`正在生成${VARIANT_NAMES[marker]}版本…`);
  try {
    const updated = /** @type {Job} */ (await api(`/api/jobs/${job.id}/variant`, { method: "POST" }));
    upsertJob(updated);
    if (reader.jobId === job.id) {
      reader.job = updated;
      renderReaderBar(updated);
    }
    const o = updated.outputs.find((out) => outputVariant(out.name) === marker);
    if (o) {
      const a = document.createElement("a");
      a.href = o.url;
      a.download = o.name;
      document.body.append(a);
      a.click();
      a.remove();
    }
  } catch (err) {
    toast(`生成失败：${messageOf(err)}`);
  } finally {
    variantInflight.delete(job.id);
  }
}

// Rarely used actions that lose data go into "更多": clearing the cache and deleting. Delete comes last,
// separated from the rest
/** @param {Job} job */
function moreMenu(job) {
  return menuButton("⋯", "btn sm ghost more", () => {
    /** @type {MenuItem[]} */
    const items = [];
    if (!ACTIVE.has(job.status) && job.status !== "idle") items.push({ label: "清理缓存…", onSelect: () => purgeJob(job) }, "sep");
    items.push({ label: "删除", danger: true, onSelect: () => deleteJob(job) });
    return items;
  }, "更多操作");
}

/** @param {number} ts seconds since the epoch */
function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "刚刚";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} 天前`;
  return new Date(ts * 1000).toLocaleDateString();
}

// Token usage: only LLM APIs report it. Google reports none and shows nothing; an LLM service (a relay) that sent
// requests without reporting usage shows 未提供 rather than a misleading 0
/** @param {Job} job */
function tokensText(job) {
  if (!job.tokens_reported) return job.requests && job.requests.count && job.model ? "tokens 未提供" : "";
  const total = (job.prompt_tokens || 0) + (job.completion_tokens || 0);
  const text = total >= 1e6 ? `${(total / 1e6).toFixed(2)}M` : total >= 1e3 ? `${(total / 1e3).toFixed(1)}k` : `${total}`;
  return `${text} tokens`;
}

/** @param {Job} job */
function tokensTitle(job) {
  if (!job.tokens_reported) return "翻译服务没有返回 token 用量";
  /** @param {number} n */
  const n = (n) => (n || 0).toLocaleString();
  const input = [`输入 ${n(job.prompt_tokens)}`];
  if (job.cached_tokens) input.push(`其中缓存命中 ${n(job.cached_tokens)}`);
  if (job.cache_write_tokens) input.push(`写入缓存 ${n(job.cache_write_tokens)}`);
  const output = `输出 ${n(job.completion_tokens)}${job.reasoning_tokens ? `（其中推理 ${n(job.reasoning_tokens)}）` : ""}`;
  return `${input.length > 1 ? `${input[0]}（${input.slice(1).join("，")}）` : input[0]} · ${output}`;
}

// What the record's requests to the service looked like (requests.jsonl in its directory holds each one), to tell
// why one run was slower than another: more requests (retranslated or split batches), retries, or slower answers
const REQUEST_REASONS = /** @type {Record<string, string>} */ ({
  repeat: "重新请求", retranslate: "单段补译", split: "拆分重译", terms: "术语表",
});

/** @param {Job} job */
function requestsTitle(job) {
  const r = job.requests;
  if (!r || !r.count) return "";
  const lines = [
    `请求 ${r.count} 次${r.retries ? `（重试 ${r.retries} 次）` : ""}${r.failed ? `，失败 ${r.failed} 次` : ""}`,
    `平均耗时 ${r.mean_latency.toFixed(1)} 秒 · P90 ${r.p90_latency.toFixed(1)} 秒 · 平均排队 ${r.mean_gate_wait.toFixed(1)} 秒`,
  ];
  const extra = Object.entries(r.reasons || {}).filter(([reason]) => reason in REQUEST_REASONS);
  if (extra.length) lines.push(extra.map(([reason, n]) => `${REQUEST_REASONS[reason]} ${n}`).join(" · "));
  return lines.join("\n");
}

/** @param {Job} job */
function expiryText(job) {
  const days = state.retentionDays;
  if (!days || ACTIVE.has(job.status)) return "";
  const left = (job.finished || job.created) + days * 86400 - Date.now() / 1000;
  if (left <= 0) return "即将删除";
  const d = Math.ceil(left / 86400);
  return d <= 3 ? `${d} 天后自动删除` : "";
}

/** @param {Job} job */
function langPair(job) {
  const from = job.source_lang === "auto" ? job.detected_lang || "auto" : job.source_lang;
  return `${langShort(from)} → ${langShort(job.target_lang)}`;
}

/** @param {string} code */
function langShort(code) {
  const l = langInfo(code);
  if (!l || code === "auto") return code === "auto" ? "自动" : code;
  return l.native || l.name;
}

/**
 * A new element, with its class and text when given.
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag
 * @param {string} [cls]
 * @param {string} [text]
 * @returns {HTMLElementTagNameMap[K]}
 */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// selectable: a row on the records page, with a checkbox on the left; recent records on the translate page have none
/**
 * @param {Job} job
 * @param {boolean} [selectable]
 */
function jobRow(job, selectable = false) {
  const li = el("li", selectable ? "job selectable" : "job");
  li.dataset.id = job.id;
  if (selectable) {
    const box = el("input", "job-check");
    box.type = "checkbox";
    box.checked = state.selected.has(job.id);
    box.setAttribute("aria-label", `选择「${job.filename}」`);
    box.addEventListener("change", () => {
      if (box.checked) state.selected.add(job.id);
      else state.selected.delete(job.id);
      renderSelection();
    });
    li.append(box);
  }
  const fmt = fmtOf(job.filename);
  li.append(el("div", `job-fmt ${fmt}`, fmt.toUpperCase()));

  const main = el("div", "job-main");
  const name = el("button", "job-name", job.filename);
  name.type = "button";
  // File names are often truncated; hovering shows the full name
  name.title = job.filename;
  name.disabled = !job.preview;
  name.addEventListener("click", () => openReader(job.id));
  main.append(name);

  const meta = el("div", "job-meta");
  // A fixed two-row layout: languages/mode/service on the first row, size/time on the second. With a single
  // flex-wrap row the wrap point drifts with the button width, cards wrap at different places, and a "·" can end
  // up at the start of a line
  for (const part of [
    langPair(job),
    job.mode === "translated" ? "仅译文" : "双语对照",
    serviceLabel(job),
  ].filter(Boolean)) {
    meta.append(el("span", "", part));
  }
  const meta2 = el("div", "job-meta");
  for (const part of [formatSize(job.size), timeAgo(job.created)].filter(Boolean)) {
    meta2.append(el("span", "", part));
  }
  // Record id, matching data/jobs/<id>/ when investigating problems. One click selects the whole string for copying
  const jobId = el("span", "job-id", job.id);
  jobId.title = "记录 ID";
  meta2.append(jobId);
  main.append(meta, meta2);

  const status = el("div", "job-status");
  status.append(el("span", `badge ${job.status}`, STATUS_TEXT[job.status] || job.status));
  if (ACTIVE.has(job.status) || (job.status === "paused" && job.total)) {
    // Round down: an unfinished job never shows 100% (1184/1188 would round to 100% and look stuck)
    const staged = inStage(job);
    const pct = staged ? stagePct(job) : Math.floor((job.done / job.total) * 100);
    const bar = el("div", `job-progress${job.status === "paused" ? " paused" : ""}`);
    const fill = el("div");
    fill.style.width = `${pct}%`;
    bar.append(fill);
    status.append(bar, el("span", "hint", staged ? stageText(job) : `${job.done} / ${job.total} 段 · ${pct}%`));
  } else if (job.status === "error" || job.status === "interrupted") {
    const err = el("span", `job-error ${job.status}`, explainError(job.error));
    err.title = job.error; // hover to see the raw error
    status.append(err);
  } else if (job.status === "idle") {
    status.append(el("span", "hint", ["缓存已清理", expiryText(job)].filter(Boolean).join(" · ")));
  } else {
    const parts = [job.total ? `${job.total} 段` : ""];
    if (job.skipped) parts.push(`${job.skipped} 段无需翻译`);
    if (job.failed) parts.push(`${job.failed} 段翻译失败，保留原文`);
    const tokens = tokensText(job);
    if (tokens) parts.push(tokens);
    const exp = expiryText(job);
    if (exp) parts.push(exp);
    const hint = el("span", "hint", parts.filter(Boolean).join(" · "));
    // Hover to see input/output separately, and how the requests went
    hint.title = [tokens ? tokensTitle(job) : "", requestsTitle(job)].filter(Boolean).join("\n");
    status.append(hint);
  }
  // Status has its own column between the details and the buttons on wide screens, below the details when narrow
  li.append(main, status);

  // Common actions take one click: read, preview, download (split button), continue or retry, retranslate;
  // clearing the cache and deleting go into "更多". The main action has a light highlight
  const actions = el("div", "job-actions");
  /**
   * @param {string} label
   * @param {string} cls
   * @param {(b: HTMLButtonElement) => void} onClick
   */
  const button = (label, cls, onClick) => {
    const b = el("button", cls, label);
    b.type = "button";
    b.addEventListener("click", () => onClick(b));
    return b;
  };
  const readable = job.preview && (job.status === "done" || job.done > 0);
  if (ACTIVE.has(job.status)) {
    // The first in the queue is already next
    if (job.queue_position > 1) actions.append(button("优先翻译", "btn sm", (b) => prioritizeJob(job.id, b)));
    if (job.preview) actions.append(button("查看进度", "btn sm", () => openReader(job.id)));
    actions.append(button("暂停", "btn sm", (b) => pauseJob(job.id, b)));
  } else if (job.status === "idle") {
    actions.append(button("开始翻译", "btn sm soft", () => retryJob(job.id)));
  } else {
    if (readable) actions.append(button("阅读", "btn sm", () => openReader(job.id)));
    if (job.status === "done") {
      actions.append(...[previewButton(job, "btn sm"), downloadButton(job, "btn sm soft")].filter((b) => b !== null));
    } else {
      actions.append(button(job.status === "error" ? "重试" : "继续翻译", "btn sm soft", () => retryJob(job.id)));
    }
    actions.append(button("重译", "btn sm", () => redoJob(job)));
  }
  actions.append(moreMenu(job));
  li.append(actions);
  return li;
}

// Records show "service · model"; the model is not repeated when the service name already contains it (CC Switch
// service names do not include the model)
/** @param {Job} job */
function serviceLabel(job) {
  if (!job.service_name) return "";
  return job.model && !job.service_name.includes(job.model) ? `${job.service_name} · ${job.model}` : job.service_name;
}

// Page sizes available on the records page. While jobs are running the list redraws every 2 seconds, one page only
const PAGE_SIZES = [10, 20, 50, 100];

function renderJobs() {
  if (deferWhileMenuOpen("jobs", renderJobs)) return;
  const all = state.jobs;
  const list = $("history-list");
  const pages = Math.max(1, Math.ceil(all.length / state.pageSize));
  state.historyPage = Math.min(Math.max(1, state.historyPage), pages);
  const start = (state.historyPage - 1) * state.pageSize;
  state.pageJobs = all.slice(start, start + state.pageSize);
  // Records no longer on this page (deleted, expired, pushed to another page) are no longer selected
  const ids = new Set(state.pageJobs.map((j) => j.id));
  for (const id of state.selected) if (!ids.has(id)) state.selected.delete(id);
  list.replaceChildren(...state.pageJobs.map((job) => jobRow(job, true)));
  // With a single page there are no paging buttons, but the page size can still be changed
  $("history-pager").hidden = all.length === 0;
  $("pager-pages").hidden = pages <= 1;
  $("page-info").textContent = `第 ${state.historyPage} / ${pages} 页`;
  byId("page-prev", HTMLButtonElement).disabled = state.historyPage <= 1;
  byId("page-next", HTMLButtonElement).disabled = state.historyPage >= pages;
  $("history-empty").hidden = all.length > 0;
  $("recent").hidden = all.length === 0;
  $("recent-list").replaceChildren(...all.slice(0, 3).map((job) => jobRow(job)));
  const running = all.filter((j) => ACTIVE.has(j.status)).length;
  $("running-badge").hidden = !running;
  $("running-badge").textContent = String(running);
  $("usage").textContent = `${all.length} 条记录 · 占用 ${formatSize(state.diskUsage) || "0 KB"}`;
  renderBulk(all);
  renderSelection();
}

// Selection bar above the list: pause all/continue all when nothing is ticked, deselect and delete selected once
// something is. Select all selects only the current page
function renderSelection() {
  const n = state.selected.size;
  const total = state.pageJobs.length;
  const all = byId("select-all", HTMLInputElement);
  $("history-bar").hidden = total === 0;
  all.checked = n > 0 && n === total;
  all.indeterminate = n > 0 && n < total;
  $("history-select-text").textContent = n ? `已选 ${n} 条` : "全选";
  $("history-select").hidden = n === 0;
  $("history-bulk").hidden = n > 0;
  $("history-bulk-text").hidden = n > 0;
  $("delete-selected").textContent = `删除选中（${n}）`;
}

$("select-all").addEventListener("change", () => {
  // When everything is selected a click deselects all; otherwise (including partial selection) it selects this page
  const allSelected = state.selected.size === state.pageJobs.length;
  state.selected.clear();
  if (!allSelected) for (const job of state.pageJobs) state.selected.add(job.id);
  renderJobs();
});

// Changing pages clears the selection, so records out of sight are never deleted along with the others
/** @param {number} page */
function showHistoryPage(page) {
  state.historyPage = page;
  state.selected.clear();
  renderJobs();
  $("view-history").scrollIntoView({ block: "start" });
}
$("page-prev").addEventListener("click", () => showHistoryPage(state.historyPage - 1));
$("page-next").addEventListener("click", () => showHistoryPage(state.historyPage + 1));

// The page size is stored in the browser; changing it returns to the first page
try {
  const saved = Number(localStorage.getItem("doc-translator-page-size"));
  if (PAGE_SIZES.includes(saved)) state.pageSize = saved;
} catch {
  /* ignore */
}
byId("page-size", HTMLSelectElement).value = String(state.pageSize);
$("page-size").addEventListener("change", () => {
  const size = Number(byId("page-size", HTMLSelectElement).value);
  if (!PAGE_SIZES.includes(size)) return;
  state.pageSize = size;
  try {
    localStorage.setItem("doc-translator-page-size", String(size));
  } catch {
    /* ignore */
  }
  showHistoryPage(1);
});

$("select-none").addEventListener("click", () => {
  state.selected.clear();
  renderJobs();
});

$("delete-selected").addEventListener("click", async () => {
  const chosen = state.jobs.filter((j) => state.selected.has(j.id));
  if (!chosen.length) return;
  const running = chosen.filter((j) => ACTIVE.has(j.status)).length;
  const msg = `删除选中的 ${chosen.length} 条翻译记录？原文件和译文都会被删除，无法恢复。` +
    (running ? `\n其中 ${running} 本正在翻译，将先停止翻译。` : "");
  if (!confirm(msg)) return;
  await busy(byId("delete-selected", HTMLButtonElement), "删除中…", async () => {
    try {
      const ids = chosen.map((j) => j.id);
      const result = /** @type {{ deleted: number }} */ (await api("/api/jobs/delete", { json: { ids } }));
      const gone = new Set(ids);
      state.jobs = state.jobs.filter((j) => !gone.has(j.id));
      state.selected.clear();
      if (reader.jobId !== null && gone.has(reader.jobId)) closeReader();
      renderJobs();
      toast(`已删除 ${result.deleted} 条`);
    } catch (err) {
      toast(`删除失败：${messageOf(err)}`);
      refreshJobs();
    }
  });
});

// Pause all / continue all: shown only when needed
/** @param {Job[]} all */
function renderBulk(all) {
  const active = all.filter((j) => ACTIVE.has(j.status)).length;
  const resumable = all.filter((j) => RESUMABLE.has(j.status)).length;
  for (const box of document.querySelectorAll("[data-bulk]")) {
    box.replaceChildren();
    if (active) {
      const b = el("button", "btn sm", `全部暂停（${active}）`);
      b.type = "button";
      b.addEventListener("click", () => pauseAll(b));
      box.append(b);
    }
    if (resumable) {
      const b = el("button", "btn sm", `全部继续（${resumable}）`);
      b.type = "button";
      b.addEventListener("click", () => resumeAll(b));
      box.append(b);
    }
  }
  /** @type {string[]} */
  const parts = [];
  const queued = all.filter((j) => j.queue_position).length;
  if (active - queued) parts.push(`${active - queued} 本正在翻译`);
  if (queued) parts.push(`${queued} 本排队`);
  if (resumable) parts.push(`${resumable} 本已暂停或中断`);
  $("history-bulk-text").textContent = parts.join("，");
}

/**
 * @param {string} id
 * @param {HTMLButtonElement} [btn]
 */
async function pauseJob(id, btn) {
  if (btn) btn.disabled = true;
  try {
    const job = /** @type {Job} */ (await api(`/api/jobs/${id}/pause`, { method: "POST" }));
    upsertJob(job);
    if (reader.jobId === id) refreshReader();
    toast("已暂停，已翻译的段落均已保留");
  } catch (err) {
    if (btn) btn.disabled = false;
    toast(`暂停失败：${messageOf(err)}`);
  }
}

/**
 * @param {string} id
 * @param {HTMLButtonElement} btn
 */
async function prioritizeJob(id, btn) {
  btn.disabled = true;
  try {
    upsertJob(/** @type {Job} */ (await api(`/api/jobs/${id}/prioritize`, { method: "POST" })));
    await refreshJobs();
    toast("已移到队列最前，将在下一个空位开始翻译");
  } catch (err) {
    btn.disabled = false;
    toast(`操作失败：${messageOf(err)}`);
  }
}

// After submitting or continuing: a job beyond the limit of jobs translating at once waits in the queue
/**
 * @param {Job} job
 * @param {string} started the text when it started at once
 */
function startedText(job, started) {
  return job.queue_position ? `已加入队列，排在第 ${job.queue_position} 位` : started;
}

/** @param {HTMLButtonElement} btn */
async function pauseAll(btn) {
  btn.disabled = true;
  try {
    const r = /** @type {{ paused: number }} */ (await api("/api/jobs/pause-all", { method: "POST" }));
    await refreshJobs();
    if (reader.jobId) refreshReader();
    toast(`已暂停 ${r.paused} 本`);
  } catch (err) {
    btn.disabled = false;
    toast(`暂停失败：${messageOf(err)}`);
  }
}

/** @param {HTMLButtonElement} btn */
async function resumeAll(btn) {
  btn.disabled = true;
  try {
    const r = /** @type {{ resumed: number, failed: { id: string, filename: string, error: string }[] }} */ (
      await api("/api/jobs/resume-all", { method: "POST" })
    );
    await refreshJobs();
    if (reader.jobId) refreshReader();
    const failed = r.failed.length ? `，${r.failed.length} 本无法继续：${r.failed[0].error}` : "";
    toast(`已继续 ${r.resumed} 本${failed}`);
  } catch (err) {
    btn.disabled = false;
    toast(`继续失败：${messageOf(err)}`);
  }
}

// Finishing jobs are announced: a toast while the page is in view, a system notification while it is in the
// background (when the browser allows it; permission is asked on the first submit). Only changes seen by this page
// count, so records finished before it was opened are not announced. The list and the reader poll separately and
// their answers can arrive out of order, so each run is announced once, by its finish time
/** @type {Set<string>} */
const seenActive = new Set();
/** @type {Map<string, number>} ids of jobs announced, with the finish time of the run announced */
const announced = new Map();

function canNotify() {
  return "Notification" in window && window.isSecureContext;
}

/**
 * @param {Job[]} jobs
 * @returns {Job[]}
 */
function finishedAmong(jobs) {
  /** @type {Job[]} */
  const out = [];
  for (const job of jobs) {
    if (ACTIVE.has(job.status)) {
      seenActive.add(job.id);
      continue;
    }
    const ended = job.status === "done" || job.status === "error";
    if (ended && seenActive.has(job.id) && announced.get(job.id) !== job.finished) {
      announced.set(job.id, job.finished);
      out.push(job);
    }
    seenActive.delete(job.id);
  }
  return out;
}

/** @param {Job[]} jobs */
function announceFinished(jobs) {
  if (!jobs.length) return;
  const failed = jobs.filter((j) => j.status === "error").length;
  const done = jobs.length - failed;
  const summary = [done ? `${done} 本翻译完成` : "", failed ? `${failed} 本翻译失败` : ""].filter(Boolean).join("，");
  const title = jobs.length === 1 ? (failed ? "翻译失败" : "翻译完成") : summary;
  const body = jobs.map((j) => j.filename).join("\n");
  if (document.visibilityState === "visible") {
    toast(jobs.length === 1 ? `「${jobs[0].filename}」${title}` : summary);
  } else if (canNotify() && Notification.permission === "granted") {
    const n = new Notification(title, { body, tag: jobs.length === 1 ? jobs[0].id : "doc-translator" });
    n.onclick = () => {
      window.focus();
      n.close();
    };
  }
}

/** @param {Job} job */
function upsertJob(job) {
  announceFinished(finishedAmong([job]));
  const i = state.jobs.findIndex((j) => j.id === job.id);
  if (i >= 0) state.jobs[i] = job;
  else state.jobs.unshift(job);
  renderJobs();
  scheduleJobsPoll();
}

async function refreshJobs() {
  try {
    const data = /** @type {JobList} */ (await api("/api/jobs"));
    announceFinished(finishedAmong(data.jobs));
    state.jobs = data.jobs;
    state.diskUsage = data.disk_usage;
    state.retentionDays = data.retention_days;
    renderJobs();
  } catch {
    /* keep the old list on network problems */
  }
  scheduleJobsPoll();
}

// Refresh the list every 2 seconds while jobs are running. The reader polls its own job; while it is open the list
// is refreshed every 10 seconds only for the other jobs, so their finishing is still announced. The reader's poll
// calls this every second, so a timer already waiting is kept, or it would never fire
/** @type {number | undefined} */
let jobsTimer;
function scheduleJobsPoll() {
  if (!state.jobs.some((j) => ACTIVE.has(j.status) && j.id !== reader.jobId)) {
    clearTimeout(jobsTimer);
    jobsTimer = undefined;
    return;
  }
  if (reader.jobId && jobsTimer) return;
  clearTimeout(jobsTimer);
  jobsTimer = setTimeout(() => {
    jobsTimer = undefined;
    refreshJobs();
  }, reader.jobId ? 10000 : 2000);
}

/** @param {Job} job */
async function deleteJob(job) {
  const running = ACTIVE.has(job.status);
  const msg = running
    ? `「${job.filename}」正在翻译。删除将停止翻译并删除全部相关文件，是否继续？`
    : `删除「${job.filename}」的翻译记录？原文件和译文都会被删除，无法恢复。`;
  if (!confirm(msg)) return;
  try {
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    state.jobs = state.jobs.filter((j) => j.id !== job.id);
    renderJobs();
    if (reader.jobId === job.id) closeReader();
    toast("已删除");
  } catch (err) {
    toast(`删除失败：${messageOf(err)}`);
  }
}

/**
 * @param {string} id
 * @param {{ fresh?: boolean, service_id?: string }} [options] fresh: clear the record's cache first
 */
async function retryJob(id, options = {}) {
  try {
    const job = /** @type {Job} */ (await api(`/api/jobs/${id}/retry`, { method: "POST", json: options }));
    upsertJob(job);
    toast(startedText(job, "已重新开始翻译"));
  } catch (err) {
    toast(`无法重新翻译：${messageOf(err)}`);
  }
}

// Regenerate a finished record with the service's current settings. By default existing translations are reused and
// only paragraphs without one are requested
/** @param {Job} job */
function redoJob(job) {
  const dialog = byId("redo-dialog", HTMLDialogElement);
  // Retranslation uses the service's currently selected model, so the original model name is not shown
  const service = job.service_name;
  $("redo-text").textContent =
    `使用${service ? `「${service}」` : "原翻译服务"}的当前设置重新生成「${job.filename}」。` +
    "已有译文的段落直接复用，仅对有变化的段落发送请求。开始后，现有的译文文件将被删除。";
  byId("redo-fresh", HTMLInputElement).checked = false;
  dialog.returnValue = "";
  dialog.addEventListener("close", () => {
    if (dialog.returnValue === "ok") retryJob(job.id, { fresh: byId("redo-fresh", HTMLInputElement).checked });
  }, { once: true });
  dialog.showModal();
}

// Clear cache: first ask the server for a list (cache entries, file sizes, records sharing the same file), and
// clear after confirmation
/** @param {Job} job */
async function purgeJob(job) {
  const dialog = byId("purge-dialog", HTMLDialogElement);
  const list = $("purge-list");
  const ok = byId("purge-ok", HTMLButtonElement);
  const boxes = byId("purge-boxes", HTMLInputElement);
  $("purge-text").textContent =
    `「${job.filename}」的翻译缓存和生成结果将被清除，记录保留并回到“未翻译”。之后重新翻译时，所有段落都会重新请求翻译服务。`;
  list.replaceChildren(el("div", "purge-loading", "正在统计…"));
  $("purge-shared").hidden = true;
  $("purge-boxes-row").hidden = true;
  boxes.checked = false;
  boxes.onchange = null;
  ok.disabled = true;
  dialog.returnValue = "";
  dialog.addEventListener("close", () => {
    if (dialog.returnValue === "ok") clearJobCache(job, boxes.checked);
  }, { once: true });
  dialog.showModal();

  /** @type {JobCache} */
  let info;
  try {
    info = /** @type {JobCache} */ (await api(`/api/jobs/${job.id}/cache`));
  } catch (err) {
    list.replaceChildren(el("div", "purge-loading", `无法统计：${messageOf(err)}`));
    return;
  }
  if (!dialog.open) return;
  /** @type {[string, string][]} */
  const rows = [
    ["缓存的译文", info.entries ? `${info.entries.toLocaleString()} 条` : "无"],
    ["生成的文件", info.generated_bytes
      ? [info.outputs ? `译文 ${info.outputs} 个` : "", `共 ${formatSize(info.generated_bytes)}`].filter(Boolean).join(" · ")
      : "无"],
  ];
  list.replaceChildren(...rows.flatMap(([k, v]) => [el("dt", "", k), el("dd", "", v)]));
  if (info.shared) {
    $("purge-shared").textContent =
      `另有 ${info.shared} 条记录使用同一文件，它们共用翻译缓存和版面分析结果，之后重译也需要重新请求翻译服务。`;
    $("purge-shared").hidden = false;
  }
  if (info.boxes_bytes) {
    $("purge-boxes-text").textContent = `同时清除版面识别结果（${formatSize(info.boxes_bytes)}），重新识别需要几分钟`;
    $("purge-boxes-row").hidden = false;
  }
  const empty = () => !info.entries && !info.generated_bytes && !boxes.checked;
  ok.disabled = empty();
  boxes.onchange = () => (ok.disabled = empty());
}

/**
 * @param {Job} job
 * @param {boolean} boxes also clear the layout model's results
 */
async function clearJobCache(job, boxes) {
  try {
    const result = /** @type {Job & { removed: number }} */ (
      await api(`/api/jobs/${job.id}/cache${boxes ? "?boxes=true" : ""}`, { method: "DELETE" })
    );
    const { removed, ...updated } = result;
    if (reader.jobId === job.id) closeReader();
    upsertJob(updated);
    toast(removed ? `已清理 ${removed.toLocaleString()} 条缓存` : "已清理");
  } catch (err) {
    toast(`清理失败：${messageOf(err)}`);
    refreshJobs();
  }
}

// Clear unused cache: keep the translations records use, and clear the rest (web pages, records deleted earlier).
// Uses the record cache dialog with its own text
async function sweepCache() {
  const dialog = byId("purge-dialog", HTMLDialogElement);
  const list = $("purge-list");
  const ok = byId("purge-ok", HTMLButtonElement);
  $("purge-title").textContent = "清理无用缓存";
  $("purge-text").textContent =
    "清除网页翻译的译文，以及已删除记录留下的译文；翻译记录用到的译文保留，记录本身和生成的文件不受影响。";
  list.replaceChildren(el("div", "purge-loading", "正在统计…"));
  $("purge-shared").hidden = true;
  $("purge-boxes-row").hidden = true;
  ok.disabled = true;
  dialog.returnValue = "";
  dialog.addEventListener("close", async () => {
    $("purge-title").textContent = "清理缓存";
    if (dialog.returnValue !== "ok") return;
    try {
      const { removed } = /** @type {CacheSweep} */ (await api("/api/cache/unused", { method: "DELETE" }));
      toast(removed ? `已清理 ${removed.toLocaleString()} 条缓存` : "没有需要清理的缓存");
    } catch (err) {
      toast(`清理失败：${messageOf(err)}`);
    }
  }, { once: true });
  dialog.showModal();

  /** @type {CacheSweep} */
  let info;
  try {
    info = /** @type {CacheSweep} */ (await api("/api/cache/unused"));
  } catch (err) {
    list.replaceChildren(el("div", "purge-loading", `无法统计：${messageOf(err)}`));
    return;
  }
  if (!dialog.open) return;
  list.replaceChildren(...[
    ["可清除", info.removed ? `${info.removed.toLocaleString()} 条` : "无"],
    ...(info.untracked ? [["无法判断", `${info.untracked.toLocaleString()} 条，保留`]] : []),
  ].flatMap(([k, v]) => [el("dt", "", k), el("dd", "", v)]));
  if (info.untracked) {
    $("purge-shared").textContent =
      "“无法判断”的是较早版本缓存的译文，当时没有记下属于哪条记录，可能仍被记录使用，所以不清除；缓存满 5 万条时会自动淘汰最旧的。";
    $("purge-shared").hidden = false;
  }
  if (state.jobs.some((j) => ACTIVE.has(j.status))) {
    $("purge-shared").textContent = "有任务正在翻译，请等它完成或暂停后再清理。";
    $("purge-shared").hidden = false;
    return;
  }
  ok.disabled = !info.removed;
}

$("sweep-cache").addEventListener("click", sweepCache);

// Jobs translating at once; the others wait in the queue
const MAX_JOBS = 8;
const maxJobsSelect = byId("max-jobs", HTMLSelectElement);
for (let n = 1; n <= MAX_JOBS; n++) maxJobsSelect.add(new Option(`${n} 本`, String(n)));
maxJobsSelect.addEventListener("change", async () => {
  try {
    const json = { max_jobs: Number(maxJobsSelect.value) };
    const r = /** @type {Settings} */ (await api("/api/settings", { method: "PUT", json }));
    state.maxJobs = r.max_jobs;
    toast(`已保存：同时翻译 ${r.max_jobs} 本，其余排队`);
    refreshJobs();
  } catch (err) {
    toast(`保存失败：${messageOf(err)}`);
  }
  maxJobsSelect.value = String(state.maxJobs);
});

// Retention period
function renderRetention() {
  const sel = byId("retention-days", HTMLSelectElement);
  const days = String(state.retentionDays ?? 30);
  const preset = [...sel.options].some((o) => o.value === days);
  sel.value = preset ? days : "custom";
  $("retention-custom").hidden = preset;
  byId("retention-custom", HTMLInputElement).value = preset ? "" : days;
  $("retention-save").hidden = preset;
}

// As with toast: cancel the previous timer first, so a newer message is not cleared by mistake
/** @type {number | undefined} */
let retentionStatusTimer;
/** @param {number} days */
async function saveRetention(days) {
  const status = $("retention-status");
  try {
    const r = /** @type {Settings} */ (await api("/api/settings", { method: "PUT", json: { retention_days: days } }));
    state.retentionDays = r.retention_days;
    renderRetention();
    status.textContent = r.retention_days ? "已保存" : "已保存，记录将永久保留";
    refreshJobs();
  } catch (err) {
    status.textContent = messageOf(err);
    renderRetention();
  }
  clearTimeout(retentionStatusTimer);
  retentionStatusTimer = setTimeout(() => (status.textContent = ""), 2500);
}

$("retention-days").addEventListener("change", () => {
  const value = byId("retention-days", HTMLSelectElement).value;
  if (value === "custom") {
    $("retention-custom").hidden = false;
    $("retention-save").hidden = false;
    $("retention-custom").focus();
    return;
  }
  saveRetention(Number(value));
});
$("retention-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const raw = byId("retention-custom", HTMLInputElement).value.trim();
  const days = Number(raw);
  // 0 means keep forever and is valid; only empty, non-integer and negative values are rejected
  if (raw === "" || !Number.isInteger(days) || days < 0) {
    toast("保留天数需为不小于 0 的整数（0 表示永久保留）");
    return;
  }
  saveRetention(days);
});

// ---------- Reader (preview while translating) ----------
// The backend records translations in order of completion; the frontend fetches increments by version every second
// and updates only changed paragraphs, without rebuilding the file

/**
 * A chapter heading of the document (paragraph i), and the same with its line in the outline.
 * @typedef {{ i: number, level: number, div: HTMLDivElement }} Heading
 * @typedef {Heading & { li: HTMLLIElement, label: HTMLSpanElement }} OutlineEntry
 */

/**
 * The record open in the reader. nodes are its paragraphs, in document order; version is the preview's, -1 before
 * the first fetch.
 * @type {{
 *   jobId: string | null, job: Job | null, version: number, nodes: HTMLDivElement[], headings: Heading[],
 *   done: number, total: number, timer: number | undefined, focusTimer: number | undefined, lastFocus: number,
 *   following: boolean, autoScrolling: boolean, outline?: OutlineEntry[]
 * }}
 */
const reader = {
  jobId: null, job: null, version: -1, nodes: [], headings: [], done: 0, total: 0,
  timer: undefined, focusTimer: undefined, lastFocus: -1, following: true, autoScrolling: false,
};
const HEADING = /^h[1-6]$/;

/** @param {string} id */
function openReader(id) {
  closeReader(true);
  reader.jobId = id;
  reader.following = true;
  document.body.classList.add("reading");
  $("reader").hidden = false;
  $("reader-content").replaceChildren(el("div", "hint", "加载中…"));
  $("outline-list").replaceChildren();
  $("reader-notice").hidden = true;
  $("follow-btn").hidden = true;
  history.replaceState(null, "", `#read/${id}`);
  readerTick();
}

/** @param {boolean} [keepHash] keep #read/<id> in the address, as another record is opened next */
function closeReader(keepHash) {
  clearTimeout(reader.timer);
  Object.assign(reader, { jobId: null, job: null, version: -1, nodes: [], headings: [], done: 0, total: 0, lastFocus: -1 });
  $("reader").hidden = true;
  $("reader").classList.remove("show-outline");
  document.body.classList.remove("reading");
  if (!keepHash) history.replaceState(null, "", location.pathname);
}

$("reader-back").addEventListener("click", () => {
  closeReader();
  refreshJobs();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && reader.jobId && !$("reader").hidden) {
    if ($("reader").classList.contains("show-outline")) $("reader").classList.remove("show-outline");
    else {
      closeReader();
      refreshJobs();
    }
  }
});

// Refresh the reader right after pausing or continuing (without waiting for the next poll), avoiding two polling
// loops
function refreshReader() {
  clearTimeout(reader.timer);
  readerTick();
}

async function readerTick() {
  const id = reader.jobId;
  if (!id) return;
  try {
    const [job, data] = /** @type {[Job, Preview]} */ (await Promise.all([
      api(`/api/jobs/${id}`),
      api(`/api/jobs/${id}/preview?since=${reader.version}`),
    ]));
    if (reader.jobId !== id) return;
    reader.job = job;
    upsertJob(job);
    renderReaderBar(job);
    if (data.ready) {
      const initial = reader.version < 0;
      // The first fetch (since=-1) always has the paragraphs
      if (initial) buildReader(/** @type {Segment[]} */ (data.segments));
      applyUpdates(data.updates, initial);
      reader.version = data.version;
    } else if (reader.version < 0 && !ACTIVE.has(job.status)) {
      $("reader-content").replaceChildren(el("div", "hint", "此记录没有可预览的内容。"));
    }
    renderNotice(job);
    // Stop polling when the job is no longer translating and the preview is not ready. Encrypted or scanned PDFs
    // stay ready=false; after continuing, refreshReader / openReader fetch again.
    if (ACTIVE.has(job.status)) reader.timer = setTimeout(readerTick, 1000);
  } catch (err) {
    if (reader.jobId !== id) return;
    $("reader-notice").hidden = false;
    $("reader-notice").textContent = `无法读取：${messageOf(err)}`;
    // The record has been deleted (for example in another tab); further fetches would also be 404, so stop
    if (err instanceof ApiError && err.status === 404) return;
    // Keep polling after an error with a 4-second backoff, so one network hiccup does not stop updates for good
    reader.timer = setTimeout(readerTick, 4000);
  }
}

// Preparation before translation and writing after it: layout analysis and writing of a whole book take minutes
// each, so the current page is shown
/** @param {Job} job */
function inStage(job) {
  return !job.total || (job.done >= job.total && Boolean(job.stage && job.stage.name));
}

/** @param {Job} job */
function stageText(job) {
  if (job.queue_position) return `排在队列第 ${job.queue_position} 位`;
  const stage = job.stage;
  if (!stage || !stage.name) return "解析文档…";
  return stage.total ? `${stage.name} ${stage.done} / ${stage.total} ${stage.unit}` : `${stage.name}…`;
}

/** @param {Job} job */
function stagePct(job) {
  const stage = job.stage;
  return stage && stage.total ? Math.floor((stage.done / stage.total) * 100) : 0;
}

/** @param {Job} job */
function renderReaderBar(job) {
  if (deferWhileMenuOpen("reader", () => renderReaderBar(reader.job || job))) return;
  $("reader-name").textContent = job.filename;
  const meta = [langPair(job), serviceLabel(job)];
  if (ACTIVE.has(job.status)) {
    meta.push(inStage(job) ? stageText(job) : `翻译中 ${job.done} / ${job.total} 段`);
  } else {
    meta.push(STATUS_TEXT[job.status]);
  }
  const tokens = tokensText(job);
  if (tokens) meta.push(tokens);
  const metaEl = $("reader-meta");
  metaEl.textContent = meta.filter(Boolean).join(" · ");
  metaEl.title = tokens ? tokensTitle(job) : "";
  const pct = job.status === "done" ? 100 : inStage(job) ? stagePct(job) : (job.done / job.total) * 100;
  const bar = $("reader-progress-bar");
  bar.style.width = `${pct}%`;
  // The bar's track, hidden with it
  if (bar.parentElement) bar.parentElement.hidden = job.status === "done";

  const actions = $("reader-actions");
  actions.replaceChildren();
  const downloads = $("reader-downloads");
  downloads.replaceChildren();
  const outlineBtn = el("button", "btn ghost icon-btn", "☰");
  outlineBtn.type = "button";
  outlineBtn.id = "outline-toggle";
  outlineBtn.setAttribute("aria-label", "目录");
  outlineBtn.addEventListener("click", () => $("reader").classList.toggle("show-outline"));
  actions.append(outlineBtn);
  if (ACTIVE.has(job.status)) {
    const pause = el("button", "btn sm");
    pause.type = "button";
    pause.append("❙❙ ", el("span", "", "暂停"));
    pause.addEventListener("click", () => pauseJob(job.id, pause));
    actions.append(pause);
  }
  downloads.append(...[previewButton(job, "btn sm"), downloadButton(job, "btn primary sm")].filter((b) => b !== null));
}

// Turn the raw error returned by the API into a readable sentence; the original goes into "详细信息", still
// available when investigating
/** @param {string} raw */
function explainError(raw) {
  const text = raw || "";
  if (/INVALID_MODEL|Invalid model|model.*(not found|does not exist|不存在)/i.test(text)) {
    return "翻译服务暂时拒绝了该模型的请求。中转服务偶尔出现这种情况，通常稍后重试即可；若持续出现，请更换模型或服务。";
  }
  if (/\b401\b|invalid.*api.?key|Incorrect API key|unauthori[sz]ed/i.test(text)) return "API Key 无效或已过期，请在「翻译服务」中检查。";
  if (/\b403\b|forbidden|permission/i.test(text)) return "该 API Key 无权使用此模型。";
  if (/\b429\b|rate.?limit|quota|余额|insufficient/i.test(text)) return "请求过于频繁或额度不足。请稍后重试，或在翻译服务设置中降低并发数。";
  if (/不是 JSON|API 地址/.test(text)) return "API 地址可能有误（通常以 /v1 结尾），请在「翻译服务」中检查。";
  if (/服务重启/.test(text)) return "服务重启导致翻译中断。已翻译的段落已缓存，继续翻译时直接复用。";
  if (/请求失败|timed? ?out|ConnectError|超时/i.test(text)) return "无法连接翻译服务，请检查网络或服务地址后重试。";
  return "翻译未完成。";
}

/** @param {Job} job */
function renderNotice(job) {
  const notice = $("reader-notice");
  notice.replaceChildren();
  notice.className = "reader-notice";
  const stopped = job.status === "error" || RESUMABLE.has(job.status);
  if (stopped) {
    const paused = job.status === "paused";
    notice.classList.add("is-action", paused ? "is-paused" : "is-error");
    const body = el("div", "notice-body");
    const done = job.done ? `已翻译 ${job.done} / ${job.total} 段。` : "";
    const title = paused ? "已暂停" : job.status === "interrupted" ? "翻译已中断" : "翻译失败";
    const text = paused
      ? `${done}继续翻译时，已翻译的段落直接复用，其余段落使用翻译服务当前选择的模型。`
      : `${explainError(job.error)}${done ? " " + done : ""}`;
    body.append(el("div", "notice-title", title), el("div", "notice-text", text));
    if (job.error && job.status === "error") {
      const det = el("details", "notice-detail");
      det.append(el("summary", "", "详细信息"), el("code", "", job.error));
      body.append(det);
    }
    const actions = el("div", "notice-actions");
    const services = state.services.filter((s) => s.enabled);
    const pick = el("select", "notice-service");
    pick.setAttribute("aria-label", "重新翻译使用的服务");
    for (const s of services) pick.add(new Option(s.name, s.id));
    pick.value = services.some((s) => s.id === job.service_id) ? job.service_id : state.defaultId;
    // The same action uses the same word as on the records page; "重新翻译" is easily confused with translating
    // everything again under "重译"
    const retry = el("button", "btn primary sm", job.status === "error" ? "重试" : "继续翻译");
    retry.type = "button";
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      try {
        const json = { service_id: pick.value };
        const next = /** @type {Job} */ (await api(`/api/jobs/${job.id}/retry`, { method: "POST", json }));
        upsertJob(next);
        openReader(job.id);
      } catch (err) {
        retry.disabled = false;
        toast(`无法重新翻译：${messageOf(err)}`);
      }
    });
    actions.append(pick, retry);
    notice.append(body, actions);
  } else if (job.same_lang) {
    notice.append(el("div", "notice-text", `文档语言与目标语言相同（${langLabel(job.target_lang)}），可能没有需要翻译的内容。`));
  }
  notice.hidden = !notice.childNodes.length;
  // Once the job stops, paragraphs not yet translated no longer show the blinking loading state and switch to a
  // static untranslated state
  $("reader-content").classList.toggle("stopped", !ACTIVE.has(job.status));
}

/** @param {Segment[]} segments */
function buildReader(segments) {
  const frag = document.createDocumentFragment();
  const lang = reader.job?.target_lang || "";
  /** @type {HTMLDetailsElement | null} */
  let toc = null;
  reader.headings = [];
  reader.nodes = segments.map((seg, i) => {
    const div = el("div", `seg k-${seg.k}`);
    div.dataset.i = String(i);
    if (!seg.s.trim()) {
      div.hidden = true;
      frag.append(div);
      return div;
    }
    div.classList.add("pending");
    const src = el("div", "src", previewText(reader.job, seg.s));
    const dst = el("div", "dst");
    if (lang) dst.lang = lang;
    div.append(src, dst);
    // The book's own table of contents (nav / ncx) goes into a collapsible block, so it does not interrupt the text
    if (seg.k === "toc") {
      if (!toc) {
        toc = el("details", "toc-block");
        toc.append(el("summary", "", "书中目录"));
        frag.append(toc);
      }
      toc.append(div);
    } else {
      frag.append(div);
      if (HEADING.test(seg.k)) reader.headings.push({ i, level: Number(seg.k[1]), div });
    }
    return div;
  });
  $("reader-content").replaceChildren(frag);
  reader.total = segments.filter((s) => s.s.trim()).length;
  reader.done = 0;
  buildOutline();
}

// Outline on the left: headings from the document, up to three levels; hidden when there are too few headings
function buildOutline() {
  const list = $("outline-list");
  const levels = [...new Set(reader.headings.map((h) => h.level))].sort().slice(0, 3);
  const items = reader.headings.filter((h) => levels.includes(h.level));
  // With too few chapter headings the outline is hidden and the text takes the full width
  $("reader").classList.toggle("no-outline", items.length < 2);
  /** @type {OutlineEntry[]} */
  const outline = [];
  list.replaceChildren(
    ...items.map((h) => {
      const li = el("li", `l${levels.indexOf(h.level) + 1}`);
      const btn = el("button");
      btn.type = "button";
      const label = el("span", "label", query(h.div, ".src").textContent ?? "");
      btn.append(el("span", "dot"), label);
      btn.addEventListener("click", () => {
        $("reader").classList.remove("show-outline");
        jumpTo(h.i);
      });
      li.append(btn);
      outline.push(Object.assign(h, { li, label }));
      return li;
    }),
  );
  reader.outline = outline;
  if (items.length < 2) $("outline-list").replaceChildren(el("li", "hint", "无章节标题"));
}

/**
 * @param {PreviewUpdate[]} updates
 * @param {boolean} initial the first fetch, shown without highlighting or scrolling
 */
function applyUpdates(updates, initial) {
  /** @type {HTMLDivElement | null} */
  let last = null;
  for (const [i, text, skipped] of updates) {
    const div = reader.nodes[i];
    if (!div || !div.classList.contains("pending")) continue;
    div.classList.remove("pending");
    div.classList.toggle("skipped", !!skipped);
    query(div, ".dst").textContent = skipped ? "" : previewText(reader.job, text);
    if (!initial) {
      div.classList.add("fresh");
      setTimeout(() => div.classList.remove("fresh"), 1500);
    }
    reader.done += 1;
    last = div;
  }
  refreshOutlineState();
  // Follow translation progress: bring the latest translated paragraph into view; paused when the user scrolls,
  // resumed with "跟随"
  if (last && !initial && reader.following) {
    reader.autoScrolling = true;
    last.scrollIntoView({ block: "center" });
    setTimeout(() => (reader.autoScrolling = false), 80);
  }
  const active = reader.job && ACTIVE.has(reader.job.status);
  $("follow-btn").hidden = reader.following || !active;
}

// Dots in the outline: gray not started, blue translating, green done; once a chapter heading is translated the
// outline shows the translation
function refreshOutlineState() {
  const items = reader.outline || [];
  items.forEach((h, n) => {
    const end = n + 1 < items.length ? items[n + 1].i : reader.nodes.length;
    let pending = 0;
    let total = 0;
    for (let k = h.i; k < end; k++) {
      const d = reader.nodes[k];
      if (!d || d.hidden) continue;
      total += 1;
      if (d.classList.contains("pending")) pending += 1;
    }
    h.li.classList.toggle("complete", pending === 0);
    h.li.classList.toggle("partial", pending > 0 && pending < total);
    const dst = query(h.div, ".dst").textContent;
    if (dst && state.readerView !== "src") h.label.textContent = dst;
  });
}

/** @param {number} i */
function jumpTo(i) {
  const div = reader.nodes[i];
  if (!div) return;
  setFollowing(false);
  div.scrollIntoView({ block: "start" });
  sendFocus(i);
}

/** @param {boolean} on */
function setFollowing(on) {
  reader.following = on;
  const active = reader.job && ACTIVE.has(reader.job.status);
  $("follow-btn").hidden = on || !active;
}

$("follow-btn").addEventListener("click", () => {
  setFollowing(true);
  const lastDone = [...reader.nodes].reverse().find((n) => !n.hidden && !n.classList.contains("pending"));
  (lastDone || reader.nodes[0])?.scrollIntoView({ block: "center" });
});

function visibleIndex() {
  const top = $("reader-body").getBoundingClientRect().top;
  const div = reader.nodes.find((n) => !n.hidden && n.getBoundingClientRect().bottom > top + 8);
  return div ? Number(div.dataset.i) : 0;
}

// When the user scrolls in the reader, ask the backend to translate the visible part first
/** @param {number} index */
function sendFocus(index) {
  if (!reader.jobId || index === reader.lastFocus || !reader.job || !ACTIVE.has(reader.job.status)) return;
  reader.lastFocus = index;
  fetch(`/api/jobs/${reader.jobId}/focus`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index }),
  }).catch(() => {});
}

function highlightCurrentHeading() {
  const items = reader.outline || [];
  const top = $("reader-body").getBoundingClientRect().top + 80;
  /** @type {OutlineEntry | null} */
  let current = null;
  for (const h of items) {
    if (h.div.getBoundingClientRect().top <= top) current = h;
    else break;
  }
  for (const h of items) h.li.classList.toggle("current", h === current);
  current?.li.scrollIntoView({ block: "nearest" });
}

$("reader-body").addEventListener("scroll", () => {
  if (!reader.autoScrolling) setFollowing(false);
  clearTimeout(reader.focusTimer);
  reader.focusTimer = setTimeout(() => {
    highlightCurrentHeading();
    if (!reader.following) sendFocus(visibleIndex());
  }, 200);
});

// Display mode: side by side / translation only / original only
for (const r of queryAll(document, 'input[name="reader-view"]', HTMLInputElement)) {
  r.addEventListener("change", () => setReaderView(r.value));
}
/** @param {string} view "both", "dst" or "src" */
function setReaderView(view) {
  state.readerView = view;
  const content = $("reader-content");
  content.classList.remove("view-both", "view-dst", "view-src");
  content.classList.add(`view-${view}`);
  for (const r of queryAll(document, 'input[name="reader-view"]', HTMLInputElement)) r.checked = r.value === view;
  try {
    localStorage.setItem("doc-translator-reader-view", view);
  } catch {
    /* ignore */
  }
}


// ---------- Initialization ----------

async function init() {
  const [langs, providers] = /** @type {[Language[], Record<string, Provider>]} */ (
    await Promise.all([api("/api/languages"), api("/api/providers")])
  );
  state.providers = providers;
  state.languages = langs;
  const sourceLang = field(form, "source_lang", HTMLSelectElement);
  const targetLang = field(form, "target_lang", HTMLSelectElement);
  for (const lang of langs) {
    const label = langLabel(lang.code);
    if (lang.code === "auto") {
      sourceLang.add(new Option("自动检测", "auto"));
      continue;
    }
    sourceLang.add(new Option(label, lang.code));
    targetLang.add(new Option(label, lang.code));
  }
  for (const [key, meta] of Object.entries(providers)) {
    if (meta.llm) byId("add-provider", HTMLSelectElement).add(new Option(meta.label, key));
  }
  const saved = loadSettings();
  for (const name of ["source_lang", "target_lang", "mode"]) {
    if (saved[name] !== undefined) control(form, name).value = saved[name];
  }
  if (!targetLang.value) targetLang.value = defaultTargetLang();
  let view = "translate";
  let readerView = "both";
  try {
    view = localStorage.getItem("doc-translator-view") || view;
    readerView = localStorage.getItem("doc-translator-reader-view") || readerView;
  } catch {
    /* ignore */
  }
  setReaderView(["both", "dst", "src"].includes(readerView) ? readerView : "both");
  await Promise.all([
    loadServices(),
    loadGlossaries(),
    refreshJobs(),
    api("/api/settings").then((answer) => {
      const s = /** @type {Settings} */ (answer);
      state.retentionDays = s.retention_days;
      state.maxJobs = s.max_jobs;
    }),
  ]);
  renderRetention();
  maxJobsSelect.value = String(state.maxJobs);
  renderJobs();
  checkLangPair();
  showView(view);
  // On reload, return to the book being viewed (jobs with preview=false have no preview content and are not
  // restored)
  const m = location.hash.match(/^#read\/(\w+)$/);
  if (m && state.jobs.some((j) => j.id === m[1] && j.preview !== false)) openReader(m[1]);
}

// ---------- Layout model, formula detection model ----------
// Both cards have the same structure, with element ids starting with layout- / formula- and the endpoints
// /api/layout-model and /api/formula-model

const MODEL_KINDS = ["layout", "formula"];
/** @type {Record<string, string>} */
const MODEL_NAMES = { layout: "版面模型", formula: "公式检测模型" };
/** @type {Record<string, number>} timers polling a model's download or int8 generation, by kind */
const modelPolls = {};

/** @param {number} bytes */
function formatMB(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/**
 * @typedef {{ keepForm?: boolean }} RenderModelOptions keepForm: leave the settings form as the user has it
 */

/**
 * @param {string} kind "layout" or "formula"
 * @param {ModelStatus} m
 * @param {RenderModelOptions} [options]
 */
function renderModel(kind, m, { keepForm = false } = {}) {
  /** @param {string} name */
  const el = (name) => $(`${kind}-${name}`);
  const runtime = el("runtime");
  runtime.replaceChildren();
  if (m.runtime) {
    runtime.textContent = `onnxruntime ${m.runtime}`;
  } else {
    const warn = document.createElement("span");
    warn.className = "warn";
    warn.textContent = "未安装 onnxruntime，请在项目目录运行 ";
    const code = document.createElement("code");
    code.textContent = m.install_hint;
    runtime.append(warn, code, " 后重启服务");
  }
  runtime.classList.toggle("is-ok", Boolean(m.runtime));
  const file = el("file");
  file.replaceChildren();
  file.classList.toggle("is-ok", m.file.exists);
  // Show only the file name, with the full path in the hover tooltip; the default location is also visible in the
  // path field of the advanced settings
  const where = document.createElement("code");
  where.textContent = m.resolved.split(/[\\/]/).pop() ?? "";
  where.title = m.resolved;
  if (m.file.exists) {
    file.append(where, `（${formatMB(m.file.size)}）`);
  } else {
    const warn = document.createElement("span");
    warn.className = "warn";
    warn.textContent = "文件不存在：";
    file.append(warn, where);
  }
  const onOff = el("state");
  onOff.textContent = m.enabled ? "已开启" : "未开启";
  onOff.classList.toggle("is-on", m.enabled);
  if (!keepForm) {
    byId(`${kind}-enabled`, HTMLInputElement).checked = m.enabled;
    const path = byId(`${kind}-path`, HTMLInputElement);
    path.value = m.path;
    path.placeholder = `留空时使用默认位置：${m.default_path}`;
    // Only the formula detection model has a size, and only it has workers and threads (with auto)
    if (m.size) byId(`${kind}-size`, HTMLSelectElement).value = String(m.size);
    if (m.precision) byId(`${kind}-precision`, HTMLSelectElement).value = m.precision;
    for (const select of queryAll($(`${kind}-form`), "select.has-hint", HTMLSelectElement)) showOptionHint(select);
    if (m.auto) {
      for (const name of /** @type {const} */ (["workers", "threads"])) {
        const input = byId(`${kind}-${name}`, HTMLInputElement);
        input.value = String(m[name] || m.auto[name]);
        input.dataset.auto = String(m.auto[name]);
      }
    }
    // Expand the advanced settings when they hold changed values, so they are not hidden
    if (m.path || m.workers || m.threads) byId(`${kind}-advanced`, HTMLDetailsElement).open = true;
    const select = byId(`${kind}-source`, HTMLSelectElement);
    if (!select.options.length) {
      for (const source of m.sources) select.add(new Option(source.label, source.id));
    }
  }
  const d = m.download;
  const status = el("download-status");
  status.classList.toggle("warn", Boolean(d.error));
  if (d.running) {
    status.textContent = d.total ? `下载中 ${formatMB(d.done)} / ${formatMB(d.total)}` : `下载中 ${formatMB(d.done)}`;
  } else if (d.error) {
    status.textContent = `下载失败：${d.error}`;
  } else if (d.source && m.file.exists) {
    status.textContent = "下载完成";
  } else {
    status.textContent = "";
  }
  const download = byId(`${kind}-download`, HTMLButtonElement);
  download.disabled = d.running;
  download.textContent = m.file.exists ? "重新下载" : "下载模型";
  // Without a model file, downloading is the first thing to do on this page
  download.classList.toggle("primary", !m.file.exists);
  if (m.int8) renderInt8(kind, m);
  clearTimeout(modelPolls[kind]);
  const busyNow = d.running || Boolean(m.int8 && m.int8.state.running);
  if (busyNow) modelPolls[kind] = setTimeout(() => loadModel(kind, { keepForm: true }), 1000);
}

// The model's int8 version: whether it is generated, which page generation has reached, and how to install onnx
// when it is missing. The row is labeled "int8 模型", so the text does not repeat it
/**
 * @param {string} kind
 * @param {ModelStatus} m
 */
function renderInt8(kind, m) {
  const q = m.int8;
  const status = $(`${kind}-int8-status`);
  const btn = byId(`${kind}-quantize`, HTMLButtonElement);
  const ready = q.file.exists && !q.state.running && !q.state.error;
  closest(status, "dd").classList.toggle("is-ok", ready);
  // int8 is optional: only a failed generation is shown in red; a missing onnx is shown normally
  status.classList.toggle("warn", Boolean(q.state.error));
  btn.textContent = q.file.exists ? "重新生成" : "生成 int8 模型";
  btn.disabled = q.state.running || !q.available || !m.file.exists;
  if (q.state.running) {
    status.textContent = q.state.done < q.state.total
      ? `生成中：校准 ${q.state.done} / ${q.state.total} 页`
      : "生成中：量化并核对结果…";
  } else if (q.state.error) {
    status.textContent = `生成失败：${q.state.error}`;
  } else if (!q.file.exists && !m.file.exists) {
    status.textContent = "下载模型后才能生成。";
  } else if (q.file.exists) {
    const size = formatMB(q.file.size);
    if (m.in_use !== q.path) status.textContent = `已生成（${size}）。当前精度为 fp32，翻译时不使用`;
    else status.textContent = m.enabled ? `已生成（${size}），翻译时使用` : `已生成（${size}），开启模型后在翻译时使用`;
  } else if (!q.available) {
    status.textContent = `需要先安装 onnx：请在项目目录运行 ${q.install_hint}，安装后重启服务。`;
  } else {
    status.textContent = "未生成，翻译时使用 fp32。生成约需 3 分钟，从翻译记录中的 PDF 抽取 64 页校准。";
  }
}

// The select lists only values; the description of the selected one is shown below it (from the option's
// data-hint)
/** @param {HTMLSelectElement} select */
function showOptionHint(select) {
  $(`${select.id}-hint`).textContent = select.selectedOptions[0]?.dataset.hint || "";
}
for (const select of queryAll(document, "select.has-hint", HTMLSelectElement)) {
  select.addEventListener("change", () => showOptionHint(select));
}

/**
 * @param {string} kind
 * @param {RenderModelOptions} [options]
 */
async function loadModel(kind, options = {}) {
  try {
    renderModel(kind, /** @type {ModelStatus} */ (await api(`/api/${kind}-model`)), options);
  } catch (err) {
    toast(`读取${MODEL_NAMES[kind]}设置失败：${messageOf(err)}`);
  }
}

// Form fields to save. The formula detection model adds the input size, the pages inferred at once and the threads
// per page. The last two show values computed from the core count; unchanged or cleared they are saved as 0, so
// another machine still decides by its core count
/** @param {string} kind */
function modelForm(kind) {
  /**
   * @type {{
   *   enabled: boolean, path: string, size?: number, precision?: string, workers?: number, threads?: number
   * }}
   */
  const form = {
    enabled: byId(`${kind}-enabled`, HTMLInputElement).checked, path: byId(`${kind}-path`, HTMLInputElement).value,
  };
  // The fields only one of the models has
  const size = document.getElementById(`${kind}-size`);
  if (size instanceof HTMLSelectElement) form.size = Number(size.value);
  const precision = document.getElementById(`${kind}-precision`);
  if (precision instanceof HTMLSelectElement) form.precision = precision.value;
  for (const name of /** @type {const} */ (["workers", "threads"])) {
    const input = document.getElementById(`${kind}-${name}`);
    if (input instanceof HTMLInputElement) {
      form[name] = input.value === input.dataset.auto ? 0 : Number(input.value || 0);
    }
  }
  return form;
}

function loadLayoutModel() {
  for (const kind of MODEL_KINDS) loadModel(kind);
}

for (const kind of MODEL_KINDS) {
  $(`${kind}-quantize`).addEventListener("click", async () => {
    try {
      const m = /** @type {ModelStatus} */ (await api(`/api/${kind}-model/quantize`, { method: "POST" }));
      renderModel(kind, m, { keepForm: true });
    } catch (err) {
      toast(messageOf(err));
    }
  });
}

// Each model has its own tab, and the last one viewed is remembered
/** @param {string | null | undefined} kind one of MODEL_KINDS; anything else shows the first */
function showModelTab(kind) {
  if (!kind || !MODEL_KINDS.includes(kind)) kind = MODEL_KINDS[0];
  for (const k of MODEL_KINDS) {
    $(`${k}-tab`).setAttribute("aria-selected", String(k === kind));
    $(`${k}-form`).hidden = k !== kind;
  }
  try {
    localStorage.setItem("doc-translator-model-tab", kind);
  } catch {
    /* ignore */
  }
}
for (const tab of queryAll(document, ".model-tab", HTMLElement)) {
  tab.addEventListener("click", () => showModelTab(tab.dataset.model));
}
try {
  showModelTab(localStorage.getItem("doc-translator-model-tab"));
} catch {
  showModelTab(MODEL_KINDS[0]);
}

for (const kind of MODEL_KINDS) {
  /** @param {string} name */
  const el = (name) => $(`${kind}-${name}`);

  $(`${kind}-form`).addEventListener("submit", async (e) => {
    e.preventDefault();
    const status = el("save-status");
    await busy(byId(`${kind}-save`, HTMLButtonElement), "保存中…", async () => {
      try {
        const json = modelForm(kind);
        const m = /** @type {ModelStatus} */ (await api(`/api/${kind}-model`, { method: "PUT", json }));
        renderModel(kind, m);
        status.textContent = m.enabled && !(m.runtime && m.file.exists)
          ? "已保存。运行环境或模型文件缺失时，翻译仍只使用规则"
          : "已保存";
      } catch (err) {
        status.textContent = `保存失败：${messageOf(err)}`;
      }
    });
  });

  el("download").addEventListener("click", async () => {
    try {
      renderModel(
        kind,
        /** @type {ModelStatus} */ (await api(`/api/${kind}-model/download`, {
          json: {
            source: byId(`${kind}-source`, HTMLSelectElement).value, path: byId(`${kind}-path`, HTMLInputElement).value,
          },
        })),
        { keepForm: true },
      );
    } catch (err) {
      toast(messageOf(err));
    }
  });

  el("test").addEventListener("click", async () => {
    const box = el("test-result");
    box.hidden = false;
    box.className = "test-result";
    box.textContent = "测试中…";
    await busy(byId(`${kind}-test`, HTMLButtonElement), "测试中…", async () => {
      try {
        const r = /** @type {ModelTest} */ (await api(`/api/${kind}-model/test`, { json: modelForm(kind) }));
        box.classList.add("ok");
        const found = r.labels.length ? `识别出 ${r.labels.join("、")}` : "未识别出任何区域";
        box.textContent = `测试成功：加载 ${r.load_ms} ms，单页推理 ${r.page_ms} ms，样例页${found}`;
      } catch (err) {
        box.classList.add("fail");
        box.textContent = `测试失败：${messageOf(err)}`;
      }
    });
  });
}

init().catch((err) => alert(`初始化失败：${messageOf(err)}`));

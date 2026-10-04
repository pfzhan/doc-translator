const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "doc-translator-settings";

const state = {
  providers: {}, services: [], defaultId: "google", selectedId: null,
  jobs: [], diskUsage: 0, retentionDays: 30, readerView: "both", view: "translate",
};

async function api(url, options = {}) {
  const init = { ...options };
  if (init.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `请求失败 (${res.status})`);
  return data;
}

// ---------- 视图切换 ----------
// 页面：translate（翻译）/ history（翻译记录）/ services（翻译服务）；阅读器是覆盖在上面的全屏层

const VIEWS = ["translate", "history", "services"];

function showView(name) {
  if (!VIEWS.includes(name)) name = "translate";
  for (const tab of document.querySelectorAll(".tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === name));
  }
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== name;
  state.view = name;
  if (name === "history") refreshJobs();
  try {
    localStorage.setItem("doc-translator-view", name);
  } catch {
    /* 忽略 */
  }
}
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    if (reader.jobId) closeReader();
    showView(tab.dataset.view);
  });
}
document.addEventListener("click", (e) => {
  const goto = e.target.closest("[data-goto]");
  if (goto) showView(goto.dataset.goto);
});

let toastTimer;
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 2600);
}

// 异步按钮的即时反馈：点击后立即禁用并显示忙碌文案，结束（无论成败）后恢复
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


// ---------- 服务数据 ----------

async function loadServices() {
  const data = await api("/api/services");
  state.services = data.services;
  state.defaultId = data.default;
  state.ccswitch = data.ccswitch;
  if (!state.services.some((s) => s.id === state.selectedId)) state.selectedId = state.defaultId;
  renderServiceList();
  renderServiceSelect();
  renderDetail();
}

const ICONS = { google: "G", openai: "⌬", claude: "✳", gemini: "✦", grok: "𝕏", custom: "⚙" };

function iconFor(provider) {
  const span = document.createElement("span");
  span.className = `icon icon-${provider}`;
  span.textContent = ICONS[provider] || "?";
  span.setAttribute("aria-hidden", "true");
  return span;
}

function renderServiceSelect() {
  const select = $("service_id");
  // 打开页面时选中默认服务；之后用户手动改过的保持不变（刷新服务列表时不会被重置）
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
  refresh.title = "在 CC Switch 里切换供应商后，点这里重新读取";
  refresh.addEventListener("click", () =>
    busy(refresh, "刷新中…", async () => {
      try {
        await loadServices();
        toast("服务列表已刷新");
      } catch (err) {
        toast(`刷新失败：${err.message}`);
      }
    })
  );
  const sections = [["本地配置", local, null]];
  if (state.ccswitch?.found || cc.length) sections.push(["CC Switch 当前生效", cc, refresh]);
  for (const [title, items, extra] of sections) {
    list.append(groupHeader(title, extra));
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "hint service-empty";
      empty.textContent = "没有读到配置";
      list.append(empty);
    }
    for (const s of items) list.append(serviceItem(s));
  }
}

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
    setDefault.className = "link small";
    setDefault.textContent = "设为默认";
    setDefault.addEventListener("click", async () => {
      try {
        await api(`/api/services/${s.id}/default`, { method: "POST" });
        await loadServices();
      } catch (err) {
        alert(err.message);
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
  toggle.setAttribute("aria-label", `启用 ${s.name}`);
  toggle.addEventListener("change", async () => {
    try {
      await api(`/api/services/${s.id}`, { method: "PUT", json: { enabled: toggle.checked } });
      await loadServices();
    } catch (err) {
      toggle.checked = !toggle.checked;
      alert(err.message);
    }
  });
  li.append(toggle);
  return li;
}

// ---------- 服务详情 ----------

// CC Switch 服务复制到本地：连接信息固化，之后不再跟随 CC Switch
async function cloneService(s, btn) {
  const run = async () => {
    try {
      const svc = await api(`/api/services/${s.id}/clone`, { method: "POST" });
      state.selectedId = svc.id;
      await loadServices();
      toast("已复制到本地配置，连接信息已固化，不再跟随 CC Switch");
    } catch (err) {
      toast(`复制失败：${err.message}`);
    }
  };
  if (btn) await busy(btn, "复制中…", run);
  else await run();
}

const svcForm = $("svc-form");

function selected() {
  return state.services.find((s) => s.id === state.selectedId);
}

function setModelOptions(models, current) {
  const select = $("model-select");
  select.replaceChildren();
  const all = [...new Set([...(current ? [current] : []), ...models])];
  for (const m of all) select.add(new Option(m, m));
  select.value = current || all[0] || "";
}

function useCustomModel(on) {
  $("custom-model").checked = on;
  $("model-select").hidden = on;
  $("model-input").hidden = !on;
}

function renderDetail() {
  const s = selected();
  if (!s) return;
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
  // CC Switch 服务可以复制成本地配置（连接信息固化，不再跟随 CC Switch）；不可用的没有 Key 可复制
  $("clone-btn").hidden = !fromCC;
  $("clone-btn").disabled = fromCC && !s.available;
  // CC Switch 的地址、Key、名称以 CC Switch 为准，这里只读
  for (const id of ["api_key", "base_url", "svc-name-input"]) $(id).readOnly = fromCC;
  $("show-key").closest("label").hidden = fromCC;
  $("cc-notice").hidden = !fromCC;
  if (fromCC) {
    $("cc-notice").textContent = s.available
      ? `来自 CC Switch 当前生效的「${s.name.split(" · ").pop()}」。地址和 API Key 每次翻译时现读，在 CC Switch 里切换供应商后点列表里的「刷新」即可。模型和高级设置可以在这里单独调整。`
      : `CC Switch 当前生效的「${s.name.split(" · ").pop()}」不能用于翻译：${s.unavailable_reason}。在 CC Switch 里切换到带 API Key 的供应商后点「刷新」。`;
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
      a.textContent = "去申请 Key";
      notice.append("，", a);
    }
  }
  notice.hidden = !notice.childNodes.length;

  const f = svcForm.elements;
  f.api_key.value = "";
  f.api_key.placeholder = fromCC
    ? `来自 CC Switch（${s.api_key_hint}）`
    : s.api_key_hint ? `已保存（${s.api_key_hint}），留空则不修改` : meta.needs_key ? "必填" : "可选";
  f.api_key.type = "password";
  $("show-key").checked = false;
  f.name.value = s.name;
  f.base_url.value = s.base_url;
  f.base_url.placeholder = meta.base_url || "例如 https://api.deepseek.com/v1";
  f.base_url.required = !meta.base_url;
  f.concurrency.value = s.concurrency;
  f.max_items.value = s.max_items;
  f.max_chars.value = s.max_chars;
  f.temperature.value = s.temperature ?? "";
  f.prompt.value = s.prompt;
  f.user_prompt.value = s.user_prompt || "";
  showDefaultPromptPlaceholders();

  // 获取过模型列表就用获取到的，否则用预置模型
  const fetched = fetchedModels[s.id];
  const models = fetched || meta.models || [];
  setModelOptions(models, s.model);
  f.model.value = s.model;
  $("fetch-models").textContent = fetched ? "重新获取" : "获取模型列表";
  $("fetch-models").disabled = false;
  setFetchStatus(fetched ? `已获取 ${fetched.length} 个模型` : "");
  // 没有可选模型（如 OpenAI 兼容接口）、或者当前模型不在列表里（中转站自定义的模型名）时，默认用输入框
  useCustomModel(!models.length || (fromCC && !models.includes(s.model)));
  if (fromCC) {
    $("model-input").placeholder = `留空使用 CC Switch 里的模型（${s.ccswitch_model}）`;
  }
}

// 默认提示词随目标语言变化，留空的输入框里用 placeholder 展示
const defaultPromptCache = {};

async function getDefaultPrompts() {
  const to = $("target_lang").value || "zh-CN";
  const from = $("source_lang").value || "auto";
  const key = `${from}2${to}`;
  if (!defaultPromptCache[key]) {
    const qs = new URLSearchParams({ target_lang: to, source_lang: from });
    defaultPromptCache[key] = await api(`/api/prompts/default?${qs}`);
  }
  return defaultPromptCache[key];
}

async function showDefaultPromptPlaceholders() {
  try {
    const d = await getDefaultPrompts();
    svcForm.elements.prompt.placeholder = d.system;
    svcForm.elements.user_prompt.placeholder = d.user;
  } catch {
    /* 拿不到默认值不影响使用 */
  }
}

$("load-default-prompt").addEventListener("click", async () => {
  const f = svcForm.elements;
  if ((f.prompt.value || f.user_prompt.value) && !confirm("用默认提示词覆盖当前内容吗？")) return;
  const d = await getDefaultPrompts();
  f.prompt.value = d.system;
  f.user_prompt.value = d.user;
});

for (const id of ["target_lang", "source_lang"]) {
  $(id).addEventListener("change", showDefaultPromptPlaceholders);
}

$("show-key").addEventListener("change", (e) => {
  svcForm.elements.api_key.type = e.target.checked ? "text" : "password";
});
$("custom-model").addEventListener("change", (e) => {
  if (e.target.checked) $("model-input").value = $("model-select").value;
  else setModelOptions([...$("model-select").options].map((o) => o.value), $("model-input").value);
  useCustomModel(e.target.checked);
});

function formPayload() {
  const f = svcForm.elements;
  const s = selected();
  const model = $("custom-model").checked ? $("model-input").value : $("model-select").value;
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
    prompt: f.prompt.value,
    user_prompt: f.user_prompt.value,
    target_lang: $("target_lang").value || "zh-CN",
    source_lang: $("source_lang").value || "auto",
  };
}

svcForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formPayload();
  if (!payload.model && selected()?.source !== "ccswitch") {
    $("save-status").textContent = "请先选择或输入模型";
    return;
  }
  await busy($("save-btn"), "保存中…", async () => {
    try {
      await api(`/api/services/${payload.id}`, { method: "PUT", json: payload });
      await loadServices();
      $("save-status").textContent = "已保存";
    } catch (err) {
      $("save-status").textContent = `保存失败：${err.message}`;
    }
  });
});

$("delete-btn").addEventListener("click", async (e) => {
  const s = selected();
  if (!s || !confirm(`确定删除「${s.name}」吗？`)) return;
  await busy(e.currentTarget, "删除中…", async () => {
    try {
      await api(`/api/services/${s.id}`, { method: "DELETE" });
      state.selectedId = state.defaultId;
      await loadServices();
    } catch (err) {
      alert(err.message);
    }
  });
});

$("clone-btn").addEventListener("click", (e) => {
  const s = selected();
  if (s) cloneService(s, e.currentTarget);
});

$("test-btn").addEventListener("click", async (e) => {
  const box = $("test-result");
  const s = selected();
  box.hidden = false;
  box.className = "test-result";
  box.textContent = "测试中…";
  await busy(e.currentTarget, "测试中…", async () => {
    try {
      const payload = state.providers[s.provider]?.llm ? formPayload() : { id: s.id, provider: s.provider };
      const r = await api("/api/services/test", { json: payload });
      box.classList.add("ok");
      box.textContent = `测试成功（${r.ms} ms）：${r.source} → ${r.result}`;
    } catch (err) {
      box.classList.add("fail");
      box.textContent = `测试失败：${err.message}`;
    }
  });
});

// 已获取的模型列表按服务缓存，切换服务再切回来时不用重新获取
const fetchedModels = {};

function setFetchStatus(text, isError = false) {
  const el = $("fetch-models-status");
  el.textContent = text;
  el.classList.toggle("warn", isError);
}

$("fetch-models").addEventListener("click", async () => {
  const btn = $("fetch-models");
  const sid = state.selectedId;
  btn.disabled = true;
  setFetchStatus("获取中…");
  try {
    const payload = formPayload();
    payload.model = payload.model || "placeholder"; // 获取列表时还没选模型
    const { models } = await api("/api/services/models", { json: payload });
    if (!models.length) throw new Error("接口没有返回模型");
    fetchedModels[sid] = models;
    if (state.selectedId !== sid) return; // 获取期间切到了别的服务
    const current = $("custom-model").checked ? $("model-input").value : $("model-select").value;
    // 只用接口返回的列表，不再混入预置模型；当前模型不在列表里（服务商根本不支持）时改选第一个，不再保留失效选择
    const fallback = models.includes(current) ? current : models[0];
    setModelOptions(models, fallback);
    useCustomModel(false);
    const note = current && current !== fallback ? `，原模型 ${current} 不在列表中已切换` : "";
    setFetchStatus(`已获取 ${models.length} 个模型${note} · ${new Date().toLocaleTimeString()}`);
  } catch (err) {
    if (state.selectedId === sid) setFetchStatus(`获取失败：${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = fetchedModels[sid] ? "重新获取" : "获取模型列表";
  }
});

$("add-btn").addEventListener("click", async (e) => {
  await busy(e.currentTarget, "添加中…", async () => {
    try {
      const svc = await api("/api/services", { json: { provider: $("add-provider").value } });
      state.selectedId = svc.id;
      await loadServices();
      svcForm.elements.api_key.focus();
    } catch (err) {
      alert(err.message);
    }
  });
});

// ---------- 翻译页 ----------

const form = $("form");
const fileInput = $("file");
const drop = $("drop");
// 翻译服务不记住：每次打开页面都从默认服务开始
const SAVED_FIELDS = ["source_lang", "target_lang", "mode"];

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
  } catch {
    return {};
  }
}

function saveSettings() {
  const data = {};
  for (const name of SAVED_FIELDS) data[name] = form.elements[name].value;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
  } catch {
    /* 隐私模式下可能不可用，忽略 */
  }
}


// ---------- 语言 ----------

function langInfo(code) {
  return (state.languages || []).find((l) => l.code === code);
}

function langLabel(code) {
  const l = langInfo(code);
  if (!l) return code;
  if (code === "auto") return "自动检测";
  // 本地名 + 英文名，如 “日本語 Japanese”；两者相同时只显示一个
  return l.native && l.native !== l.name ? `${l.native} ${l.name}` : l.name;
}

// 和插件一样，默认目标语言跟随浏览器语言（navigator.language）
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
  const id = $("service_id").value;
  return state.services.find((s) => s.id === id);
}

function checkLangPair() {
  const from = $("source_lang").value;
  const to = $("target_lang").value;
  const warn = $("lang-warning");
  const msgs = [];
  if (from !== "auto" && from === to) msgs.push("源语言和目标语言相同，没有需要翻译的内容。");
  const svc = selectedService();
  if (svc && svc.provider === "google") {
    const bad = [from, to].filter((c) => c !== "auto" && !langInfo(c)?.google);
    if (bad.length) msgs.push(`谷歌翻译不支持${bad.map(langLabel).join("、")}，请换用 AI 翻译服务。`);
  }
  warn.textContent = msgs.join(" ");
  warn.hidden = !msgs.length;
  $("submit").disabled = msgs.length > 0;
}

for (const id of ["source_lang", "target_lang", "service_id"]) {
  $(id).addEventListener("change", checkLangPair);
}
$("service_id").addEventListener("change", () => (state.serviceTouched = true));

$("swap-lang").addEventListener("click", () => {
  const src = $("source_lang");
  const dst = $("target_lang");
  // 源语言是“自动检测”时没法交换到目标语言，用上一次检测到的语言代替
  // 用最近一条记录里检测到的语言
  const lastDetected = (state.jobs || []).find((j) => j.detected_lang)?.detected_lang;
  const from = src.value === "auto" ? lastDetected : src.value;
  if (!from) {
    alert("源语言是自动检测，先翻译一次或手动选择源语言后再交换");
    return;
  }
  src.value = dst.value;
  dst.value = from;
  checkLangPair();
  showDefaultPromptPlaceholders();
});

function formatSize(bytes) {
  if (!bytes) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function showFile() {
  const f = fileInput.files[0];
  drop.classList.toggle("has-file", !!f);
  $("drop-text").textContent = f ? f.name : "选择文件，或拖到这里";
  $("drop-sub").textContent = f ? `${formatSize(f.size)} · 点击更换` : "最大 200 MB";
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
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    showFile();
  }
});

function showSubmitError(text) {
  const el = $("submit-error");
  el.textContent = text || "";
  el.hidden = !text;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;
  saveSettings();
  showSubmitError("");
  const btn = $("submit");
  btn.disabled = true;
  btn.textContent = "上传中…";
  try {
    const job = await api("/api/jobs", { method: "POST", body: new FormData(form) });
    fileInput.value = "";
    showFile();
    upsertJob(job);
    // 留在当前页：新任务出现在下方「最近翻译」里，想看的时候点「查看进度」进入阅读器
    toast("已开始翻译");
  } catch (err) {
    showSubmitError(err.message);
  } finally {
    btn.textContent = "开始翻译";
    btn.disabled = false;
    checkLangPair();
  }
});

// ---------- 翻译记录 ----------
// 记录存在服务端磁盘上：刷新页面、换书、重启服务都不会丢，只有手动删除或超过保留时间才会删

const STATUS_TEXT = { queued: "排队中", running: "翻译中", done: "已完成", error: "失败", interrupted: "已中断", paused: "已暂停" };
// 可以“继续翻译”的状态：已翻好的段落在缓存里，继续时直接复用
const RESUMABLE = new Set(["paused", "interrupted"]);
const ACTIVE = new Set(["queued", "running"]);

// 电子书富文本段落的原文/译文带行内标签（<b>、<a> 等），预览显示时剥掉：
// 纯字符串处理（不走 DOM，译文来自模型，避免 onerror 之类的属性被触发）
function htmlToText(s) {
  if (!/<[a-zA-Z/]/.test(s)) return s;
  return s
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]*>/g, "")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(n))
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

function isEbookJob(job) {
  return /\.(epub|mobi|azw3?|azw)$/i.test(job?.filename || "");
}

function fmtOf(name) {
  return (name.split(".").pop() || "").toLowerCase().replace("markdown", "md");
}

// 和 app/formats/__init__.py 的 output_variant 同一条规则：标记紧挨扩展名
function outputVariant(name) {
  const stem = name.slice(0, Math.max(0, name.lastIndexOf(".")));
  const token = stem.slice(stem.lastIndexOf(".") + 1);
  return token === "bilingual" || token === "translated" ? token : "";
}

function downloadLabel(name, multiple) {
  const variant = outputVariant(name) === "bilingual" ? "双语" : outputVariant(name) === "translated" ? "译文" : "";
  return `下载${variant}${multiple ? ` ${fmtOf(name).toUpperCase()}` : ""}`;
}

// 批量算按钮文字：只有同一版本存在多个格式（如 MOBI 任务的 EPUB+MOBI）才带格式名
function downloadLabels(outputs) {
  const variants = outputs.map((o) => outputVariant(o.name));
  const dup = variants.some((v, i) => v && variants.indexOf(v) !== i);
  return outputs.map((o) => downloadLabel(o.name, dup));
}

// 另一种版本的按钮：文案和已有文件的下载按钮完全一致（“下载译文/下载双语”），
// 文件还没生成时点击就在后台现生成（服务端按缓存重新排版，几秒）然后直接开始下载
const variantInflight = new Set();

function variantButton(job) {
  if (job.status !== "done") return null;
  const want = job.mode === "bilingual" ? "译文" : "双语";
  const marker = job.mode === "bilingual" ? "translated" : "bilingual";
  if (job.outputs.some((o) => outputVariant(o.name) === marker)) return null;
  const btn = el("button", "btn sm", `下载${want}`);
  btn.type = "button";
  if (variantInflight.has(job.id)) {
    btn.disabled = true;
    btn.textContent = "生成中…";
  }
  btn.addEventListener("click", async () => {
    if (variantInflight.has(job.id)) return;
    variantInflight.add(job.id);
    btn.disabled = true;
    btn.textContent = "生成中…";
    try {
      const updated = await api(`/api/jobs/${job.id}/variant`, { method: "POST" });
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
      toast(`生成失败：${err.message}`);
    } finally {
      variantInflight.delete(job.id);
      const current = state.jobs.find((j) => j.id === job.id);
      if (current) upsertJob(current);
      if (reader.jobId === job.id) renderReaderBar(reader.job || current);
    }
  });
  return btn;
}

function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "刚刚";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} 天前`;
  return new Date(ts * 1000).toLocaleDateString();
}

// token 用量：大模型接口才有，谷歌等返回 0 不显示
function tokensText(job) {
  const total = (job.prompt_tokens || 0) + (job.completion_tokens || 0);
  if (!total) return "";
  const text = total >= 1e6 ? `${(total / 1e6).toFixed(2)}M` : total >= 1e3 ? `${(total / 1e3).toFixed(1)}k` : `${total}`;
  return `${text} tokens`;
}

function tokensTitle(job) {
  return `输入 ${(job.prompt_tokens || 0).toLocaleString()} · 输出 ${(job.completion_tokens || 0).toLocaleString()}`;
}

function expiryText(job) {
  const days = state.retentionDays;
  if (!days || ACTIVE.has(job.status)) return "";
  const left = (job.finished || job.created) + days * 86400 - Date.now() / 1000;
  if (left <= 0) return "即将删除";
  const d = Math.ceil(left / 86400);
  return d <= 3 ? `${d} 天后自动删除` : "";
}

function langPair(job) {
  const from = job.source_lang === "auto" ? job.detected_lang || "auto" : job.source_lang;
  return `${langShort(from)} → ${langShort(job.target_lang)}`;
}

function langShort(code) {
  const l = langInfo(code);
  if (!l || code === "auto") return code === "auto" ? "自动" : code;
  return l.native || l.name;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function jobRow(job) {
  const li = el("li", "job");
  li.dataset.id = job.id;
  const fmt = fmtOf(job.filename);
  li.append(el("div", `job-fmt ${fmt}`, fmt.toUpperCase()));

  const main = el("div", "job-main");
  const name = el("button", "job-name", job.filename);
  name.type = "button";
  name.title = job.preview ? "打开阅读" : job.filename;
  name.disabled = !job.preview;
  name.addEventListener("click", () => openReader(job.id));
  main.append(name);

  const meta = el("div", "job-meta");
  // 两行固定布局：第一行语言/模式/服务，第二行大小/时间。
  // 单行 flex-wrap 的换行点会随按钮宽度漂移，不同卡片换行位置不一致，还会把“·”甩到行首
  for (const [i, part] of [
    langPair(job),
    job.mode === "translated" ? "仅译文" : "双语对照",
    serviceLabel(job),
  ].filter(Boolean).entries()) {
    meta.append(el("span", i ? "sep" : "", part));
  }
  const meta2 = el("div", "job-meta");
  for (const [i, part] of [formatSize(job.size), timeAgo(job.created)].filter(Boolean).entries()) {
    meta2.append(el("span", i ? "sep" : "", part));
  }
  main.append(meta, meta2);

  const status = el("div", "job-status");
  status.append(el("span", `badge ${job.status}`, STATUS_TEXT[job.status] || job.status));
  if (ACTIVE.has(job.status) || (job.status === "paused" && job.total)) {
    // 向下取整：没翻完不显示 100%（1184/1188 四舍五入会显示 100%，看起来像卡住了）
    const pct = job.total ? Math.floor((job.done / job.total) * 100) : 0;
    const bar = el("div", `job-progress${job.status === "paused" ? " paused" : ""}`);
    const fill = el("div");
    fill.style.width = `${pct}%`;
    bar.append(fill);
    status.append(bar, el("span", "hint", job.total ? `${job.done} / ${job.total} 段 · ${pct}%` : "解析文档…"));
  } else if (job.status === "error" || job.status === "interrupted") {
    const err = el("span", "job-error", explainError(job.error));
    err.title = job.error; // 鼠标悬停看原始错误
    status.append(err);
  } else {
    const parts = [job.total ? `${job.total} 段` : ""];
    if (job.skipped) parts.push(`${job.skipped} 段无需翻译`);
    const tokens = tokensText(job);
    if (tokens) parts.push(tokens);
    const exp = expiryText(job);
    if (exp) parts.push(exp);
    const hint = el("span", "hint", parts.filter(Boolean).join(" · "));
    if (tokens) hint.title = tokensTitle(job);  // 悬停看输入/输出拆分
    status.append(hint);
  }
  main.append(status);
  li.append(main);

  const actions = el("div", "job-actions");
  if (job.preview && (job.status !== "error" || job.done)) {
    const read = el("button", "btn sm", ACTIVE.has(job.status) ? "查看进度" : "阅读");
    read.type = "button";
    read.addEventListener("click", () => openReader(job.id));
    actions.append(read);
  }
  const curVariant = job.mode === "bilingual" ? "bilingual" : "translated";
  for (const [i, o] of job.outputs.entries()) {
    // 当前模式对应的版本是主按钮；另一种版本已生成时降为普通按钮，和“生成X版”占位一致
    const primary = outputVariant(o.name) === curVariant;
    const a = el("a", primary ? "btn sm primary" : "btn sm", downloadLabels(job.outputs)[i]);
    a.href = o.url;
    a.download = o.name;
    a.title = o.name;
    actions.append(a);
  }
  const variant = variantButton(job);
  if (variant) actions.append(variant);
  if (ACTIVE.has(job.status)) {
    const pause = el("button", "btn sm", "暂停");
    pause.type = "button";
    pause.addEventListener("click", () => pauseJob(job.id, pause));
    actions.append(pause);
  } else if (RESUMABLE.has(job.status) || job.status === "error") {
    const retry = el("button", "btn sm", job.status === "error" ? "重试" : "继续翻译");
    retry.type = "button";
    retry.addEventListener("click", () => retryJob(job.id));
    actions.append(retry);
  }
  const del = el("button", "btn sm ghost", "删除");
  del.type = "button";
  del.addEventListener("click", () => deleteJob(job));
  actions.append(del);
  li.append(actions);
  return li;
}

// 记录里显示“服务 · 模型”；模型名已经包含在服务名里时（CC Switch 的服务名不带模型）不重复
function serviceLabel(job) {
  if (!job.service_name) return "";
  return job.model && !job.service_name.includes(job.model) ? `${job.service_name} · ${job.model}` : job.service_name;
}

function renderJobs() {
  const all = state.jobs;
  const list = $("history-list");
  list.replaceChildren(...all.map(jobRow));
  $("history-empty").hidden = all.length > 0;
  $("recent").hidden = all.length === 0;
  $("recent-list").replaceChildren(...all.slice(0, 3).map(jobRow));
  const running = all.filter((j) => ACTIVE.has(j.status)).length;
  $("running-badge").hidden = !running;
  $("running-badge").textContent = running;
  $("usage").textContent = `${all.length} 条记录 · 占用 ${formatSize(state.diskUsage) || "0 KB"}`;
  renderBulk(all);
}

// 全部暂停 / 全部继续：只在有需要的时候出现
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
  const parts = [];
  if (active) parts.push(`${active} 本正在翻译`);
  if (resumable) parts.push(`${resumable} 本已暂停或中断`);
  $("history-bulk").hidden = !parts.length;
  $("history-bulk-text").textContent = parts.join("，");
}

async function pauseJob(id, btn) {
  if (btn) btn.disabled = true;
  try {
    const job = await api(`/api/jobs/${id}/pause`, { method: "POST" });
    upsertJob(job);
    if (reader.jobId === id) refreshReader();
    toast("已暂停，翻好的段落都保留着");
  } catch (err) {
    if (btn) btn.disabled = false;
    toast(`暂停失败：${err.message}`);
  }
}

async function pauseAll(btn) {
  btn.disabled = true;
  try {
    const r = await api("/api/jobs/pause-all", { method: "POST" });
    await refreshJobs();
    if (reader.jobId) refreshReader();
    toast(`已暂停 ${r.paused} 本`);
  } catch (err) {
    btn.disabled = false;
    toast(`暂停失败：${err.message}`);
  }
}

async function resumeAll(btn) {
  btn.disabled = true;
  try {
    const r = await api("/api/jobs/resume-all", { method: "POST" });
    await refreshJobs();
    if (reader.jobId) refreshReader();
    const failed = r.failed.length ? `，${r.failed.length} 本无法继续：${r.failed[0].error}` : "";
    toast(`已继续 ${r.resumed} 本${failed}`);
  } catch (err) {
    btn.disabled = false;
    toast(`继续失败：${err.message}`);
  }
}

function upsertJob(job) {
  const i = state.jobs.findIndex((j) => j.id === job.id);
  if (i >= 0) state.jobs[i] = job;
  else state.jobs.unshift(job);
  renderJobs();
  scheduleJobsPoll();
}

async function refreshJobs() {
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    state.diskUsage = data.disk_usage;
    state.retentionDays = data.retention_days;
    renderJobs();
  } catch {
    /* 网络问题时保留旧列表 */
  }
  scheduleJobsPoll();
}

// 有任务在跑时每 2 秒刷新列表（阅读器自己轮询，不重复）
let jobsTimer;
function scheduleJobsPoll() {
  clearTimeout(jobsTimer);
  if (reader.jobId) return;
  if (state.jobs.some((j) => ACTIVE.has(j.status))) jobsTimer = setTimeout(refreshJobs, 2000);
}

async function deleteJob(job) {
  const running = ACTIVE.has(job.status);
  const msg = running
    ? `「${job.filename}」还在翻译，删除会停止翻译并删除所有文件。确定吗？`
    : `删除「${job.filename}」的翻译记录？原文件和译文都会被删除，无法恢复。`;
  if (!confirm(msg)) return;
  try {
    await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    state.jobs = state.jobs.filter((j) => j.id !== job.id);
    renderJobs();
    if (reader.jobId === job.id) closeReader();
    toast("已删除");
  } catch (err) {
    toast(`删除失败：${err.message}`);
  }
}

async function retryJob(id) {
  try {
    const job = await api(`/api/jobs/${id}/retry`, { method: "POST", json: {} });
    upsertJob(job);
    toast("已重新开始翻译");
  } catch (err) {
    toast(`无法重新翻译：${err.message}`);
  }
}

// 保留时间
function renderRetention() {
  const sel = $("retention-days");
  const days = String(state.retentionDays ?? 30);
  const preset = [...sel.options].some((o) => o.value === days);
  sel.value = preset ? days : "custom";
  $("retention-custom").hidden = preset;
  $("retention-custom").value = preset ? "" : days;
  $("retention-save").hidden = preset;
}

// 参照 toast：先取消上一次定时器，避免误清新的提示
let retentionStatusTimer;
async function saveRetention(days) {
  const status = $("retention-status");
  try {
    const r = await api("/api/settings", { method: "PUT", json: { retention_days: days } });
    state.retentionDays = r.retention_days;
    renderRetention();
    status.textContent = r.retention_days ? "已保存" : "已保存，记录将永久保留";
    refreshJobs();
  } catch (err) {
    status.textContent = err.message;
    renderRetention();
  }
  clearTimeout(retentionStatusTimer);
  retentionStatusTimer = setTimeout(() => (status.textContent = ""), 2500);
}

$("retention-days").addEventListener("change", (e) => {
  if (e.target.value === "custom") {
    $("retention-custom").hidden = false;
    $("retention-save").hidden = false;
    $("retention-custom").focus();
    return;
  }
  saveRetention(Number(e.target.value));
});
$("retention-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const raw = $("retention-custom").value.trim();
  const days = Number(raw);
  // 0 表示永久保留，是合法值；只拒绝空、非整数和负数
  if (raw === "" || !Number.isInteger(days) || days < 0) {
    toast("保留天数需为不小于 0 的整数（0 表示永久保留）");
    return;
  }
  saveRetention(days);
});

// ---------- 阅读器（边翻边预览） ----------
// 后端按完成顺序记录译文，前端每秒用 version 增量拉取，只更新变化的段落，不重建文件

const reader = {
  jobId: null, job: null, version: -1, nodes: [], headings: [], done: 0, total: 0,
  timer: null, focusTimer: null, lastFocus: -1, following: true, autoScrolling: false,
};
const HEADING = /^h[1-6]$/;

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

// 暂停、继续之后立刻刷新阅读器（不等下一次轮询），避免出现两条轮询
function refreshReader() {
  clearTimeout(reader.timer);
  readerTick();
}

async function readerTick() {
  const id = reader.jobId;
  if (!id) return;
  try {
    const [job, data] = await Promise.all([
      api(`/api/jobs/${id}`),
      api(`/api/jobs/${id}/preview?since=${reader.version}`),
    ]);
    if (reader.jobId !== id) return;
    reader.job = job;
    upsertJob(job);
    renderReaderBar(job);
    if (data.ready) {
      const initial = reader.version < 0;
      if (initial) buildReader(data.segments);
      applyUpdates(data.updates, initial);
      reader.version = data.version;
    } else if (reader.version < 0 && !ACTIVE.has(job.status)) {
      $("reader-content").replaceChildren(el("div", "hint", "这条记录没有可预览的内容。"));
    }
    renderNotice(job);
    // 已经不在翻译、预览也没就绪时不再轮询。加密或扫描版 PDF 会一直 ready=false，
    // 继续翻译后由 refreshReader / openReader 重新拉。
    if (ACTIVE.has(job.status)) reader.timer = setTimeout(readerTick, 1000);
  } catch (err) {
    if (reader.jobId !== id) return;
    $("reader-notice").hidden = false;
    $("reader-notice").textContent = `无法读取：${err.message}`;
    // 出错后也要继续轮询，退避 4 秒，避免一次网络抖动就永久停更
    reader.timer = setTimeout(readerTick, 4000);
  }
}

function renderReaderBar(job) {
  $("reader-name").textContent = job.filename;
  const meta = [langPair(job), serviceLabel(job)];
  if (ACTIVE.has(job.status)) {
    meta.push(job.total ? `翻译中 ${job.done} / ${job.total} 段` : "正在解析文档…");
  } else {
    meta.push(STATUS_TEXT[job.status]);
  }
  const tokens = tokensText(job);
  if (tokens) meta.push(tokens);
  const metaEl = $("reader-meta");
  metaEl.textContent = meta.filter(Boolean).join(" · ");
  metaEl.title = tokens ? tokensTitle(job) : "";
  const pct = job.status === "done" ? 100 : job.total ? (job.done / job.total) * 100 : 0;
  $("reader-progress-bar").style.width = `${pct}%`;
  $("reader-progress-bar").parentElement.hidden = job.status === "done";

  const actions = $("reader-actions");
  actions.replaceChildren();
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
  const curVariant = job.mode === "bilingual" ? "bilingual" : "translated";
  for (const [i, o] of job.outputs.entries()) {
    const primary = outputVariant(o.name) === curVariant;
    const a = el("a", primary ? "btn primary sm" : "btn sm");
    a.href = o.url;
    a.download = o.name;
    a.append("↓ ", el("span", "", downloadLabels(job.outputs)[i]));
    actions.append(a);
  }
  const variant = variantButton(job);
  if (variant) actions.append(variant);
}

// 把接口返回的原始错误整理成一句人话；原文放在“详细信息”里，排查时还能看到
function explainError(raw) {
  const text = raw || "";
  if (/INVALID_MODEL|Invalid model|model.*(not found|does not exist|不存在)/i.test(text)) {
    return "翻译服务暂时不接受这个模型。中转站偶尔会这样，通常过一会儿重试就好；一直不行的话，换个模型或服务。";
  }
  if (/\b401\b|invalid.*api.?key|Incorrect API key|unauthori[sz]ed/i.test(text)) return "API Key 无效或已过期，请到「翻译服务」里检查。";
  if (/\b403\b|forbidden|permission/i.test(text)) return "这个 API Key 没有权限使用该模型。";
  if (/\b429\b|rate.?limit|quota|余额|insufficient/i.test(text)) return "请求太频繁或额度不足，稍后重试，或在服务里调低并发。";
  if (/不是 JSON|API 地址/.test(text)) return "API 地址可能不对（通常以 /v1 结尾），请到「翻译服务」里检查。";
  if (/服务重启/.test(text)) return "服务重启时翻译被中断了。已经翻好的段落有缓存，继续翻译会很快。";
  if (/请求失败|timed? ?out|ConnectError|超时/i.test(text)) return "连不上翻译服务，请检查网络或服务地址后重试。";
  return "翻译没有完成。";
}

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
      ? `${done}继续翻译时，已翻好的段落直接复用，剩下的段落用翻译服务当前选中的模型。`
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
    pick.setAttribute("aria-label", "用哪个服务重新翻译");
    for (const s of services) pick.add(new Option(s.name, s.id));
    pick.value = services.some((s) => s.id === job.service_id) ? job.service_id : state.defaultId;
    const retry = el("button", "btn primary sm", job.status === "error" ? "重新翻译" : "继续翻译");
    retry.type = "button";
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      try {
        const next = await api(`/api/jobs/${job.id}/retry`, { method: "POST", json: { service_id: pick.value } });
        upsertJob(next);
        openReader(job.id);
      } catch (err) {
        retry.disabled = false;
        toast(`无法重新翻译：${err.message}`);
      }
    });
    actions.append(pick, retry);
    notice.append(body, actions);
  } else if (job.same_lang) {
    notice.append(el("div", "notice-text", `文档语言和目标语言一致（${langLabel(job.target_lang)}），可能没有需要翻译的内容。`));
  }
  notice.hidden = !notice.childNodes.length;
  // 任务停了，还没翻到的段落不再显示“加载中”的闪烁，改成静态的未翻译状态
  $("reader-content").classList.toggle("stopped", !ACTIVE.has(job.status));
}

function buildReader(segments) {
  const frag = document.createDocumentFragment();
  const lang = reader.job?.target_lang || "";
  const ebook = isEbookJob(reader.job);
  let toc = null;
  reader.headings = [];
  reader.nodes = segments.map((seg, i) => {
    const div = el("div", `seg k-${seg.k}`);
    div.dataset.i = i;
    if (!seg.s.trim()) {
      div.hidden = true;
      frag.append(div);
      return div;
    }
    div.classList.add("pending");
    const src = el("div", "src", ebook ? htmlToText(seg.s) : seg.s);
    const dst = el("div", "dst");
    if (lang) dst.lang = lang;
    div.append(src, dst);
    // 书里自带的目录（nav / ncx）收进一个可折叠块，不打断正文
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

// 左侧大纲：用文档里的标题，最多三级；标题太少时不显示
function buildOutline() {
  const list = $("outline-list");
  const levels = [...new Set(reader.headings.map((h) => h.level))].sort().slice(0, 3);
  const items = reader.headings.filter((h) => levels.includes(h.level));
  // 章节标题太少时不显示大纲，正文占满宽度
  $("reader").classList.toggle("no-outline", items.length < 2);
  list.replaceChildren(
    ...items.map((h) => {
      const li = el("li", `l${levels.indexOf(h.level) + 1}`);
      const btn = el("button");
      btn.type = "button";
      btn.append(el("span", "dot"), el("span", "label", h.div.querySelector(".src").textContent));
      btn.addEventListener("click", () => {
        $("reader").classList.remove("show-outline");
        jumpTo(h.i);
      });
      li.append(btn);
      h.li = li;
      h.label = btn.querySelector(".label");
      return li;
    }),
  );
  reader.outline = items;
  if (items.length < 2) $("outline-list").replaceChildren(el("li", "hint", "没有章节标题"));
}

function applyUpdates(updates, initial) {
  let last = null;
  for (const [i, text, skipped] of updates) {
    const div = reader.nodes[i];
    if (!div || !div.classList.contains("pending")) continue;
    div.classList.remove("pending");
    div.classList.toggle("skipped", !!skipped);
    div.querySelector(".dst").textContent = skipped ? "" : (isEbookJob(reader.job) ? htmlToText(text) : text);
    if (!initial) {
      div.classList.add("fresh");
      setTimeout(() => div.classList.remove("fresh"), 1500);
    }
    reader.done += 1;
    last = div;
  }
  refreshOutlineState();
  // 跟随翻译进度：把最新翻好的段落带进视野；用户自己滚动后暂停，点“跟随”恢复
  if (last && !initial && reader.following) {
    reader.autoScrolling = true;
    last.scrollIntoView({ block: "center" });
    setTimeout(() => (reader.autoScrolling = false), 80);
  }
  const active = reader.job && ACTIVE.has(reader.job.status);
  $("follow-btn").hidden = reader.following || !active;
}

// 大纲上的点：灰色未开始、蓝色翻译中、绿色已完成；章节标题翻好后大纲显示译文
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
    const dst = h.div.querySelector(".dst").textContent;
    if (dst && state.readerView !== "src") h.label.textContent = dst;
  });
}

function jumpTo(i) {
  const div = reader.nodes[i];
  if (!div) return;
  setFollowing(false);
  div.scrollIntoView({ block: "start" });
  sendFocus(i);
}

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

// 用户在阅读器里滚动时，告诉后端优先翻译当前看到的位置
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

// 显示方式：对照 / 只看译文 / 只看原文
for (const r of document.querySelectorAll('input[name="reader-view"]')) {
  r.addEventListener("change", () => setReaderView(r.value));
}
function setReaderView(view) {
  state.readerView = view;
  const content = $("reader-content");
  content.classList.remove("view-both", "view-dst", "view-src");
  content.classList.add(`view-${view}`);
  for (const r of document.querySelectorAll('input[name="reader-view"]')) r.checked = r.value === view;
  try {
    localStorage.setItem("doc-translator-reader-view", view);
  } catch {
    /* 忽略 */
  }
}


// ---------- 初始化 ----------

async function init() {
  const [langs, providers] = await Promise.all([api("/api/languages"), api("/api/providers")]);
  state.providers = providers;
  state.languages = langs;
  for (const lang of langs) {
    const label = langLabel(lang.code);
    if (lang.code === "auto") {
      form.elements.source_lang.add(new Option("自动检测", "auto"));
      continue;
    }
    form.elements.source_lang.add(new Option(label, lang.code));
    form.elements.target_lang.add(new Option(label, lang.code));
  }
  for (const [key, meta] of Object.entries(providers)) {
    if (meta.llm) $("add-provider").add(new Option(meta.label, key));
  }
  const saved = loadSettings();
  for (const name of ["source_lang", "target_lang", "mode"]) {
    if (saved[name] !== undefined) form.elements[name].value = saved[name];
  }
  if (!form.elements.target_lang.value) form.elements.target_lang.value = defaultTargetLang();
  let view = "translate";
  let readerView = "both";
  try {
    view = localStorage.getItem("doc-translator-view") || view;
    readerView = localStorage.getItem("doc-translator-reader-view") || readerView;
  } catch {
    /* 忽略 */
  }
  setReaderView(["both", "dst", "src"].includes(readerView) ? readerView : "both");
  await Promise.all([loadServices(), refreshJobs(), api("/api/settings").then((s) => (state.retentionDays = s.retention_days))]);
  renderRetention();
  renderJobs();
  checkLangPair();
  showView(view);
  // 刷新页面时回到正在看的那本书（preview=false 的任务没有预览内容，不恢复）
  const m = location.hash.match(/^#read\/(\w+)$/);
  if (m && state.jobs.some((j) => j.id === m[1] && j.preview !== false)) openReader(m[1]);
}

init().catch((err) => alert(`初始化失败：${err.message}`));

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "doc-translator-settings";

const state = { providers: {}, services: [], defaultId: "google", selectedId: null };

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

function showView(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === name));
  }
  $("view-translate").hidden = name !== "translate";
  $("view-services").hidden = name !== "services";
  try {
    localStorage.setItem("doc-translator-view", name);
  } catch {
    /* 忽略 */
  }
}
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showView(tab.dataset.view));
}

// ---------- 服务数据 ----------

async function loadServices() {
  const data = await api("/api/services");
  state.services = data.services;
  state.defaultId = data.default;
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
  const previous = select.value || loadSettings().service_id;
  select.replaceChildren();
  for (const s of state.services.filter((s) => s.enabled)) {
    const label = s.id === state.defaultId ? `${s.name}（默认）` : s.name;
    select.add(new Option(label, s.id));
  }
  const ids = [...select.options].map((o) => o.value);
  select.value = ids.includes(previous) ? previous : state.defaultId;
}

function renderServiceList() {
  const list = $("service-list");
  list.replaceChildren();
  for (const s of state.services) {
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
    toggle.disabled = s.id === state.defaultId;
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
    list.append(li);
  }
}

// ---------- 服务详情 ----------

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

  const editable = meta.llm;
  $("svc-fields").hidden = !editable;
  $("save-btn").hidden = !editable;
  $("delete-btn").hidden = s.builtin;
  if (!editable) return;

  const notice = $("key-notice");
  notice.replaceChildren();
  if (meta.needs_key && !s.api_key_hint) {
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
  f.api_key.placeholder = s.api_key_hint ? `已保存（${s.api_key_hint}），留空则不修改` : meta.needs_key ? "必填" : "可选";
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

  const models = meta.models || [];
  setModelOptions(models, s.model);
  f.model.value = s.model;
  // 没有预置模型（如 OpenAI 兼容接口）时默认用输入框
  useCustomModel(!models.length);
}

// 默认提示词随目标语言变化，留空的输入框里用 placeholder 展示
const defaultPromptCache = {};

async function getDefaultPrompts() {
  const lang = $("target_lang").value || "zh-CN";
  if (!defaultPromptCache[lang]) {
    defaultPromptCache[lang] = await api(`/api/prompts/default?target_lang=${encodeURIComponent(lang)}`);
  }
  return defaultPromptCache[lang];
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

$("target_lang").addEventListener("change", showDefaultPromptPlaceholders);

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
  };
}

svcForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formPayload();
  if (!payload.model) {
    $("save-status").textContent = "请先选择或输入模型";
    return;
  }
  try {
    await api(`/api/services/${payload.id}`, { method: "PUT", json: payload });
    await loadServices();
    $("save-status").textContent = "已保存";
  } catch (err) {
    $("save-status").textContent = `保存失败：${err.message}`;
  }
});

$("delete-btn").addEventListener("click", async () => {
  const s = selected();
  if (!s || !confirm(`确定删除「${s.name}」吗？`)) return;
  try {
    await api(`/api/services/${s.id}`, { method: "DELETE" });
    state.selectedId = state.defaultId;
    await loadServices();
  } catch (err) {
    alert(err.message);
  }
});

$("test-btn").addEventListener("click", async () => {
  const box = $("test-result");
  const s = selected();
  box.hidden = false;
  box.className = "test-result";
  box.textContent = "测试中…";
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

$("fetch-models").addEventListener("click", async () => {
  const btn = $("fetch-models");
  btn.disabled = true;
  btn.textContent = "获取中…";
  try {
    const payload = formPayload();
    payload.model = payload.model || "placeholder"; // 获取列表时还没选模型
    const { models } = await api("/api/services/models", { json: payload });
    if (!models.length) throw new Error("接口没有返回模型");
    const current = $("custom-model").checked ? $("model-input").value : $("model-select").value;
    setModelOptions(models, models.includes(current) ? current : models[0]);
    useCustomModel(false);
    btn.textContent = `已获取 ${models.length} 个模型`;
  } catch (err) {
    btn.textContent = "获取模型列表";
    alert(`获取失败：${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

$("add-btn").addEventListener("click", async () => {
  try {
    const svc = await api("/api/services", { json: { provider: $("add-provider").value } });
    state.selectedId = svc.id;
    await loadServices();
    svcForm.elements.api_key.focus();
  } catch (err) {
    alert(err.message);
  }
});

// ---------- 翻译页 ----------

const form = $("form");
const fileInput = $("file");
const drop = $("drop");
const SAVED_FIELDS = ["target_lang", "mode", "service_id"];

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

function showFile() {
  const f = fileInput.files[0];
  $("drop-text").textContent = f ? `${f.name}（${(f.size / 1024 / 1024).toFixed(2)} MB）` : "点击选择文件，或把文件拖到这里";
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

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function render(job) {
  $("job-title").textContent = job.filename;
  const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
  $("bar").value = job.status === "done" ? 100 : pct;
  const outputs = $("outputs");
  outputs.replaceChildren();
  if (job.status === "error") {
    setStatus(`翻译失败：${job.error}`, true);
  } else if (job.status === "done") {
    setStatus("翻译完成");
    for (const o of job.outputs) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = o.url;
      a.textContent = `下载 ${o.name}`;
      a.download = o.name;
      li.append(a);
      outputs.append(li);
    }
  } else if (job.total) {
    setStatus(`翻译中 ${job.done} / ${job.total} 段（${pct}%）`);
  } else {
    setStatus("正在解析文档…");
  }
}

async function poll(id) {
  while (true) {
    const res = await fetch(`/api/jobs/${id}`);
    if (!res.ok) {
      setStatus("任务丢失，服务可能已重启", true);
      return;
    }
    const job = await res.json();
    render(job);
    if (job.status === "done" || job.status === "error") return;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;
  saveSettings();
  const btn = $("submit");
  btn.disabled = true;
  $("job").hidden = false;
  $("outputs").replaceChildren();
  $("bar").value = 0;
  $("job-title").textContent = fileInput.files[0].name;
  setStatus("上传中…");
  try {
    const data = await api("/api/jobs", { method: "POST", body: new FormData(form) });
    await poll(data.id);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ---------- 初始化 ----------

async function init() {
  const [langs, providers] = await Promise.all([api("/api/languages"), api("/api/providers")]);
  state.providers = providers;
  for (const [code, name] of Object.entries(langs)) {
    form.elements.target_lang.add(new Option(name, code));
  }
  for (const [key, meta] of Object.entries(providers)) {
    if (meta.llm) $("add-provider").add(new Option(meta.label, key));
  }
  const saved = loadSettings();
  for (const name of ["target_lang", "mode"]) {
    if (saved[name] !== undefined) form.elements[name].value = saved[name];
  }
  await loadServices();
  let view = "translate";
  try {
    view = localStorage.getItem("doc-translator-view") || view;
  } catch {
    /* 忽略 */
  }
  showView(view);
}

init().catch((err) => alert(`初始化失败：${err.message}`));

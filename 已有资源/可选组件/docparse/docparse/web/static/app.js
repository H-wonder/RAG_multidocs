"use strict";

const $ = (sel) => document.querySelector(sel);

const STATUS = {
  queued: ["排队", "muted"],
  running: ["解析中", "info"],
  done: ["完成", "ok"],
  partial: ["部分完成", "warn"],
  failed: ["失败", "err"],
  cancelled: ["已取消", "muted"],
  interrupted: ["已中断", "warn"],
};

let selectedFiles = [];
let jobsCache = [];
let currentJob = null;
let toastTimer = null;

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function badge(status) {
  const [label, cls] = STATUS[status] || [status, "muted"];
  return h("span", { class: `badge ${cls}`, text: label });
}

function fileUrl(jobId, name) {
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${encodeURIComponent(name)}`;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add("hidden"), 4200);
}

/* ── file selection ─────────────────────────────────────────────────────── */
function addFiles(list) {
  for (const file of list) selectedFiles.push(file);
  renderFiles();
}

function renderFiles() {
  const root = $("#fileList");
  root.innerHTML = "";
  selectedFiles.forEach((file, index) => {
    root.append(h("li", {},
      h("span", { class: "name", text: file.name }),
      h("span", { class: "size", text: `${(file.size / 1024 / 1024).toFixed(2)} MB` }),
      h("button", {
        class: "ghost", type: "button",
        onclick: () => { selectedFiles.splice(index, 1); renderFiles(); },
      }, "移除"),
    ));
  });
  $("#start").disabled = selectedFiles.length === 0;
}

/* ── submit ─────────────────────────────────────────────────────────────── */
async function start() {
  if (!selectedFiles.length) return;
  const options = {
    mode: $("#mode").value,
    dpi: Number($("#dpi").value) || 200,
    max_pages: $("#maxPages").value ? Number($("#maxPages").value) : null,
    fields: $("#fields").value.split(",").map((s) => s.trim()).filter(Boolean),
    formats: [...document.querySelectorAll(".formats input:checked")].map((i) => i.value),
  };
  const form = new FormData();
  for (const file of selectedFiles) form.append("files", file, file.name);
  form.append("options", JSON.stringify(options));

  $("#start").disabled = true;
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || res.statusText);
    }
    selectedFiles = [];
    renderFiles();
    await refreshJobs();
  } catch (err) {
    toast(err.message || String(err));
  } finally {
    $("#start").disabled = selectedFiles.length === 0;
  }
}

/* ── jobs ───────────────────────────────────────────────────────────────── */
async function refreshJobs() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const data = await res.json();
    jobsCache = data.jobs || [];
    renderJobs();
  } catch {
    /* keep the last known list */
  }
}

function renderJobs() {
  const root = $("#jobs");
  root.innerHTML = "";
  if (!jobsCache.length) {
    root.append(h("p", { class: "muted", text: "暂无任务" }));
    return;
  }
  for (const job of jobsCache) root.append(jobCard(job));
}

function jobCard(job) {
  const active = job.status === "queued" || job.status === "running";
  const retryable = ["failed", "partial", "cancelled", "interrupted"].includes(job.status);
  const hasOutput = job.files.some((f) => f.outputs && f.outputs.length);
  return h("div", { class: "job" },
    h("div", { class: "job-head" },
      h("span", { class: "job-id", text: job.id }),
      badge(job.status),
      h("span", { class: "job-opts",
        text: `${job.options.mode || "auto"} · ${(job.options.formats || []).join(", ")}` }),
      h("span", { class: "spacer" }),
      hasOutput ? h("button", {
        class: "ghost", type: "button", onclick: () => openJob(job.id),
      }, "查看") : null,
      hasOutput ? h("a", { class: "ghost", href: `/api/jobs/${job.id}/bundle.zip` }, "ZIP") : null,
      active ? h("button", {
        class: "ghost", type: "button", onclick: () => cancelJob(job.id),
      }, "取消") : null,
      retryable ? h("button", {
        class: "ghost", type: "button", onclick: () => retryJob(job.id),
      }, "重试") : null,
    ),
    h("div", { class: "job-files" }, ...job.files.map((task) => fileRow(job, task))),
  );
}

function fileRow(job, task) {
  const progress = task.pages_total ? `${task.pages_done}/${task.pages_total} 页` : "";
  return h("div", { class: "file-row" },
    h("span", { class: "name", text: task.name }),
    badge(task.status),
    h("span", { class: "progress", text: progress }),
    task.warnings ? h("span", { class: "warn-text", text: `⚠ ${task.warnings}` }) : null,
    task.error ? h("span", { class: "err-text", text: task.error }) : null,
    h("span", { class: "links" },
      ...(task.outputs || []).map((name) => h("a", { href: fileUrl(job.id, name), text: name }))),
  );
}

async function cancelJob(id) {
  const res = await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
  if (!res.ok) toast("取消失败");
  await refreshJobs();
}

async function retryJob(id) {
  const res = await fetch(`/api/jobs/${id}/retry`, { method: "POST" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    toast(detail.detail || "重试失败");
  }
  await refreshJobs();
}

/* ── preview modal ──────────────────────────────────────────────────────── */
async function openJob(id) {
  const res = await fetch(`/api/jobs/${id}`);
  if (!res.ok) return;
  currentJob = await res.json();
  $("#modalTitle").textContent = `结果 · ${currentJob.id}`;
  $("#bundleLink").href = `/api/jobs/${id}/bundle.zip`;

  const tabs = $("#tabs");
  tabs.innerHTML = "";
  const outputs = [];
  for (const file of currentJob.files) {
    for (const name of (file.outputs || [])) outputs.push(name);
  }
  outputs.sort();
  const entries = outputs.map((name) => ({
    label: name, name, kind: name.split(".").pop().toLowerCase(),
  }));
  entries.push({ label: "告警", kind: "warnings" });

  let first = null;
  for (const entry of entries) {
    const button = h("button", { type: "button" }, entry.label);
    button.addEventListener("click", () => selectTab(entry, button));
    tabs.append(button);
    if (!first) first = { entry, button };
  }
  $("#modal").classList.remove("hidden");
  if (first) selectTab(first.entry, first.button);
}

async function selectTab(entry, button) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b === button));
  if (entry.kind === "warnings") {
    await renderWarnings();
    return;
  }
  const preview = $("#preview");
  preview.innerHTML = "<p class='muted'>加载中…</p>";
  try {
    const res = await fetch(fileUrl(currentJob.id, entry.name));
    const text = await res.text();
    preview.innerHTML = "";
    if (entry.kind === "html") {
      const frame = document.createElement("iframe");
      frame.srcdoc = text;
      preview.append(frame);
    } else {
      let body = text;
      if (entry.kind === "json") {
        try { body = JSON.stringify(JSON.parse(text), null, 2); } catch { /* keep raw */ }
      }
      const pre = document.createElement("pre");
      pre.textContent = body;
      preview.append(pre);
    }
  } catch (err) {
    preview.innerHTML = "";
    preview.append(h("p", { class: "err-text", text: String(err) }));
  }
}

async function renderWarnings() {
  const preview = $("#preview");
  preview.innerHTML = "";
  const names = currentJob.files.flatMap((f) => (f.outputs || []).filter((n) => n.endsWith(".json")));
  if (!names.length) {
    preview.append(h("p", { class: "muted", text: "没有 JSON 输出，无法汇总告警。" }));
    return;
  }
  let total = 0;
  const list = h("div", { class: "jobs-list" });
  for (const name of names) {
    let data;
    try {
      data = await (await fetch(fileUrl(currentJob.id, name))).json();
    } catch {
      continue;
    }
    const warnings = data.warnings || [];
    total += warnings.length;
    const block = h("div", { class: "job" },
      h("div", { class: "job-head" },
        h("span", { class: "job-id", text: name }),
        h("span", { class: "progress", text: `${warnings.length} 条` })));
    for (const warning of warnings) {
      block.append(h("div", { class: "file-row" },
        h("span", { class: `badge ${warning.severity === "high" ? "err" : "warn"}`,
          text: warning.severity || "info" }),
        h("span", { class: "job-id", text: warning.code || "" }),
        h("span", { class: "name", text: warning.message || "" })));
    }
    list.append(block);
  }
  preview.append(h("p", { class: total ? "warn-text" : "muted", text: `共 ${total} 条告警` }));
  preview.append(list);
}

/* ── health / init ──────────────────────────────────────────────────────── */
async function refreshHealth() {
  try {
    const data = await (await fetch("/api/health")).json();
    $("#version").textContent = `v${data.version}`;
    const node = $("#health");
    const ok = !!data.ollama;
    node.textContent = ok
      ? `Ollama · ${(data.models || []).join(", ") || "无模型"}`
      : "Ollama 不可达";
    node.className = `health ${ok ? "ok" : "err"}`;
    node.title = `layout ${data.layout ? "已安装" : "未安装"} · 磁盘剩余 ${data.disk_free_gb} GB`;
  } catch {
    const node = $("#health");
    node.textContent = "服务不可达";
    node.className = "health err";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#fileInput").addEventListener("change", (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  });
  $("#browse").addEventListener("click", () => $("#fileInput").click());
  $("#start").addEventListener("click", start);
  $("#refresh").addEventListener("click", refreshJobs);
  $("#closeModal").addEventListener("click", () => $("#modal").classList.add("hidden"));
  $("#modal").addEventListener("click", (event) => {
    if (event.target.id === "modal") $("#modal").classList.add("hidden");
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("#modal").classList.add("hidden");
  });

  const dropzone = $("#dropzone");
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
    addFiles(event.dataTransfer.files);
  });

  refreshHealth();
  refreshJobs();
  setInterval(refreshJobs, 2000);
  setInterval(refreshHealth, 15000);
});

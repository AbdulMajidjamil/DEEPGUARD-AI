/* SPDX-License-Identifier: Apache-2.0
   Copyright 2026 Abdul Majid Jamil - DEEPGUARD AI dashboard.
   No inline handlers, no innerHTML with server data, no external requests (strict CSP). */
(function () {
  "use strict";

  var VIEWS = {
    overview: "Overview", analysis: "Video Analysis", history: "Detection History", reports: "Forensic Reports",
    model: "Model Information", system: "System Health", settings: "Settings", about: "About"
  };
  var LS = { settings: "deepguard.settings", history: "deepguard.history" };
  var MAX_HISTORY = 200;

  var state = { file: null, busy: false, health: null, sysTimer: null, pollTimer: null };

  // ---------- helpers ----------
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, kids) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else if (k === "class") node.className = attrs[k];
      else if (k.indexOf("on") === 0) node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    });
    [].concat(kids || []).forEach(function (kid) {
      if (kid == null) return;
      node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    });
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function lsGet(key, fallback) {
    try { var raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch (e) { return fallback; }
  }
  function lsSet(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; } }
  function settings() { return Object.assign({ apiKey: "", pollMs: 2000, saveHistory: true }, lsGet(LS.settings, {})); }
  function pct(x) { return (Number(x) * 100).toFixed(1) + "%"; }
  function fmtBytes(n) { return n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB"; }
  function fmtDate(ts) { return ts ? new Date(ts * 1000).toLocaleString() : "-"; }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function toast(msg) {
    var t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(function () { t.hidden = true; }, 3500);
  }
  function headers() { var k = settings().apiKey; return k ? { "X-API-Key": k } : {}; }

  function api(path, opts) {
    opts = opts || {};
    opts.headers = Object.assign({}, headers(), opts.headers || {});
    return fetch(path, opts).then(function (res) {
      return res.text().then(function (text) {
        var body = null;
        try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
        if (!res.ok) {
          var detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : "Request rejected") : "HTTP " + res.status;
          var err = new Error(detail); err.status = res.status; throw err;
        }
        return body;
      });
    });
  }
  function download(path, filename) {
    return fetch(path, { headers: headers() }).then(function (res) {
      if (!res.ok) throw new Error(res.status === 404 ? "Result expired on the server" : "Download failed (HTTP " + res.status + ")");
      return res.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      var a = el("a", { href: url, download: filename });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    });
  }

  // ---------- routing ----------
  function route() {
    var name = (location.hash || "#overview").slice(1);
    if (!VIEWS[name]) name = "overview";
    Object.keys(VIEWS).forEach(function (v) { $("view-" + v).hidden = v !== name; });
    document.querySelectorAll("#nav a").forEach(function (a) {
      var on = a.getAttribute("data-view") === name;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    $("viewTitle").textContent = VIEWS[name];
    document.title = VIEWS[name] + " - DEEPGUARD AI";
    stopSysTimer();
    if (name === "overview") renderOverview();
    if (name === "history") renderHistory();
    if (name === "reports") renderReports();
    if (name === "model") loadModel();
    if (name === "system") { loadSystem(); if ($("autoRefresh").checked) startSysTimer(); }
    if (name === "settings") loadSettingsForm();
  }

  // ---------- health / banner ----------
  function refreshHealth() {
    return api("/api/v1/health").then(function (h) {
      state.health = h;
      var pill = $("connPill"); pill.textContent = "API online"; pill.className = "pill ok";
      var mode = $("modePill");
      mode.textContent = h.is_demo ? "DEMO MODE" : "Trained model";
      mode.className = "pill " + (h.is_demo ? "warn" : "ok");
      $("demoBanner").hidden = !h.is_demo;
      $("verText").textContent = h.version || "";
      return h;
    }).catch(function () {
      var pill = $("connPill"); pill.textContent = "API unreachable"; pill.className = "pill err";
      state.health = null;
    });
  }

  // ---------- history ----------
  function history() { return lsGet(LS.history, []); }
  function addHistory(job) {
    if (!settings().saveHistory) return;
    var list = history().filter(function (h) { return h.task_id !== job.task_id; });
    list.unshift({
      task_id: job.task_id, filename: job.filename, label: job.label, confidence: job.confidence,
      probability_fake: job.probability_fake, inference_mode: job.inference_mode, is_demo: !!job.is_demo,
      completed_at: job.completed_at || Date.now() / 1000, processing_ms: job.processing_ms || null
    });
    lsSet(LS.history, list.slice(0, MAX_HISTORY));
  }
  function modeText(h) { return h.is_demo ? "demo heuristic" : "trained model"; }

  function renderOverview() {
    var list = history();
    var fake = list.filter(function (h) { return h.label === "FAKE"; }).length;
    var real = list.filter(function (h) { return h.label === "REAL"; }).length;
    var times = list.map(function (h) { return h.processing_ms; }).filter(Boolean);
    $("kpiTotal").textContent = list.length;
    $("kpiFake").textContent = fake;
    $("kpiReal").textContent = real;
    $("kpiTime").textContent = times.length ? (times.reduce(function (a, b) { return a + b; }, 0) / times.length / 1000).toFixed(1) + " s" : "-";
    drawDonut($("donut"), [{ v: real, c: cssVar("--good"), l: "REAL" }, { v: fake, c: cssVar("--bad"), l: "FAKE" }]);
    $("donutNote").textContent = list.length ? real + " REAL / " + fake + " FAKE (includes demo-mode runs)" : "No analyses yet.";
    var ul = $("recentList"); clear(ul);
    if (!list.length) ul.appendChild(el("li", { class: "muted", text: "Nothing yet." }));
    list.slice(0, 5).forEach(function (h) {
      ul.appendChild(el("li", {}, [
        el("a", { href: "#history", text: h.filename || h.task_id.slice(0, 8) }),
        el("span", { class: h.label === "FAKE" ? "bad" : "good", text: h.label + " " + pct(h.confidence) })
      ]));
    });
  }

  function renderHistory() {
    var list = history(), body = document.querySelector("#historyTable tbody"); clear(body);
    $("historyEmpty").hidden = list.length > 0;
    list.forEach(function (h) {
      var open = el("button", { type: "button", class: "btn tiny", text: "Open" });
      open.addEventListener("click", function () { openTask(h.task_id); });
      var del = el("button", { type: "button", class: "btn tiny danger", text: "Delete" });
      del.addEventListener("click", function () {
        lsSet(LS.history, history().filter(function (x) { return x.task_id !== h.task_id; })); renderHistory();
      });
      body.appendChild(el("tr", {}, [
        el("td", { text: fmtDate(h.completed_at) }), el("td", { text: h.filename || "-" }),
        el("td", { class: h.label === "FAKE" ? "bad" : "good", text: h.label }),
        el("td", { text: pct(h.confidence) }), el("td", { text: modeText(h) }),
        el("td", {}, [open, " ", del])
      ]));
    });
  }

  function renderReports() {
    var list = history(), body = document.querySelector("#reportsTable tbody"); clear(body);
    $("reportsEmpty").hidden = list.length > 0;
    list.forEach(function (h) {
      var short = h.task_id.slice(0, 8);
      function btn(fmt) {
        var b = el("button", { type: "button", class: "btn tiny", text: fmt.toUpperCase() });
        b.addEventListener("click", function () {
          b.disabled = true;
          download("/api/v1/report/" + h.task_id + "?format=" + fmt, "deepguard-report-" + short + "." + fmt)
            .then(function () { toast(fmt.toUpperCase() + " report downloaded"); })
            .catch(function (e) { toast(e.message); })
            .then(function () { b.disabled = false; });
        });
        return b;
      }
      body.appendChild(el("tr", {}, [
        el("td", { text: fmtDate(h.completed_at) }), el("td", { text: h.filename || "-" }),
        el("td", { class: h.label === "FAKE" ? "bad" : "good", text: h.label }),
        el("td", {}, [el("code", { text: short })]), el("td", {}, [btn("json"), " ", btn("pdf")])
      ]));
    });
  }

  // ---------- upload / analysis ----------
  var ALLOWED = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];
  function setFile(file) {
    var err = $("formError"); err.hidden = true;
    if (!file) { state.file = null; $("fileInfo").hidden = true; $("analyzeBtn").disabled = true; return; }
    var ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    if (ALLOWED.indexOf(ext) < 0) {
      err.textContent = "Unsupported file type. Allowed: " + ALLOWED.join(", "); err.hidden = false; return;
    }
    state.file = file;
    $("fileName").textContent = file.name; $("fileSize").textContent = fmtBytes(file.size);
    $("fileInfo").hidden = false; $("analyzeBtn").disabled = state.busy;
  }

  function setBusy(busy) {
    state.busy = busy;
    $("analyzeBtn").disabled = busy || !state.file;
    $("sampleBtn").disabled = busy; $("ovSampleBtn").disabled = busy;
  }

  function showProgress(taskId) {
    $("progressCard").hidden = false; $("resultCard").hidden = true;
    $("taskId").textContent = taskId || "-";
    setBar("uploadBar", "uploadPct", 0); setBar("procBar", "procPct", 0); $("procMsg").textContent = "Starting...";
  }
  function setBar(barId, labelId, value) {
    var v = Math.max(0, Math.min(100, Math.round(value)));
    $(barId).value = v; $(labelId).textContent = v + "%";
  }

  function startUpload() {
    if (!state.file || state.busy) return;
    var form = new FormData(); form.append("file", state.file);
    setBusy(true); showProgress(null); location.hash = "#analysis";
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/detect");
    var key = settings().apiKey; if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.addEventListener("progress", function (ev) {
      if (ev.lengthComputable) setBar("uploadBar", "uploadPct", ev.loaded / ev.total * 100);
    });
    xhr.addEventListener("load", function () {
      var body = null; try { body = JSON.parse(xhr.responseText); } catch (e) { body = null; }
      if (xhr.status === 202 && body && body.task_id) {
        setBar("uploadBar", "uploadPct", 100); $("taskId").textContent = body.task_id; poll(body.task_id);
      } else {
        fail(body && typeof body.detail === "string" ? body.detail : "Upload rejected (HTTP " + xhr.status + ")");
      }
    });
    xhr.addEventListener("error", function () { fail("Could not reach the server."); });
    xhr.addEventListener("abort", function () { fail("Upload cancelled."); });
    xhr.send(form);
  }

  function startSample() {
    if (state.busy) return;
    setBusy(true); showProgress(null); location.hash = "#analysis";
    api("/api/v1/detect/sample", { method: "POST" }).then(function (b) {
      setBar("uploadBar", "uploadPct", 100); $("taskId").textContent = b.task_id; poll(b.task_id);
    }).catch(function (e) { fail(e.message); });
  }

  function fail(message) {
    setBusy(false);
    var err = $("formError"); err.textContent = message; err.hidden = false;
    $("procMsg").textContent = message;
  }

  function poll(taskId) {
    clearTimeout(state.pollTimer);
    api("/api/v1/status/" + taskId).then(function (job) {
      setBar("procBar", "procPct", (job.progress || 0) * 100);
      $("procMsg").textContent = (job.message || job.status) + (job.stage ? " (" + job.stage + ")" : "");
      if (job.status === "completed") {
        setBusy(false); addHistory(job); renderResult(job); $("progressCard").hidden = true; toast("Analysis complete");
      } else if (job.status === "failed") {
        setBusy(false); fail(job.message || "Processing failed.");
      } else {
        state.pollTimer = setTimeout(function () { poll(taskId); }, settings().pollMs);
      }
    }).catch(function (e) {
      if (e.status === 429) { state.pollTimer = setTimeout(function () { poll(taskId); }, 5000); return; }
      fail(e.message);
    });
  }

  function openTask(taskId) {
    location.hash = "#analysis"; $("progressCard").hidden = true; $("formError").hidden = true;
    api("/api/v1/status/" + taskId).then(function (job) {
      if (job.status === "completed") renderResult(job);
      else if (job.status === "failed") fail(job.message || "That analysis failed.");
      else { setBusy(true); showProgress(taskId); poll(taskId); }
    }).catch(function (e) { fail(e.status === 404 ? "That result has expired on the server." : e.message); });
  }

  // ---------- result rendering ----------
  function metric(label, value) { return el("div", { class: "metric" }, [el("span", { text: label }), el("b", { text: String(value) })]); }

  function renderResult(job) {
    var card = $("resultCard"); clear(card); card.hidden = false;
    var v = job.video || {};
    var verdict = el("div", { class: "verdict" }, [
      el("div", {}, [el("span", { class: "badge " + job.label, text: job.label }),
        job.is_demo ? el("span", { class: "chip", text: "DEMO RESULT" }) : null]),
      el("div", { class: "meter" }, [
        el("div", {}, ["Manipulation probability: ", el("b", { text: pct(job.manipulation_probability) })]),
        el("div", { class: "bar" }, [el("i", { "data-w": String(job.manipulation_probability) })]),
        el("small", { class: "muted", text: "Confidence in verdict: " + pct(job.confidence) + " (model score, not guaranteed accuracy)" })
      ])
    ]);
    card.appendChild(el("h2", { text: "Result: " + (job.filename || "") }));
    card.appendChild(verdict);
    verdict.querySelector("i").style.width = (Number(job.manipulation_probability) * 100).toFixed(1) + "%";

    if (job.warnings && job.warnings.length) {
      card.appendChild(el("ul", { class: "warnings" }, job.warnings.map(function (w) { return el("li", { text: w }); })));
    }
    card.appendChild(el("div", { class: "metrics" }, [
      metric("Frames sampled", job.frames_sampled), metric("Frames with a face", job.faces_detected),
      metric("Max faces in a frame", job.max_faces_in_frame), metric("Duration", (v.duration || 0) + " s"),
      metric("Resolution", (v.width || "?") + "x" + (v.height || "?")),
      metric("FPS", (v.fps || "?") + (v.fps_estimated ? " (estimated)" : "")),
      metric("Inference mode", job.inference_mode), metric("Device", job.device),
      metric("Model ID", job.model_id), metric("Face detector", job.detector),
      metric("Processing time", ((job.processing_ms || 0) / 1000).toFixed(1) + " s")
    ]));

    if (job.persons && job.persons.length > 1) {
      card.appendChild(el("h3", { text: "Per-person results" }));
      var rows = job.persons.map(function (p) {
        return el("tr", {}, [el("td", { text: "Person " + (p.track_id + 1) }),
          el("td", { class: p.label === "FAKE" ? "bad" : "good", text: p.label }),
          el("td", { text: pct(p.probability_fake) }), el("td", { text: p.faces_analysed })]);
      });
      card.appendChild(el("div", { class: "table-wrap" }, [el("table", {}, [
        el("thead", {}, [el("tr", {}, ["Track", "Label", "P(fake)", "Faces analysed"].map(function (t) { return el("th", { text: t }); }))]),
        el("tbody", {}, rows)])]));
    }

    card.appendChild(el("h3", { text: "Frame-level anomaly timeline" }));
    var canvas = el("canvas", { width: "820", height: "200", role: "img", "aria-label": "Per-frame anomaly index over time" });
    card.appendChild(canvas);
    card.appendChild(el("p", { class: "muted", text: "Anomaly index highlights where temporal artefacts concentrate. It is a lead for human review, not a classifier output." }));
    drawLine(canvas, (job.frame_scores || []).map(function (f) { return f.anomaly; }),
      (job.frame_scores || []).map(function (f) { return f.timestamp; }));

    card.appendChild(el("h3", { text: "Suspicious frames" }));
    var frames = job.suspicious_frames || [];
    if (!frames.length) card.appendChild(el("p", { class: "muted", text: "No frame stood out above the anomaly threshold." }));
    else card.appendChild(el("div", { class: "gallery" }, frames.map(function (f) {
      var img = el("img", { alt: "Suspicious face crop at " + f.timestamp + " seconds", loading: "lazy" });
      img.src = f.thumb;
      return el("figure", {}, [img, el("figcaption", { text: "frame " + f.index + " @ " + f.timestamp + "s - anomaly " + f.anomaly })]);
    })));

    var short = job.task_id.slice(0, 8);
    function dl(fmt) {
      var b = el("button", { type: "button", class: "btn", text: "Download " + fmt.toUpperCase() });
      b.addEventListener("click", function () {
        b.disabled = true;
        download("/api/v1/report/" + job.task_id + "?format=" + fmt, "deepguard-report-" + short + "." + fmt)
          .catch(function (e) { toast(e.message); }).then(function () { b.disabled = false; });
      });
      return b;
    }
    card.appendChild(el("div", { class: "row" }, [dl("json"), dl("pdf")]));
    card.appendChild(el("p", { class: "muted", text: job.disclaimer || "" }));
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- charts (canvas) ----------
  function prepCanvas(c) {
    var ratio = window.devicePixelRatio || 1, w = c.width, h = c.height;
    if (!c._base) c._base = [w, h];
    w = c._base[0]; h = c._base[1];
    c.width = w * ratio; c.height = h * ratio;
    var ctx = c.getContext("2d"); ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }
  function drawLine(canvas, values, times) {
    var p = prepCanvas(canvas), ctx = p.ctx, w = p.w, h = p.h, pad = 34;
    ctx.font = "11px system-ui"; ctx.fillStyle = cssVar("--muted"); ctx.strokeStyle = cssVar("--line");
    for (var g = 0; g <= 4; g++) {
      var y = pad / 2 + (h - pad * 1.4) * g / 4;
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - 8, y); ctx.stroke();
      ctx.fillText((1 - g / 4).toFixed(2), 2, y + 4);
    }
    if (!values.length) { ctx.fillText("No frame scores", pad + 10, h / 2); return; }
    var n = values.length, plotW = w - pad - 8, plotH = h - pad * 1.4;
    function X(i) { return pad + (n === 1 ? plotW / 2 : plotW * i / (n - 1)); }
    function Y(v) { return pad / 2 + plotH * (1 - Math.max(0, Math.min(1, v))); }
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(34,211,238,.35)"); grad.addColorStop(1, "rgba(34,211,238,0)");
    ctx.beginPath(); ctx.moveTo(X(0), Y(0));
    values.forEach(function (v, i) { ctx.lineTo(X(i), Y(v)); });
    ctx.lineTo(X(n - 1), Y(0)); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath(); values.forEach(function (v, i) { if (i) ctx.lineTo(X(i), Y(v)); else ctx.moveTo(X(i), Y(v)); });
    ctx.strokeStyle = cssVar("--cyan"); ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = cssVar("--muted");
    ctx.fillText((times[0] || 0) + " s", pad, h - 8);
    var last = (times[n - 1] || 0) + " s"; ctx.fillText(last, w - 8 - ctx.measureText(last).width, h - 8);
  }
  function drawDonut(canvas, parts) {
    var p = prepCanvas(canvas), ctx = p.ctx, cx = p.w / 2, cy = p.h / 2, r = 78;
    var total = parts.reduce(function (a, x) { return a + x.v; }, 0);
    ctx.lineWidth = 26;
    if (!total) {
      ctx.strokeStyle = cssVar("--line"); ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = cssVar("--muted"); ctx.textAlign = "center"; ctx.font = "13px system-ui"; ctx.fillText("no data", cx, cy + 4);
      ctx.textAlign = "start"; return;
    }
    var start = -Math.PI / 2;
    parts.forEach(function (x) {
      if (!x.v) return;
      var end = start + Math.PI * 2 * x.v / total;
      ctx.strokeStyle = x.c; ctx.beginPath(); ctx.arc(cx, cy, r, start, end); ctx.stroke(); start = end;
    });
    ctx.fillStyle = cssVar("--text"); ctx.textAlign = "center"; ctx.font = "700 26px system-ui"; ctx.fillText(String(total), cx, cy + 6);
    ctx.font = "11px system-ui"; ctx.fillStyle = cssVar("--muted"); ctx.fillText("analyses", cx, cy + 24); ctx.textAlign = "start";
  }

  // ---------- model / system ----------
  function dl(target, pairs) {
    clear(target);
    pairs.forEach(function (p) { target.appendChild(el("dt", { text: p[0] })); target.appendChild(el("dd", { text: p[1] == null || p[1] === "" ? "-" : String(p[1]) })); });
  }
  function loadModel() {
    api("/api/v1/model").then(function (m) {
      var c = m.checkpoint || {};
      dl($("modelDl"), [
        ["Inference mode", m.inference_mode + (m.is_demo ? " (DEMO - no trained checkpoint)" : "")],
        ["Model ID", m.model_id], ["Task", m.task], ["Architecture", m.architecture],
        ["Spatial encoder", m.spatial], ["Temporal model", m.temporal], ["Head", m.head], ["Loss (training)", m.loss],
        ["Input", m.input], ["Device", m.device + (m.device_note ? " - " + m.device_note : "")],
        ["Weights file present", m.weights_present ? "yes" : "no"], ["Weights loaded", m.weights_loaded ? "yes (validated)" : "no"],
        ["Checkpoint hash", c.sha256], ["Checkpoint epoch", c.epoch],
        ["Checkpoint rejected", m.checkpoint_error], ["Parameters", m.parameters_millions ? m.parameters_millions + " M" : null],
        ["Modalities", "video: " + m.modalities.video + ", image: " + m.modalities.image + ", audio: " + m.modalities.audio],
        ["Note", m.note], ["License", m.license]
      ]);
      renderEval(m.evaluation);
    }).catch(function (e) { toast(e.message); });
  }
  function renderEval(ev) {
    var box = $("evalBox"); clear(box);
    if (!ev || !ev.metrics) {
      box.appendChild(el("p", { class: "muted", text: "No evaluation report found. Metrics appear here only after you run scripts/evaluate.py on a labelled test set with a trained checkpoint. No accuracy figures are claimed for this build." }));
      return;
    }
    var m = ev.metrics;
    var keys = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"], ["fpr", "False-positive rate"], ["fnr", "False-negative rate"]];
    box.appendChild(el("div", { class: "metrics" }, keys.filter(function (k) { return m[k[0]] != null; })
      .map(function (k) { return metric(k[1], Number(m[k[0]]).toFixed(4)); })));
    if (ev.confusion_matrix) {
      var cm = ev.confusion_matrix;
      box.appendChild(el("table", { class: "cm" }, [
        el("tr", {}, [el("th", { text: "" }), el("th", { text: "Pred REAL" }), el("th", { text: "Pred FAKE" })]),
        el("tr", {}, [el("th", { text: "True REAL" }), el("td", { text: cm[0][0] }), el("td", { text: cm[0][1] })]),
        el("tr", {}, [el("th", { text: "True FAKE" }), el("td", { text: cm[1][0] }), el("td", { text: cm[1][1] })])
      ]));
    }
    box.appendChild(el("p", { class: "muted", text: "Evaluated on " + (ev.n_samples || "?") + " samples" + (ev.dataset ? " (" + ev.dataset + ")" : "") + ". Results apply only to data resembling that test set." }));
  }

  function svc(name, level, text) {
    return el("li", {}, [el("span", {}, [el("span", { class: "dot " + level }), name]), el("span", { class: "muted", text: text })]);
  }
  function loadSystem() {
    refreshHealth().then(function (h) {
      var ul = $("svcList"); clear(ul);
      if (!h) { ul.appendChild(svc("API", "err", "unreachable")); return; }
      ul.appendChild(svc("API", "ok", "online, up " + h.uptime_seconds + " s"));
      ul.appendChild(svc("Redis", h.redis ? "ok" : "warn", h.redis ? "connected" : "not available - in-process job store"));
      ul.appendChild(svc("Celery worker", h.celery_worker ? "ok" : "warn", h.celery_worker ? "running" : "none - using in-process worker"));
      ul.appendChild(svc("Model", h.is_demo ? "warn" : "ok", h.inference_mode));
      ul.appendChild(svc("Authentication", h.auth === "api_key" ? "ok" : "warn", h.auth === "api_key" ? "API key required" : "open (no API key set)"));
    });
    api("/api/v1/system").then(function (s) {
      dl($("sysDl"), [
        ["Version", s.version], ["Python", s.python], ["Platform", s.platform], ["PyTorch", s.torch], ["OpenCV", s.opencv],
        ["CUDA available", s.cuda_available ? "yes" : "no"], ["Compute device", s.device + (s.device_note ? " - " + s.device_note : "")],
        ["Dispatch", s.dispatch], ["Job backend", s.job_backend],
        ["Disk free", s.disk.free_gb + " / " + s.disk.total_gb + " GB"],
        ["Max upload", s.limits.max_upload_mb + " MB"], ["Max video length", s.limits.max_video_seconds + " s"],
        ["Allowed types", s.limits.allowed_extensions.join(" ")], ["Upload retention", s.limits.upload_retention_seconds + " s"]
      ]);
    }).catch(function (e) { dl($("sysDl"), [["Runtime info", e.status === 401 ? "API key required - set it in Settings" : e.message]]); });
  }
  function startSysTimer() { stopSysTimer(); state.sysTimer = setInterval(loadSystem, 5000); }
  function stopSysTimer() { if (state.sysTimer) { clearInterval(state.sysTimer); state.sysTimer = null; } }

  // ---------- settings ----------
  function loadSettingsForm() {
    var s = settings();
    $("apiKey").value = s.apiKey; $("pollMs").value = String(s.pollMs); $("saveHistory").checked = !!s.saveHistory;
  }

  // ---------- wiring ----------
  function init() {
    window.addEventListener("hashchange", route);
    $("refreshBtn").addEventListener("click", function () { refreshHealth().then(function () { toast("Status refreshed"); route(); }); });
    var drop = $("drop"), input = $("fileInput");
    drop.addEventListener("click", function () { input.click(); });
    drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (e) { if (e.dataTransfer && e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });
    input.addEventListener("change", function () { setFile(input.files[0] || null); });
    $("clearFile").addEventListener("click", function () { input.value = ""; setFile(null); });
    $("analyzeBtn").addEventListener("click", startUpload);
    $("sampleBtn").addEventListener("click", startSample);
    $("ovSampleBtn").addEventListener("click", startSample);
    $("copyTask").addEventListener("click", function () {
      var id = $("taskId").textContent;
      if (navigator.clipboard && id && id !== "-") navigator.clipboard.writeText(id).then(function () { toast("Task ID copied"); }, function () { toast("Copy failed"); });
      else toast("Copy not available in this browser");
    });
    $("clearHistory").addEventListener("click", function () {
      if (history().length && confirm("Remove all saved analyses from this browser?")) { lsSet(LS.history, []); renderHistory(); toast("History cleared"); }
    });
    $("settingsForm").addEventListener("submit", function (e) {
      e.preventDefault();
      lsSet(LS.settings, { apiKey: $("apiKey").value.trim(), pollMs: parseInt($("pollMs").value, 10) || 2000, saveHistory: $("saveHistory").checked });
      toast("Settings saved"); refreshHealth();
    });
    $("testKey").addEventListener("click", function () {
      var key = $("apiKey").value.trim();
      fetch("/api/v1/system", { headers: key ? { "X-API-Key": key } : {} }).then(function (r) {
        toast(r.ok ? "Connection OK" : r.status === 401 ? "API key rejected" : "Server returned HTTP " + r.status);
      }).catch(function () { toast("Could not reach the server"); });
    });
    $("autoRefresh").addEventListener("change", function () { if (this.checked) { loadSystem(); startSysTimer(); } else stopSysTimer(); });
    window.addEventListener("resize", function () { if (!$("view-overview").hidden) renderOverview(); });
    refreshHealth().then(route);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();

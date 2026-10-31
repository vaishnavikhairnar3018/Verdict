// ============================== Helpers ==============================

const $ = (selector) => document.querySelector(selector);
const API = window.VERDICT_API || "";   // backend address, set in config.js

// Web text (titles, snippets, claims) is untrusted: escape it before putting it in HTML,
// or a page title like <img onerror=...> could run code in our dashboard (XSS).
const esc = (value) => String(value ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  shield: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><path d="m9 12 2 2 4-4"/>',
  scale: '<path d="M12 3v18M7 21h10M5 7h14M5 7l-3 7a3 3 0 0 0 6 0zM19 7l-3 7a3 3 0 0 0 6 0z"/>',
  pen: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13 7 4 4"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  minus: '<path d="M6 12h12"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5V14M12 17h.01"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
};
// Only http(s) links may become clickable; "javascript:" or broken URLs become harmless "#".
function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : "#";
  } catch { return "#"; }
}

function domainOf(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return "unknown source"; }
}

const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;

const AGENTS = [
  { id: "researcher", name: "Researcher", tint: "--lavender", icon: "search",
    desc: "Searches the web and extracts claims, each with its sources." },
  { id: "skeptic", name: "Skeptic", tint: "--blue", icon: "shield",
    desc: "Hunts for evidence against every claim and checks the sources." },
  { id: "judge", name: "Judge", tint: "--green", icon: "scale",
    desc: "Weighs both sides and scores how much to trust each claim." },
  { id: "writer", name: "Writer", tint: "--peach", icon: "pen",
    desc: "Answers using only the claims that survived." },
];
const EXAMPLES = [
  "Is breakfast the most important meal of the day?",
  "Does cracking your knuckles cause arthritis?",
  "Do goldfish have a 3-second memory?",
  "Can AI detectors reliably identify AI-written text?",
];
const VERDICT_ICON = { supported: "check", refuted: "x", contested: "scale", unverifiable: "help" };
const AVATAR_TINTS = ["--lavender", "--blue", "--green", "--peach"];

let config = { threshold: 0.6 };
let run = null;               // the run currently on screen
let showAllSources = false;   // Sources card: preview or full list
const SOURCES_PREVIEW = 6;

// ============================== Theme & navigation ==============================

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelectorAll("[data-theme-set]").forEach((b) => b.classList.toggle("current", b.dataset.themeSet === theme));
  try { localStorage.setItem("theme", theme); } catch {}
}

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
  document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "history") renderHistory();
}

// ============================== Agents panel ==============================

function renderAgents() {
  $("#agents").innerHTML = AGENTS.map((a) => `
    <div class="agent" id="agent-${a.id}" style="--tint: var(${a.tint})">
      <h3>${a.name}</h3>
      <p>${a.desc}</p>
      <div class="agent-foot">
        <span class="agent-icon">${icon(a.icon)}</span>
        <span class="agent-status">Ready</span>
      </div>
    </div>`).join("");
}

// What each agent did, computed from a finished run (used for past runs too).
function showAgentSummaries() {
  const t = config.threshold;
  const strong = run.objections.filter((o) => o.strength === "strong").length;
  const kept = run.verdicts.filter((v) => v.final_confidence >= t).length;
  const rounds = run.round > 1 ? ` in ${run.round} rounds` : "";
  const took = (id) => (run.usage?.timings?.[id] != null ? ` · ${run.usage.timings[id]}s` : "");
  setAgent("researcher", "done", `Read ${run.sources.length} pages, wrote ${run.claims.length} claims${rounds}${took("researcher")}`);
  setAgent("skeptic", "done", `Challenged ${run.objections.length} claims, ${strong} strong objection${strong === 1 ? "" : "s"}${took("skeptic")}`);
  setAgent("judge", "done", `${kept} of ${run.verdicts.length} claims passed${took("judge")}`);
  setAgent("writer", "done", (kept ? "Answer written from verified claims" : "Not enough evidence to answer") + took("writer"));
}

function setAgent(id, state, text) {
  const el = $(`#agent-${id}`);
  el.classList.remove("active", "waiting", "done");
  el.classList.add(state);
  el.querySelector(".agent-status").textContent = text;
}

// ============================== Rendering a run ==============================

function render() {
  if (!run) return;
  if (run.slug !== render.lastSlug) showAllSources = false;   // new run: start collapsed
  render.lastSlug = run.slug;

  // Show the model(s) that really answered this run; older runs didn't record it.
  const models = run.usage?.models?.length ? run.usage.models : null;
  $("#model-name").textContent = models ? models.join(" + ") : config.model;
  $(".model-chip").title = models ? "Model(s) used for this run" : "Configured model";
  const t = config.threshold;
  const claims = Object.fromEntries(run.claims.map((c) => [c.id, c]));
  const verdicts = [...run.verdicts].sort((a, b) => b.final_confidence - a.final_confidence);
  const kept = verdicts.filter((v) => v.final_confidence >= t);

  // Answer: escape it, then turn [C1] into clickable citation chips.
  $("#question-line").textContent = run.question;
  $("#answer").classList.toggle("empty", !run.answer);
  $("#answer").innerHTML = esc(run.answer || "No answer yet.")
    .replace(/\[(C\d+)\]/g, (_, id) => `<button class="cite" data-claim="${id}">${id}</button>`);

  const download = $("#download");
  download.hidden = !run.slug;
  download.href = `${API}/api/runs/${run.slug}/report.md`;

  const stats = [
    [verdicts.length, "Claims checked"], [kept.length, "Kept"], [verdicts.length - kept.length, "Dropped"],
    [run.sources.length, "Sources read"], [run.usage?.calls ?? "—", "LLM calls"],
    [run.usage?.total_seconds != null ? `${run.usage.total_seconds}s` : "—", "Time taken"],
  ];
  $("#stats").innerHTML = stats.map(([n, label]) => `<div class="stat"><b>${n}</b><span>${label}</span></div>`).join("");

  renderChart(verdicts, claims);
  renderClaims(verdicts, claims);
  renderSources();
}

function renderChart(verdicts, claims) {
  const W = 420, H = 200, top = 10, bottom = 26, right = 36;
  const plotH = H - top - bottom, plotW = W - right;
  const t = config.threshold;
  const y = (v) => top + plotH * (1 - v);
  const step = plotW / Math.max(verdicts.length, 1);
  const barW = Math.min(34, step * 0.62);

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Final confidence for each claim">
    <defs><linearGradient id="kept" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="var(--chart-a)"/><stop offset="1" stop-color="var(--chart-b)"/>
    </linearGradient></defs>`;
  for (const v of [0, 0.5, 1]) {
    svg += `<line class="grid-line" x1="0" x2="${plotW}" y1="${y(v)}" y2="${y(v)}"/>`;
    // Skip a gridline label that would collide with the threshold label.
    if (Math.abs(v - t) >= 0.15) {
      svg += `<text class="axis" x="${W}" y="${y(v) + 3}" text-anchor="end">${v * 100}%</text>`;
    }
  }
  verdicts.forEach((v, i) => {
    const x = i * step + (step - barW) / 2;
    const h = Math.max(plotH * v.final_confidence, 3);
    const keptBar = v.final_confidence >= t;
    // Path = a bar with only the top corners rounded, anchored to the baseline.
    const r = Math.min(4, h);
    svg += `<path class="bar" data-claim="${esc(v.claim_id)}" fill="${keptBar ? "url(#kept)" : "var(--dropped)"}"
      d="M${x},${y(0)} V${y(0) - h + r} q0,-${r} ${r},-${r} H${x + barW - r} q${r},0 ${r},${r} V${y(0)} Z"/>
      <text class="axis" x="${x + barW / 2}" y="${H - 8}" text-anchor="middle">${esc(v.claim_id)}</text>`;
  });
  svg += `<line x1="0" x2="${plotW}" y1="${y(t)}" y2="${y(t)}" stroke="var(--threshold)" stroke-width="1.5" stroke-dasharray="4 4"/>
          <text class="axis" x="${W}" y="${y(t) + 3}" text-anchor="end">${Math.round(t * 100)}%</text></svg>`;
  $("#chart").innerHTML = verdicts.length ? svg : `<p class="empty-note">No claims yet.</p>`;

  const tip = $("#tooltip");
  $("#chart").querySelectorAll(".bar").forEach((bar) => {
    const v = verdicts.find((x) => x.claim_id === bar.dataset.claim);
    bar.addEventListener("mousemove", (e) => {
      tip.hidden = false;
      tip.style.left = `${e.clientX}px`;
      tip.style.top = `${e.clientY}px`;
      tip.innerHTML = `<b>${Math.round(v.final_confidence * 100)}%</b> ${esc(v.claim_id)} · ${v.final_confidence >= t ? "kept" : "dropped"}<br>${esc(claims[v.claim_id].text)}`;
    });
    bar.addEventListener("mouseleave", () => { tip.hidden = true; });
    bar.addEventListener("click", () => openClaim(v.claim_id));
  });
}

function renderClaims(verdicts, claims) {
  const t = config.threshold;
  const filter = $("#claim-filter").value;
  const shown = verdicts.filter((v) =>
    filter === "all" || (filter === "kept") === (v.final_confidence >= t));
  $("#claims").innerHTML = shown.map((v) => {
    const pct = Math.round(v.final_confidence * 100);
    const keptRow = v.final_confidence >= t;
    // A dropped claim gets a neutral "–" icon: the Judge may have said "supported",
    // but the claim did not reach the threshold, so it must not look like a kept one.
    const mark = keptRow
      ? `<span class="row-icon ${esc(v.verdict)}" title="${esc(v.verdict)}">${icon(VERDICT_ICON[v.verdict] || "help")}</span>`
      : `<span class="row-icon dropped" title="Dropped: below the keep threshold">${icon("minus")}</span>`;
    return `<li class="row clickable ${keptRow ? "" : "is-dropped"}" data-claim="${esc(v.claim_id)}" tabindex="0">
      ${mark}
      <div>
        <div class="row-text">${esc(claims[v.claim_id].text)}</div>
        <div class="bar-track">
          <span class="bar-fill ${keptRow ? "" : "dropped"}" style="width:${pct}%"></span>
          <span class="bar-tick" style="left:${t * 100}%"></span>
        </div>
      </div>
      <div class="row-num">${pct}<small>%</small></div>
    </li>`;
  }).join("") || `<li class="empty-note">No claims match this filter.</li>`;
}

function renderSources() {
  // Count how often each source page is cited for and against claims.
  const forCount = {}, againstCount = {};
  run.claims.forEach((c) => c.source_ids.forEach((id) => { forCount[id] = (forCount[id] || 0) + 1; }));
  run.objections.forEach((o) => o.source_ids.forEach((id) => { againstCount[id] = (againstCount[id] || 0) + 1; }));

  // Group pages by website: two pages from the same site are one source, the same rule
  // the confidence score uses ("independent websites").
  const sites = {};
  for (const s of run.sources) {
    const pro = forCount[s.id] || 0, con = againstCount[s.id] || 0;
    if (pro + con === 0) continue;                       // read but never cited
    const domain = domainOf(s.url);
    sites[domain] ??= { domain, pages: [], pro: 0, con: 0 };
    sites[domain].pages.push(s);
    sites[domain].pro += pro;
    sites[domain].con += con;
  }
  const cited = Object.values(sites).sort((a, b) => (b.pro + b.con) - (a.pro + a.con));
  const shown = showAllSources ? cited : cited.slice(0, SOURCES_PREVIEW);

  $("#sources-sub").textContent = `${cited.length} websites cited · ${run.sources.length} pages read`;
  $("#sources").innerHTML = shown.map((site) => {
    const tint = AVATAR_TINTS[[...site.domain].reduce((n, c) => n + c.charCodeAt(0), 0) % AVATAR_TINTS.length];
    const pages = site.pages.length > 1 ? `${site.pages.length} pages · ` : "";
    return `<li class="row">
      <span class="avatar" style="background: var(${tint})" aria-hidden="true">${esc(site.domain[0].toUpperCase())}</span>
      <div>
        <div class="row-text"><a href="${esc(safeUrl(site.pages[0].url))}" target="_blank" rel="noopener noreferrer">${esc(site.domain)}</a></div>
        <div class="row-meta">${pages}${site.pro} supporting · ${site.con} against · ${esc(site.pages[0].title)}</div>
      </div>
      <div class="row-num">${site.pro + site.con}</div>
    </li>`;
  }).join("") || `<li class="empty-note">No sources cited yet.</li>`;

  const toggle = $("#sources-toggle");
  toggle.hidden = cited.length <= SOURCES_PREVIEW;
  toggle.textContent = showAllSources ? "Show less" : `Show all ${cited.length}`;
}

// ============================== Claim detail dialog ==============================

function sourceLinks(ids) {
  const byId = Object.fromEntries(run.sources.map((s) => [s.id, s]));
  return ids.filter((id) => byId[id]).map((id) =>
    `<li><b>${esc(id)}</b><a href="${esc(safeUrl(byId[id].url))}" target="_blank" rel="noopener noreferrer">${esc(byId[id].title || byId[id].url)}</a></li>`
  ).join("");
}

function openClaim(id) {
  const claim = run.claims.find((c) => c.id === id);
  const verdict = run.verdicts.find((v) => v.claim_id === id);
  const objection = run.objections.find((o) => o.claim_id === id);
  if (!claim || !verdict) return;
  const keptClaim = verdict.final_confidence >= config.threshold;

  const dialog = $("#claim-dialog");
  dialog.innerHTML = `
    <div class="dialog-head">
      <div style="display:flex; gap:8px">
        <span class="badge">${esc(id)}</span>
        <span class="badge">${icon(VERDICT_ICON[verdict.verdict] || "help").replace("<svg", '<svg width="14" height="14"')} ${esc(verdict.verdict)}</span>
        <span class="badge">${Math.round(verdict.final_confidence * 100)}% · ${keptClaim ? "kept" : "dropped"}</span>
      </div>
      <button class="ghost-btn" id="close-dialog" aria-label="Close">${icon("x")}</button>
    </div>
    <h2>${esc(claim.text)}</h2>
    <div class="panes">
      <div class="pane" style="--tint: var(--green)">
        <h4>Judge</h4>
        <p>${esc(verdict.reasoning)}</p>
        <p style="margin-top:10px">Judge said ${Math.round(verdict.confidence * 100)}%, calibrated to
          ${Math.round(verdict.final_confidence * 100)}% using source count and the Skeptic's objection.</p>
        <ul class="links">${sourceLinks(claim.source_ids)}</ul>
      </div>
      <div class="pane" style="--tint: var(--blue)">
        <h4>Skeptic · ${esc(objection?.strength ?? "no objection")}</h4>
        <p>${esc(objection?.objection ?? "The Skeptic did not challenge this claim.")}</p>
        <ul class="links">${objection ? sourceLinks(objection.source_ids) : ""}</ul>
      </div>
    </div>`;
  dialog.showModal();
  $("#close-dialog").addEventListener("click", () => dialog.close());
}

// ============================== History ==============================

async function renderHistory() {
  const runs = await (await fetch(`${API}/api/runs`)).json();
  $("#history").innerHTML = runs.map((r) => `
    <button class="run-card" data-slug="${esc(r.slug)}">
      <h3>${esc(r.question)}</h3>
      <div class="bar-track"><span class="bar-fill" style="width:${r.claims ? (r.kept / r.claims) * 100 : 0}%"></span></div>
      <div class="meta">${r.kept} of ${r.claims} claims kept · ${r.sources} sources · ${new Date(r.time * 1000).toLocaleString()}</div>
    </button>`).join("") || `<p class="empty-note">No runs yet. Ask a question on the dashboard.</p>`;
}

async function openRun(slug) {
  run = await (await fetch(`${API}/api/runs/${encodeURIComponent(slug)}`)).json();
  render();
  showAgentSummaries();
  showView("dashboard");
}

// ============================== Running the crew (live) ==============================

function startRun(question) {
  const button = $("#ask-form button");
  button.disabled = true;
  ["#answer-card", "#claims-card", "#sources-card"].forEach((s) => $(s).classList.add("is-busy"));
  AGENTS.forEach((a) => setAgent(a.id, "waiting", "Waiting"));
  let researchRounds = 0;

  // Live timer: once a second, show how long the working agent has been busy.
  const startedAt = {};
  const label = {};
  const ticker = setInterval(() => {
    for (const [agent, t0] of Object.entries(startedAt)) {
      setAgent(agent, "active", `${label[agent]} ${Math.round((Date.now() - t0) / 1000)}s`);
    }
  }, 1000);

  const finish = () => {
    clearInterval(ticker);
    stream.close();   // important: otherwise the browser auto-reconnects and re-runs the crew
    button.disabled = false;
    ["#answer-card", "#claims-card", "#sources-card"].forEach((s) => $(s).classList.remove("is-busy"));
  };

  // EventSource = the browser side of Server-Sent Events: the server pushes, we listen.
  const stream = new EventSource(`${API}/api/research?q=${encodeURIComponent(question)}`);
  stream.addEventListener("start", (e) => {
    const { agent } = JSON.parse(e.data);
    if (agent === "researcher") researchRounds += 1;
    label[agent] = researchRounds > 1 && agent === "researcher" ? `Round ${researchRounds}: working…` : "Working…";
    startedAt[agent] = Date.now();
    setAgent(agent, "active", `${label[agent]} 0s`);
  });
  stream.addEventListener("finish", (e) => {
    const { agent, text, seconds } = JSON.parse(e.data);
    delete startedAt[agent];
    setAgent(agent, "done", seconds != null ? `${text} · ${seconds}s` : text);
  });
  stream.addEventListener("done", (e) => {
    run = JSON.parse(e.data);
    finish();
    render();
    showAgentSummaries();   // whole-run totals (live messages only covered the last round)
  });
  stream.addEventListener("failed", (e) => {
    finish();
    toast(JSON.parse(e.data).message);
  });
  stream.onerror = () => { finish(); };
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  setTimeout(() => { el.hidden = true; }, 6000);
}

// ============================== Start-up ==============================

document.querySelectorAll("[data-icon]").forEach((el) => { el.innerHTML = icon(el.dataset.icon); });
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
document.querySelectorAll("[data-theme-set]").forEach((b) => b.addEventListener("click", () => setTheme(b.dataset.themeSet)));

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q) startRun(q);
});
$("#claim-filter").addEventListener("change", render);

// Example chips fill the input; the user still presses the arrow (each run uses free quota).
$("#examples").innerHTML = `<span class="examples-label">Try</span>` + EXAMPLES.map((q) =>
  `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("#examples").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  $("#question").value = chip.textContent;
  $("#question").focus();
});
$("#sources-toggle").addEventListener("click", () => { showAllSources = !showAllSources; renderSources(); });

// Clicks on claim rows, citation chips and history cards (event delegation: one listener for many items).
document.addEventListener("click", (e) => {
  const claimEl = e.target.closest("[data-claim]:not(.bar)");
  if (claimEl) openClaim(claimEl.dataset.claim);
  const runEl = e.target.closest(".run-card");
  if (runEl) openRun(runEl.dataset.slug);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches(".row[data-claim]")) openClaim(e.target.dataset.claim);
});

$("#about").innerHTML = AGENTS.map((a, i) => `
  <div class="agent" style="--tint: var(${a.tint})">
    <h3>${i + 1}. ${a.name}</h3><p>${a.desc}</p>
  </div>`).join("") + `
  <div class="note">
    <strong>Why trust the scores?</strong> The Judge's own confidence is mixed with signals we count ourselves:
    <code>final = 0.6 × judge + 0.4 × independent websites − skeptic penalty</code>.
    Claims below the threshold are dropped, and the Writer never sees them.
    If too few claims survive, the Researcher goes back for another round.
  </div>`;

// Light by default; dark only if this visitor picked it with the moon button before.
try {
  setTheme(localStorage.getItem("theme") === "dark" ? "dark" : "light");
} catch { setTheme("light"); }
renderAgents();

(async () => {
  // Free hosting puts the backend to sleep when idle; the first request can take ~1 minute.
  $("#model-name").textContent = "Connecting…";
  const slowHint = setTimeout(() => {
    $("#model-name").textContent = "Waking up the server (free hosting, up to a minute)…";
  }, 4000);
  try {
    config = await (await fetch(`${API}/api/config`)).json();
    $("#model-name").textContent = config.model;
    const runs = await (await fetch(`${API}/api/runs`)).json();
    if (runs.length) await openRun(runs[0].slug);
  } catch {
    $("#model-name").textContent = "Server offline";
    toast("Can't reach the Verdict server. Check that the backend is running.");
  } finally {
    clearTimeout(slowHint);
  }
})();

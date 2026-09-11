/* ===========================================================================
   TrustGraph — frontend
   ---------------------------------------------------------------------------
   The grid is the hero (§7.1). Everything else exists to make the grid
   readable: the composition bar summarises it, the head-to-head bars slice it,
   and the drawer proves it by showing the response text behind any single cell.

   Cells are rendered as soon as their result lands on the stream, so the
   diagnostic pattern forms while the remaining calls are still in flight.
   =========================================================================== */

"use strict";

const STATES = {
  win: { label: "WIN", aria: "Win — target recommended, no competitor recommended" },
  loss: { label: "LOSS", aria: "Loss — a competitor recommended instead of the target" },
  both: { label: "BOTH", aria: "Both — target and a competitor both recommended" },
  neither: {
    label: "—",
    aria: "Neither — no tracked vendor was recommended; the model answered off-axis",
  },
};

const state = {
  spec: null,
  models: [],
  paraphrases: [],
  cells: new Map(), // "queryIndex|modelKey" -> cell
  scores: null,
  running: false,
  selected: null,
};

const $ = (id) => document.getElementById(id);
const key = (q, m) => `${q}|${m}`;

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const pct = (v, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : `${(v * 100).toFixed(digits)}%`;

/* =========================================================================
   Boot
   ========================================================================= */

document.addEventListener("DOMContentLoaded", () => {
  $("run-form").addEventListener("submit", onSubmit);
  $("group-by-intent").addEventListener("change", renderMatrix);
  $("drawer-close").addEventListener("click", closeDrawer);
  $("scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", onKeydown);

  ["count", "competitors"].forEach((id) =>
    $(id).addEventListener("input", updateEstimate)
  );
  $("recheck").addEventListener("change", updateEstimate);

  renderLegend();
  loadHealth();
});

function onKeydown(event) {
  if (event.key === "Escape") closeDrawer();
}

/* =========================================================================
   Health / provenance
   ========================================================================= */

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    renderRosterChips(data);
    renderNotices(data.warnings || [], data.roster || []);
    $("count").value = data.settings?.paraphrase_count ?? 18;
    $("recheck").checked = Boolean(data.settings?.llm_recheck);
    state.healthRoster = data.roster || [];
    updateEstimate();
  } catch (err) {
    renderNotices([`Could not reach the API: ${err.message}`], []);
  }
}

function renderRosterChips(data) {
  const host = $("roster-chips");
  const providers = data.providers || {};
  const rows = Object.entries(providers).map(([name, info]) => {
    const cls = info.configured ? "chip" : "chip off";
    const mark = info.configured ? "key present" : "no key";
    return `<span class="${cls}">${escapeHtml(info.label || name)} · ${mark}</span>`;
  });
  const columns = (data.roster || []).map((m) => m.label).join(" · ");
  rows.push(
    `<span class="muted">columns: ${escapeHtml(columns || "none")}</span>`
  );
  host.innerHTML = rows.join("");
}

function renderNotices(warnings, roster) {
  const host = $("notices");
  const items = [];

  const simulated = roster.length > 0 && roster.every((m) => m.provider === "simulated");
  if (simulated) {
    items.push({
      kind: "sim",
      tag: "Fixtures",
      text:
        "No provider keys are configured, so this run uses the deterministic " +
        "simulated provider. Every number below is a reproducible fixture, not a " +
        "measurement of live model behaviour. Add a key to .env to measure for real.",
    });
  }

  const families = new Set(roster.map((m) => m.family));
  if (!simulated && roster.length > 1 && families.size === 1) {
    items.push({
      kind: "warn",
      tag: "Within-family",
      text:
        `All ${roster.length} columns are ${[...families][0]}-family models. ` +
        "Per-model Trust and RSI are valid, but the consensus figures measure " +
        "within-family agreement and do not support the cross-family AITC claim " +
        "in §3.5 — an entity favoured by one vendor's models is a weaker fact " +
        "than one favoured across vendors.",
    });
  }

  // The backend emits its own simulated/within-family warnings. When a banner
  // above already carries that message, showing the raw warning too just says
  // the same thing twice.
  const covered = items.map((i) => i.kind);
  warnings.forEach((w) => {
    const lower = w.toLowerCase();
    if (covered.includes("sim") && lower.includes("simulated provider")) return;
    if (covered.includes("warn") && lower.includes("within-family")) return;
    items.push({ kind: "info", tag: "Note", text: w });
  });

  host.innerHTML = items
    .map(
      (i) =>
        `<div class="notice ${i.kind}"><b>${escapeHtml(i.tag)}</b><span>${escapeHtml(
          i.text
        )}</span></div>`
    )
    .join("");
}

function updateEstimate() {
  const count = Number($("count").value) || 0;
  const columns = (state.models.length || state.healthRoster?.length || 0) || 0;
  const perCell = $("recheck").checked ? 2 : 1;
  const total = count * columns * perCell + 1;
  $("call-estimate").textContent = columns
    ? `≈ ${total} provider calls (${count} × ${columns} columns${
        perCell === 2 ? " × 2 passes" : ""
      } + 1 paraphrase)`
    : "";
}

/* =========================================================================
   Run
   ========================================================================= */

async function onSubmit(event) {
  event.preventDefault();
  if (state.running) return;

  const competitors = $("competitors")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 5);

  const body = {
    entity: $("entity").value.trim(),
    category: $("category").value.trim(),
    competitors,
    paraphrase_count: Number($("count").value) || null,
    llm_recheck: $("recheck").checked,
    fresh: $("fresh").checked,
  };

  if (!body.entity || !body.category) return;

  setRunning(true);
  state.cells.clear();
  state.scores = null;
  state.selected = null;
  closeDrawer();

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new Error(`server returned ${res.status}`);
    }
    await consume(res.body);
  } catch (err) {
    renderNotices([`Run failed: ${err.message}`], state.models);
  } finally {
    setRunning(false);
  }
}

function setRunning(running) {
  state.running = running;
  $("run-btn").disabled = running;
  $("run-btn").textContent = running ? "Measuring…" : "Run measurement";
}

/** Read an SSE stream off a fetch body. */
async function consume(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue; // ignore SSE comments
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          handleEvent(JSON.parse(payload));
        } catch {
          /* a truncated frame is not worth killing the run over */
        }
      }
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "run_started":
      state.spec = event.spec;
      state.models = event.models || [];
      state.paraphrases = [];
      $("readout-panel").classList.remove("hidden");
      $("headline").textContent = `Measuring ${event.spec.entity}…`;
      setProgress(0, event.planned_calls || 0, "queued");
      renderNotices(event.meta?.warnings || [], state.models);
      updateEstimate();
      break;

    case "paraphrases":
      state.paraphrases = event.paraphrases || [];
      renderNotices(event.warnings || [], state.models);
      $("grid-panel").classList.remove("hidden");
      renderMatrix();
      break;

    case "cell": {
      const cell = event.cell;
      state.cells.set(key(cell.query_index, cell.model_key), cell);
      paintCell(cell);
      setProgress(event.done, event.total, "responses");
      break;
    }

    case "scores":
      state.scores = event.scores;
      renderScores();
      break;

    case "run_complete":
      finishRun(event.result);
      break;

    case "error":
      renderNotices([event.message], state.models);
      break;

    default:
      break;
  }
}

function setProgress(done, total, unit) {
  const frac = total ? done / total : 0;
  $("progress-fill").style.width = `${Math.round(frac * 100)}%`;
  $("progress-bar").setAttribute("aria-valuenow", String(Math.round(frac * 100)));
  $("progress-label").textContent = total ? `${done} / ${total} ${unit}` : "";
}

function finishRun(result) {
  const meta = result.meta || {};
  const bits = [];
  if (meta.duration_ms) bits.push(`${(meta.duration_ms / 1000).toFixed(1)}s`);
  if (meta.provider_calls) bits.push(`${meta.provider_calls} live calls`);
  if (meta.cached_calls) bits.push(`${meta.cached_calls} from cache`);
  if (meta.failed_calls) bits.push(`${meta.failed_calls} failed`);
  $("progress-label").textContent = bits.join(" · ");
  $("progress-fill").style.width = "100%";
  renderNotices(meta.warnings || [], result.models || state.models);
}

/* =========================================================================
   Legend
   ========================================================================= */

function renderLegend() {
  const meanings = {
    win: "Target recommended; no competitor recommended.",
    loss: "A competitor recommended instead of the target.",
    both: "Target and at least one competitor both recommended.",
    neither: "No tracked vendor recommended — the model answered off-axis.",
  };
  $("legend").innerHTML = Object.entries(STATES)
    .map(
      ([code, def]) => `
      <span class="item">
        <span class="key s-${code}" aria-hidden="true"
              style="background:var(--${code});${
                code === "neither" ? "border-color:#2f403c;color:var(--ink-3)" : ""
              }">${def.label}</span>
        <span>${escapeHtml(meanings[code])}</span>
      </span>`
    )
    .join("");
  // The keys reuse the cell fills, so the swatch colours stay in lockstep
  // with the grid by construction rather than by a duplicated hex value.
  $("legend")
    .querySelectorAll(".key")
    .forEach((el) => {
      const code = [...el.classList].find((c) => c.startsWith("s-"))?.slice(2);
      if (code === "win") el.style.color = "#1a1206";
      if (code === "loss") el.style.color = "#1e0c12";
      if (code === "both") el.style.color = "#06201d";
    });
}

/* =========================================================================
   The grid
   ========================================================================= */

function orderedRows() {
  const rows = state.paraphrases.slice();
  if (!$("group-by-intent").checked) return [{ label: null, rows }];

  // Group by framing, groups in order of first appearance. A framing-contiguous
  // grid is what turns §7.1's example — a band of losses across every
  // price-framed phrasing — from a scatter into something you can see.
  const groups = new Map();
  for (const row of rows) {
    const label = row.intent || "unlabelled";
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(row);
  }
  return [...groups.entries()].map(([label, rows]) => ({ label, rows }));
}

function renderMatrix() {
  if (!state.paraphrases.length) return;

  const table = $("matrix");
  const groups = orderedRows();

  const head = `
    <thead>
      <tr>
        <th scope="col">Phrasing (${state.paraphrases.length})</th>
        ${state.models
          .map(
            (m) => `<th scope="col" class="col-model">${escapeHtml(m.label)}
              <span class="fam">${escapeHtml(m.provider)}</span></th>`
          )
          .join("")}
      </tr>
    </thead>`;

  const body = groups
    .map((group) => {
      const header =
        group.label === null
          ? ""
          : `<tr class="group-head"><th scope="rowgroup" colspan="${
              state.models.length + 1
            }"><span class="glabel">${escapeHtml(group.label)} · ${
              group.rows.length
            } phrasing${group.rows.length === 1 ? "" : "s"}</span></th></tr>`;

      const rows = group.rows
        .map(
          (para) => `
        <tr data-q="${para.index}">
          <th scope="row">
            <span class="qmeta">
              <span class="qnum">${String(para.index + 1).padStart(2, "0")}</span>
              ${
                $("group-by-intent").checked || !para.intent
                  ? ""
                  : `<span class="intent">${escapeHtml(para.intent)}</span>`
              }
              <span class="qtext" title="${escapeHtml(para.text)}">${escapeHtml(
            para.text
          )}</span>
            </span>
          </th>
          ${state.models
            .map(
              (m) => `<td>${cellButton(para.index, m)}</td>`
            )
            .join("")}
        </tr>`
        )
        .join("");

      return header + rows;
    })
    .join("");

  // The caption stays short on purpose: on an auto-width table a long caption
  // contributes to the table's minimum width and would stretch the matrix. The
  // full explanation lives in the visible paragraph above, which serves sighted
  // and assistive-tech readers alike.
  table.innerHTML =
    `<caption id="matrix-caption">Outcome grid: phrasing (rows) &times; model (columns).</caption>` +
    head +
    `<tbody>${body}</tbody>` +
    footRow();

  table.querySelectorAll("button.cell").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cell = state.cells.get(key(Number(btn.dataset.q), btn.dataset.model));
      if (cell) openDrawer(cell);
    });
  });

  // Repaint anything that already arrived (matters when the grouping toggle
  // re-renders mid-run).
  state.cells.forEach((cell) => paintCell(cell));
  if (state.scores) renderColumnStats();
}

function cellButton(queryIndex, model) {
  const cell = state.cells.get(key(queryIndex, model.key));
  if (!cell) {
    return `<button class="cell s-pending" data-q="${queryIndex}" data-model="${escapeHtml(
      model.key
    )}" disabled aria-label="pending">·</button>`;
  }
  return renderCellButton(cell, model);
}

function renderCellButton(cell, model) {
  const def = STATES[cell.state] || STATES.neither;
  const cls = cell.error ? "s-error" : `s-${cell.state}`;
  const label = cell.error ? "ERR" : def.label;
  const aside =
    !cell.error &&
    (cell.target_mentioned_only || (cell.competitors_mentioned_only || []).length);
  return `<button class="cell ${cls}${aside ? " has-aside" : ""}"
      data-q="${cell.query_index}" data-model="${escapeHtml(cell.model_key)}"
      aria-label="${escapeHtml(
        `${model?.label || cell.model_key}: ${cell.error ? "call failed" : def.aria}`
      )}"
      title="${escapeHtml(cell.error || def.aria)}">
      <span class="glyph">${label}</span></button>`;
}

function paintCell(cell) {
  const selector = `button.cell[data-q="${cell.query_index}"][data-model="${cell.model_key.replace(
    /"/g,
    '\\"'
  )}"]`;
  const existing = $("matrix").querySelector(selector);
  if (!existing) return;
  const model = state.models.find((m) => m.key === cell.model_key);
  const wrapper = document.createElement("template");
  wrapper.innerHTML = renderCellButton(cell, model).trim();
  const next = wrapper.content.firstElementChild;
  next.addEventListener("click", () => openDrawer(cell));
  if (state.selected === key(cell.query_index, cell.model_key)) {
    next.classList.add("selected");
  }
  existing.replaceWith(next);
}

function footRow() {
  return `<tfoot><tr id="matrix-foot">
    <th scope="row"><span class="mono muted">Trust · RSI</span></th>
    ${state.models.map(() => `<td class="colstat"></td>`).join("")}
  </tr></tfoot>`;
}

function renderColumnStats() {
  const foot = $("matrix").querySelector("#matrix-foot");
  if (!foot || !state.scores) return;
  const cells = foot.querySelectorAll("td.colstat");
  state.models.forEach((model, i) => {
    const score = state.scores.per_model.find((s) => s.model_key === model.key);
    const td = cells[i];
    if (!td || !score) return;
    td.innerHTML = `
      <div class="t">${pct(score.trust)}</div>
      <div class="r">RSI ${score.rsi.toFixed(2)}${
      score.errors ? ` · ${score.errors} err` : ""
    }</div>
      <div class="badge ${score.stability}" title="${escapeHtml(
      score.stability_note
    )}">${score.stability}</div>`;
  });
}

/* =========================================================================
   Scores
   ========================================================================= */

function renderScores() {
  const s = state.scores;
  if (!s) return;

  $("headline").innerHTML = highlightHeadline(s.headline);
  $("fig-aitc").textContent = pct(s.aitc);
  $("fig-consensus").textContent = pct(s.consensus.unanimous_win_frac);

  renderComposition(s.consensus, s.cross_model);
  renderColumnStats();
  renderHeadToHead(s);

  $("composition-panel").classList.remove("hidden");
  $("h2h-panel").classList.remove("hidden");
  $("method-panel").classList.remove("hidden");
}

function highlightHeadline(text) {
  return escapeHtml(text).replace(/(\d+%)/g, '<span class="k">$1</span>');
}

function renderComposition(consensus, cross) {
  const segments = [
    { cls: "seg-win", n: consensus.unanimous_win, frac: consensus.unanimous_win_frac, name: "unanimous win" },
    { cls: "seg-split", n: consensus.split, frac: consensus.split_frac, name: "split" },
    { cls: "seg-loss", n: consensus.unanimous_loss, frac: consensus.unanimous_loss_frac, name: "unanimous loss" },
  ];

  $("composition").innerHTML = segments
    .map(
      (seg) =>
        `<span class="${seg.cls}${seg.n === 0 ? " empty" : ""}" style="flex-grow:${
          seg.n
        }" title="${escapeHtml(`${seg.name}: ${seg.n} phrasings`)}">${
          seg.frac >= 0.12 ? `${Math.round(seg.frac * 100)}%` : ""
        }</span>`
    )
    .join("");

  $("composition-key").innerHTML = segments
    .map(
      (seg) => `<span><span class="sw" style="background:var(--${
        seg.cls === "seg-win" ? "win" : seg.cls === "seg-split" ? "both" : "loss"
      })"></span>${escapeHtml(seg.name)} — <span class="mono">${seg.n}</span> of <span class="mono">${
        consensus.n
      }</span> (${pct(seg.frac)})</span>`
    )
    .join("");

  const r =
    cross.coefficient === null || cross.coefficient === undefined
      ? "undefined"
      : cross.coefficient.toFixed(2);
  $("composition-note").textContent =
    `Mean pairwise correlation of per-phrasing outcomes across models: r = ${r}. ` +
    cross.note +
    " A split segment is the interesting one: it marks phrasings where models " +
    "disagree, which an averaged score would erase.";
}

function renderHeadToHead(scores) {
  const host = $("h2h");
  const competitors = state.spec?.competitors || [];
  if (!competitors.length) {
    host.innerHTML = `<div class="h2h-none">No competitors named — head-to-head needs at least one.</div>`;
    return;
  }

  host.innerHTML = competitors
    .map((competitor) => {
      const total = scores.head_to_head.find((h) => h.competitor === competitor);
      const perModel = state.models
        .map((model) => {
          const score = scores.per_model.find((s) => s.model_key === model.key);
          const record = score?.head_to_head.find((h) => h.competitor === competitor);
          return { model, record };
        })
        .filter((x) => x.record);

      const aggregate =
        total && total.eligible
          ? `${pct(total.win_rate)} across ${total.eligible} contested phrasing${
              total.eligible === 1 ? "" : "s"
            }`
          : "no contested phrasings";

      return `
      <div class="h2h-block">
        <div class="h2h-title">
          <span class="vs">${escapeHtml(state.spec.entity)} vs</span>
          <span class="name">${escapeHtml(competitor)}</span>
          <span class="agg">${escapeHtml(aggregate)}</span>
        </div>
        ${perModel.map((x) => h2hRow(x.model, x.record)).join("")}
      </div>`;
    })
    .join("");
}

function h2hRow(model, record) {
  if (!record.eligible) {
    return `<div class="h2h-row">
      <span class="who">${escapeHtml(model.label)}</span>
      <span class="h2h-none">neither was recommended on any phrasing</span>
      <span class="tally"></span>
    </div>`;
  }
  const w = record.wins;
  const t = record.ties;
  const l = record.losses;
  return `<div class="h2h-row">
    <span class="who">${escapeHtml(model.label)}</span>
    <span class="h2hbar" role="img" aria-label="${escapeHtml(
      `${model.label}: ${w} wins, ${t} ties, ${l} losses of ${record.eligible} contested phrasings`
    )}">
      <span class="b-win${w ? "" : " empty"}" style="flex-grow:${w}">${
    w / record.eligible >= 0.16 ? pct(w / record.eligible) : ""
  }</span>
      <span class="b-tie${t ? "" : " empty"}" style="flex-grow:${t}">${
    t / record.eligible >= 0.16 ? pct(t / record.eligible) : ""
  }</span>
      <span class="b-loss${l ? "" : " empty"}" style="flex-grow:${l}">${
    l / record.eligible >= 0.16 ? pct(l / record.eligible) : ""
  }</span>
    </span>
    <span class="tally">W${w} · T${t} · L${l}</span>
  </div>`;
}

/* =========================================================================
   Evidence drawer (§7.2)
   ========================================================================= */

function openDrawer(cell) {
  const model = state.models.find((m) => m.key === cell.model_key);
  const para = state.paraphrases.find((p) => p.index === cell.query_index);
  const def = STATES[cell.state] || STATES.neither;

  state.selected = key(cell.query_index, cell.model_key);
  $("matrix")
    .querySelectorAll("button.cell.selected")
    .forEach((el) => el.classList.remove("selected"));
  const active = $("matrix").querySelector(
    `button.cell[data-q="${cell.query_index}"][data-model="${cell.model_key}"]`
  );
  if (active) active.classList.add("selected");

  const chip = $("drawer-state");
  chip.textContent = cell.error ? "CALL FAILED" : def.label;
  chip.style.background = cell.error ? "transparent" : `var(--${cell.state})`;
  chip.style.color = cell.error
    ? "var(--loss)"
    : cell.state === "neither"
    ? "var(--ink-2)"
    : "#101a18";
  chip.style.border = cell.error ? "1px solid var(--loss)" : "1px solid transparent";

  $("drawer-model").textContent = model ? `${model.label} · ${model.model}` : cell.model_key;
  $("drawer-title").textContent = para ? para.text : `phrasing #${cell.query_index + 1}`;

  const sub = [];
  if (para?.intent) sub.push(`framing: ${para.intent}`);
  if (cell.latency_ms) sub.push(`${cell.latency_ms} ms`);
  if (cell.cached) sub.push("served from cache");
  sub.push(def.aria);
  $("drawer-sub").textContent = sub.join(" · ");

  renderVerdicts(cell);

  if (cell.error) {
    $("drawer-response").innerHTML = `<span class="muted">${escapeHtml(
      cell.error
    )}</span>`;
  } else {
    $("drawer-response").innerHTML = highlightResponse(cell);
  }

  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");
  $("drawer-close").focus();
}

function renderVerdicts(cell) {
  const findings = [cell.target, ...(cell.competitors || [])];
  const targetName = cell.target?.name;

  $("drawer-verdicts").innerHTML = findings
    .map((finding) => {
      const isTarget = finding.name === targetName;
      const provenance = provenanceLabel(finding);
      const quote =
        finding.evidence && finding.verdict !== "absent"
          ? `<div class="evidence-quote">${escapeHtml(finding.evidence)}</div>`
          : "";
      return `
      <div>
        <div class="verdict-row">
          <span class="nm${isTarget ? " is-target" : ""}">${escapeHtml(
        finding.name
      )}${isTarget ? " · target" : ""}</span>
          <span class="vd v-${finding.verdict}">${escapeHtml(finding.verdict)}</span>
          <span class="prov${provenance.disagree ? " disagree" : ""}">${escapeHtml(
        provenance.text
      )}</span>
        </div>
        ${quote}
      </div>`;
    })
    .join("");
}

function provenanceLabel(finding) {
  const det = finding.deterministic_verdict;
  const llm = finding.llm_verdict;
  if (finding.agreed === true) return { text: "both passes agree", disagree: false };
  if (finding.agreed === false) {
    return {
      text: `disagreed · rules:${short(det)} model:${short(llm)}`,
      disagree: true,
    };
  }
  if (llm === null || llm === undefined) {
    return { text: "rules only", disagree: false };
  }
  return { text: "—", disagree: false };
}

const short = (v) => (v ? String(v).replace("Verdict.", "").slice(0, 3) : "—");

/**
 * Rebuild the response with every matched name wrapped in a <mark>.
 * Uses the byte offsets the matcher actually recorded, so the drawer shows the
 * literal spans the classification was computed from — not a re-search that
 * might highlight something different from what was scored.
 */
function highlightResponse(cell) {
  const text = cell.response_text || "";
  const targetName = cell.target?.name;
  const spans = [];

  [cell.target, ...(cell.competitors || [])].forEach((finding) => {
    (finding.matches || []).forEach((m) => {
      spans.push({
        start: m.start,
        end: m.end,
        target: finding.name === targetName,
        via: m.via,
      });
    });
  });

  spans.sort((a, b) => a.start - b.start);

  let out = "";
  let cursor = 0;
  for (const span of spans) {
    if (span.start < cursor) continue; // defensive: never emit overlapping marks
    out += escapeHtml(text.slice(cursor, span.start));
    out += `<mark class="hit${span.target ? " target" : ""}" title="matched via ${escapeHtml(
      span.via
    )}">${escapeHtml(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  }
  out += escapeHtml(text.slice(cursor));
  return out || '<span class="muted">empty response</span>';
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
}

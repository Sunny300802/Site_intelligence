/* web/static/js/app.js
   Plain JavaScript - no framework, no build step.
   Polls the API and paints the dashboard; attaches the MJPEG streams. */

const $ = (id) => document.getElementById(id);
const BODY = document.body;
const STREAM_PORT = BODY.dataset.streamPort || "8001";
const STREAM_BASE = `http://${location.hostname}:${STREAM_PORT}`;

let knownNames = {};      // visit id -> name, to detect late identification

/* ---------------- helpers ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function localTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso + "Z");           // API sends UTC
  return d.toLocaleString([], {
    month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
}

function timeOnly(iso) {
  if (!iso) return "—";
  return new Date(iso + "Z").toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
}

function dur(sec) {
  sec = Math.round(sec || 0);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

async function getJSON(url) {
  const r = await fetch(url, { credentials: "same-origin" });
  if (r.status === 401) { window.location = "/login"; return null; }
  if (!r.ok) return null;
  return r.json();
}

function filters() {
  const cam = $("cameraFilter").value;
  const hours = $("windowFilter").value;
  return { cam, hours, q: cam ? `&camera=${encodeURIComponent(cam)}` : "" };
}

/* ---------------- live video ---------------- */
function setupCameras() {
  document.querySelectorAll(".cam-img").forEach(img => {
    const tile = img.closest(".cam-tile");
    const offline = tile.querySelector(".cam-offline");
    img.addEventListener("error", () => {
      img.style.visibility = "hidden";
      offline.classList.remove("d-none");
    });
    img.addEventListener("load", () => {
      img.style.visibility = "visible";
      offline.classList.add("d-none");
    });
    img.src = `${STREAM_BASE}/video/${img.dataset.key}`;
  });
}

async function checkStream() {
  const badge = $("liveBadge");
  try {
    const r = await fetch(`${STREAM_BASE}/cameras`, { cache: "no-store" });
    if (!r.ok) throw new Error();
    badge.className = "badge rounded-pill live-badge on";
    badge.innerHTML = '<i class="bi bi-broadcast"></i> pipeline live';
  } catch {
    badge.className = "badge rounded-pill live-badge off";
    badge.innerHTML = '<i class="bi bi-broadcast-pin"></i> pipeline offline';
  }
}

/* ---------------- KPIs ---------------- */
async function refreshSummary() {
  const f = filters();
  const s = await getJSON(`/api/summary?hours=${f.hours}${f.q}`);
  if (!s) return;
  $("kpiPresent").textContent = s.present_now;
  $("kpiTotal").textContent = s.total_entries;
  $("kpiIdentified").textContent = s.identified;
  $("kpiUnknown").textContent = s.unidentified;
  const label = { "1": "last 1 hour", "8": "last 8 hours",
                  "24": "last 24 hours", "168": "last 7 days" }[f.hours];
  $("kpiWindow").textContent = label || `last ${f.hours}h`;
}

/* ---------------- currently in view ---------------- */
async function refreshPresent() {
  const f = filters();
  const rows = await getJSON(`/api/present?${f.q.slice(1)}`);
  if (!rows) return;
  $("presentCount").textContent = rows.length;
  const body = $("presentBody");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="4" class="text-secondary p-3">Nobody in view.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map(r => `
    <tr>
      <td>${r.identified
            ? `<span class="pill pill-known">${esc(r.person)}</span>`
            : `<span class="pill pill-unknown">Unknown #${r.track_id}</span>`}</td>
      <td class="text-secondary">${esc(r.camera)}</td>
      <td class="mono">${timeOnly(r.entered_at)}</td>
      <td class="text-end mono">${dur(r.duration_seconds)}</td>
    </tr>`).join("");
}

/* ---------------- entry log ---------------- */
async function refreshVisits() {
  const f = filters();
  const rows = await getJSON(`/api/visits?limit=150${f.q}`);
  if (!rows) return;
  const body = $("visitsBody");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="6" class="text-secondary p-3">No entries recorded yet.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map(r => {
    // highlight rows whose name arrived after the entry (retroactive fill)
    const wasUnknown = knownNames[r.id] === "Unknown";
    const justNamed = wasUnknown && r.identified;
    knownNames[r.id] = r.identified ? r.person : "Unknown";

    const person = r.identified
      ? `<span class="pill pill-known"><i class="bi bi-person-check me-1"></i>${esc(r.person)}</span>`
      : `<span class="pill pill-unknown">Unknown #${r.track_id}</span>`;
    const status = r.active
      ? `<span class="pill pill-live">in view</span>`
      : `<span class="text-secondary small">left</span>`;
    return `
      <tr class="${justNamed ? "flash-named" : ""}">
        <td class="mono">${localTime(r.entered_at)}</td>
        <td>${person}</td>
        <td class="text-secondary mono">${esc(r.emp_code || "—")}</td>
        <td class="text-secondary">${esc(r.camera)}</td>
        <td>${status}</td>
        <td class="text-end mono">${dur(r.duration_seconds)}</td>
      </tr>`;
  }).join("");
}

/* ---------------- workspace presence ---------------- */
async function refreshPresence() {
  if (!$("presenceBody")) return;               // no workspace camera
  const f = filters();

  const totals = await getJSON(`/api/presence_totals?hours=${f.hours}${f.q}`);
  if (totals) {
    const body = $("presenceTotalsBody");
    body.innerHTML = totals.length ? totals.map(r => `
      <tr>
        <td>${r.person === "Unknown"
              ? '<span class="pill pill-unknown">Unknown</span>'
              : `<span class="pill pill-known">${esc(r.person)}</span>`}</td>
        <td class="text-secondary mono">${esc(r.emp_code || "—")}</td>
        <td class="text-end mono">${dur(r.seconds)}</td>
        <td class="text-end text-secondary mono">${r.sessions}</td>
      </tr>`).join("")
      : `<tr><td colspan="4" class="text-secondary p-3">No presence recorded yet.</td></tr>`;
  }

  const rows = await getJSON(`/api/presence?hours=${f.hours}${f.q}`);
  if (rows) {
    const body = $("presenceBody");
    body.innerHTML = rows.length ? rows.map(r => `
      <tr>
        <td>${r.identified
              ? `<span class="pill pill-known">${esc(r.person)}</span>`
              : `<span class="pill pill-unknown">Unknown #${r.track_id}</span>`}</td>
        <td class="text-secondary">${esc(r.camera)}</td>
        <td class="mono">${timeOnly(r.first_seen)}</td>
        <td class="mono">${timeOnly(r.last_seen)}</td>
        <td class="text-end mono">${dur(r.seconds)}</td>
        <td>${r.active ? '<span class="pill pill-live">present</span>'
                       : '<span class="text-secondary small">left</span>'}</td>
      </tr>`).join("")
      : `<tr><td colspan="6" class="text-secondary p-3">No sessions yet.</td></tr>`;
  }
}

/* ---------------- review queue badge ---------------- */
async function refreshReviewBadge() {
  const badge = $("reviewBadge");
  if (!badge) return;
  const data = await getJSON("/api/reviews?limit=1");
  if (!data) return;
  const n = (data.counts && data.counts.pending) || 0;
  badge.textContent = n;
  badge.classList.toggle("d-none", n === 0);
}

/* ---------------- loop ---------------- */
function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}

async function refreshAll() {
  await Promise.all([refreshSummary(), refreshPresent(), refreshVisits(),
                     refreshPresence(), refreshReviewBadge()]);
}

$("cameraFilter").addEventListener("change", refreshAll);
$("windowFilter").addEventListener("change", refreshAll);

setupCameras();
tickClock();
refreshAll();
checkStream();
setInterval(tickClock, 1000);
setInterval(refreshAll, 3000);
setInterval(checkStream, 10000);

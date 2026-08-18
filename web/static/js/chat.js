/* web/static/js/chat.js
   Ask the system questions in plain English.

   Three things this file is careful about, because earlier versions got
   them wrong:

     ONE TABLE, NOT TWO. The answer text used to carry a rendered table
     AND the rows were drawn again underneath it. The text is now a
     sentence; the table is drawn here, once.

     A RESULT IS NOT A DATABASE DUMP. A generated query returns whatever
     columns it selected - internal ids, 16-decimal similarity scores,
     raw UTC timestamps. Shown as-is that is unreadable, so noisy
     columns are hidden behind a toggle and values are formatted. The
     full result is always one click away, and the query itself is
     always shown, so nothing is hidden - it is just not shouted.

     THE CHIPS ARE NOT A MENU. They used to be a dozen fixed sentences
     printed above the conversation, which pushed the answers down the
     page and stayed just as irrelevant after the tenth question as the
     first. Now the server sends three or four with every answer, chosen
     from what that answer was about, and they live directly above the
     input where a hand already is. */

const log = document.getElementById("log");
const scroller = document.getElementById("scroll");
const welcome = document.getElementById("welcome");
const form = document.getElementById("askForm");
const box = document.getElementById("q");
const send = document.getElementById("send");
const chipbar = document.getElementById("chips");

/* Columns that are meaningful to the system and noise to a person.
   Hidden by default, revealed by "show all columns". */
const NOISY = new Set([
  "id", "track_id", "camera_key", "emp_code", "identified",
  "identified_at", "local_identified_at", "face_dims", "face_width",
  "dims", "probability", "margin",
]);

/* The chips the page was opened with, so "start again" can put them
   back rather than leaving whatever the last answer suggested. */
const OPENING_CHIPS = [...chipbar.querySelectorAll(".chip")].map(
  c => ({ label: c.textContent.trim(), q: c.dataset.q }));

let busy = false;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* Make a raw database value readable. */
function pretty(key, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";

  const k = String(key).toLowerCase();

  /* seconds -> 5 min / 1h 20m */
  if (/second|duration|seconds_present/.test(k) && !isNaN(value)) {
    const s = Math.round(Number(value));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)} min`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }

  /* an ISO timestamp -> 11 Aug 14:21 */
  if (typeof value === "string" &&
      /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(value)) {
    const d = new Date(value.replace(" ", "T"));
    if (!isNaN(d)) {
      return d.toLocaleString([], {
        day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
      });
    }
  }

  /* 0.7617613313899886 -> 0.76 */
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(2);
  }
  return value;
}

/* ------------------------------------------------------------ chips */

function setChips(chips) {
  const list = (chips && chips.length) ? chips : OPENING_CHIPS;
  chipbar.innerHTML = list.map(c =>
    `<button type="button" class="chip" data-q="${esc(c.q)}">${esc(c.label)}</button>`
  ).join("");
  chipbar.scrollLeft = 0;
}

chipbar.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip && !busy) ask(chip.dataset.q);
});

/* --------------------------------------------------------- messages */

function bubble(who, html) {
  if (welcome) welcome.style.display = "none";

  const row = document.createElement("div");
  row.className = `msg ${who}`;
  row.innerHTML = who === "bot"
    ? `<div class="avatar"><i class="bi bi-camera-video"></i></div>
       <div class="bubble"></div>`
    : `<div class="bubble"></div>`;
  const body = row.querySelector(".bubble");
  body.innerHTML = html;

  log.appendChild(row);
  toBottom();
  return body;
}

function toBottom() {
  scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
}

function table(columns, rows, showAll) {
  if (!rows || !rows.length) return "";
  let cols = (columns && columns.length) ? columns : Object.keys(rows[0]);
  const hidden = showAll ? [] : cols.filter(c => NOISY.has(String(c).toLowerCase()));
  if (!showAll) cols = cols.filter(c => !NOISY.has(String(c).toLowerCase()));
  if (!cols.length) cols = Object.keys(rows[0]);

  const limit = 12;
  const head = cols.map(c =>
    `<th>${esc(String(c).replace(/_/g, " "))}</th>`).join("");
  const body = rows.slice(0, limit).map(r =>
    `<tr>${cols.map(c => `<td>${esc(pretty(c, r[c]))}</td>`).join("")}</tr>`
  ).join("");

  const bits = [];
  if (rows.length > limit) bits.push(`${rows.length - limit} more row(s)`);
  if (hidden.length) bits.push(
    `<a href="#" class="showall">show ${hidden.length} more column(s)</a>`);
  const footer = bits.length
    ? `<div class="src mt-1">${bits.join(" &middot; ")}</div>` : "";

  return `<div class="tablewrap"><table class="mini">
            <thead><tr>${head}</tr></thead><tbody>${body}</tbody>
          </table></div>${footer}`;
}

function render(target, data, showAll) {
  const via = data.source === "sql"
    ? `<i class="bi bi-database"></i> answered by a generated query`
    : `<i class="bi bi-lightning-charge"></i> ${esc(data.source || "")}`;

  /* The query is shown, but folded away - visible when you want to
     check it, not shouting over the answer. */
  const sql = data.sql
    ? `<details class="sqlbox"><summary>show the query</summary>
         <code>${esc(data.sql)}</code></details>` : "";

  /* THE FRAME AN ANSWER WAS READ OFF, when a vision model was involved.
     Not decoration and not folded away: "Pavan is on the phone" is the
     one kind of answer this system gives that no row can corroborate,
     and if the box labelled Pavan is around somebody else you can see
     that in one glance and in no other way. */
  const shot = data.image
    ? `<img class="scene" src="${esc(data.image)}" alt="the frame this answer was read from" loading="lazy">`
    : "";

  target.innerHTML =
    `<div class="answer">${esc(data.answer).replace(/\n/g, "<br>")}</div>` +
    shot + table(data.columns, data.rows, showAll) + sql +
    `<div class="src">${via}</div>`;

  const toggle = target.querySelector(".showall");
  if (toggle) {
    toggle.addEventListener("click", (e) => {
      e.preventDefault();
      render(target, data, true);
    });
  }
}

/* ------------------------------------------------------------- ask */

function working(on) {
  busy = on;
  send.disabled = on;
  chipbar.querySelectorAll(".chip").forEach(c => c.disabled = on);
}

async function ask(question) {
  if (!question || busy) return;
  bubble("me", esc(question));
  const waiting = bubble("bot",
    `<span class="dots"><span></span><span></span><span></span></span>`);
  working(true);

  let data;
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ question }),
    });
    if (r.status === 401) { window.location = "/login"; return; }
    data = await r.json();
  } catch (e) {
    waiting.innerHTML = `<span class="text-danger">Could not reach the server.</span>`;
    working(false);
    return;
  }
  render(waiting, data, false);
  setChips(data.chips);
  working(false);
  toBottom();
  box.focus();
}

/* ----------------------------------------------------------- input */

/* The box grows with the question and shrinks back after sending, so a
   long question is readable without a scrollbar inside a one-line box. */
function autosize() {
  box.style.height = "auto";
  box.style.height = Math.min(box.scrollHeight, 128) + "px";
}

box.addEventListener("input", autosize);

box.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = box.value.trim();
  if (!question) return;
  box.value = "";
  autosize();
  ask(question);
});

document.getElementById("clear").addEventListener("click", () => {
  log.innerHTML = "";
  if (welcome) welcome.style.display = "";
  setChips(OPENING_CHIPS);
  box.focus();
});

box.focus();

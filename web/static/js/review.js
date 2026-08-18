/* web/static/js/review.js
   The "is this the same person?" queue. Confirming an item teaches the
   system permanently. */

const grid = document.getElementById("reviewGrid");
const empty = document.getElementById("emptyState");
const counter = document.getElementById("pendingCount");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function card(item) {
  const conf = Math.round(item.probability * 100);
  /* An EMPTY suggested name means the system had no idea who this is,
     rather than a guess it wants confirming. Those are the valuable
     ones - they are the people whose reference photos are missing or
     unusable - so the card asks openly instead of nudging toward a name
     nobody actually proposed. */
  const unknown = !item.suggested || item.suggested === "Unknown";
  const dir = esc(item.camera_key || "");

  /* THE FACE IS THE MAIN PICTURE. It used to be the body crop, with the
     face in a 72px corner thumbnail - so the image a reviewer was asked
     to identify somebody from was the least identifiable one on the
     card. The body crop is kept as a thumbnail because it is useful
     context (where, and what they were wearing), just not the subject. */
  const media = item.face_image
    ? `<img src="/review-images/${dir}/${esc(item.face_image)}"
            class="review-face-main" alt="face"
            onerror="this.closest('.card').classList.add('no-image')">
       ${item.body_image
          ? `<img src="/review-images/${dir}/${esc(item.body_image)}"
                  class="review-body-thumb" alt="where"
                  onerror="this.remove()">` : ""}`
    : `<img src="/review-images/${dir}/${esc(item.body_image)}"
            class="review-body-only" alt="person"
            onerror="this.closest('.card').classList.add('no-image')">`;

  /* What the vision model saw. Shown even when it agrees with the
     suggestion, because it is also the audit trail: if this says
     "female, long hair" on a card asking about a man, the thing to go
     and look at is the screening, not the reviewer. */
  const looks = item.looks_like
    ? `<div class="looks-like"><i class="bi bi-eye"></i>
         looks like <strong>${esc(item.looks_like)}</strong></div>`
    : "";

  /* The shortlist. These are people whose own photographs do not
     CONTRADICT what is on screen - not people the system recognised.
     Clicking one fills the box rather than saving straight away, so
     what is about to be filed as ground truth is visible first. */
  const chips = (item.candidates || []).length
    ? `<div class="small text-secondary mb-1">could be one of these
         &mdash; nothing about them is ruled out:</div>
       <div class="candidate-row">
         ${item.candidates.map(n =>
           `<button type="button" class="candidate-chip"
                    data-name="${esc(n)}">${esc(n)}</button>`).join("")}
       </div>`
    : "";

  return `
  <div class="col-12 col-sm-6 col-lg-4 col-xl-3" data-id="${item.id}">
    <div class="card h-100">
      <div class="review-media">${media}</div>
      <div class="card-body">
        <div class="small text-secondary mb-1">
          ${esc(item.camera)} &middot; track #${item.track_id}
        </div>
        <div class="mb-2">
          ${unknown
            ? `<strong>Who is this?</strong>
               <span class="pill pill-unknown">not recognised</span>`
            : `Is this <strong>${esc(item.suggested)}</strong>?
               <span class="pill ${conf >= 55 ? "pill-known" : "pill-unknown"}">${conf}% sure</span>`}
        </div>
        ${looks}
        ${!unknown && item.runner_up
          ? `<div class="small text-secondary mb-2">could also be
               ${esc(item.runner_up)} &middot; margin ${item.margin}</div>` : ""}
        <div class="small text-secondary mb-2">${esc(item.evidence)}</div>
        ${chips}

        ${unknown ? "" : `
        <div class="d-flex gap-2 mb-2">
          <button class="btn btn-sm btn-success flex-fill" data-act="confirm">
            <i class="bi bi-check-lg"></i> Yes
          </button>
          <button class="btn btn-sm btn-outline-danger flex-fill" data-act="reject">
            <i class="bi bi-x-lg"></i> No
          </button>
        </div>`}
        <div class="input-group input-group-sm">
          <input class="form-control" list="knownPeople"
                 placeholder="${unknown ? "type their name" : "or type the correct person"}">
          <button class="btn btn-${unknown ? "primary" : "outline-primary"}"
                  data-act="reassign">Save</button>
        </div>
        ${unknown ? `
        <button class="btn btn-sm btn-outline-secondary w-100 mt-2" data-act="reject">
          <i class="bi bi-x-lg"></i> Not a face / skip
        </button>` : ""}
      </div>
    </div>
  </div>`;
}

async function load() {
  const r = await fetch("/api/reviews", { credentials: "same-origin" });
  if (r.status === 401) { window.location = "/login"; return; }
  const data = await r.json();
  const items = data.pending || [];
  counter.textContent = `${items.length} pending`;

  if (!items.length) {
    grid.innerHTML = "";
    grid.appendChild(empty);
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";
  grid.innerHTML = items.map(card).join("");
}

/* Clicking a shortlisted name fills the box; Save files it. One extra
   click, and it means what is about to be recorded as ground truth is
   on screen before it is recorded. */
grid.addEventListener("click", (e) => {
  const chip = e.target.closest(".candidate-chip");
  if (!chip) return;
  const col = chip.closest("[data-id]");
  col.querySelectorAll(".candidate-chip").forEach(c =>
    c.classList.remove("picked"));
  chip.classList.add("picked");
  const input = col.querySelector("input");
  input.value = chip.dataset.name;
  input.focus();
});

grid.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const col = btn.closest("[data-id]");
  const id = col.dataset.id;
  const act = btn.dataset.act;

  let name = "";
  if (act === "confirm") {
    name = col.querySelector("strong").textContent.trim();
  } else if (act === "reassign") {
    name = col.querySelector("input").value.trim();
    if (!name) { col.querySelector("input").focus(); return; }
  }

  btn.disabled = true;
  const r = await fetch(`/api/review/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ action: act, name }),
  });
  const out = await r.json();
  col.style.transition = "opacity .25s";
  col.style.opacity = "0";
  setTimeout(() => { col.remove(); load(); }, 250);
  say(out, act, name);
});

/* Say what the answer actually did. Confirming used to give no feedback
   at all beyond the card vanishing, so there was no way to tell a
   confirmation that corrected the record from one that quietly did
   nothing. */
function say(out, act, name) {
  if (act === "reject") { flash("Marked as not this person."); return; }
  const rows = (out.updated?.presence || 0) + (out.updated?.visits || 0);
  const bits = [];
  if (rows) bits.push("record updated");
  if (out.learned) bits.push("will be recognised on sight");
  flash(bits.length
    ? `${name}: ${bits.join(", ")}.`
    : `${name} saved, but nothing could be learned from that crop.`);
}

function flash(text) {
  let box = document.getElementById("flash");
  if (!box) {
    box = document.createElement("div");
    box.id = "flash";
    box.className = "alert alert-success py-2 px-3 small position-fixed";
    box.style.cssText = "bottom:1rem;right:1rem;z-index:1080;max-width:22rem";
    document.body.appendChild(box);
  }
  box.textContent = text;
  box.style.opacity = "1";
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { box.style.opacity = "0"; }, 4000);
}

load();
setInterval(load, 15000);

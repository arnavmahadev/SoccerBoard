// Interactive xG pitch. SVG user units == StatsBomb pitch units (see viewBox),
// so a marker's (x, y) is already a valid model coordinate.

const PITCH = { xMin: 60, xMax: 120, yMin: 0, yMax: 80 };

const svg = document.getElementById("pitch");
const layer = document.getElementById("markers");
const xgValue = document.getElementById("xg-value");
const xgBar = document.getElementById("xg-bar");
const statusEl = document.getElementById("status");

let markers = [];
let nextId = 0;

// role -> {team, is_gk} for building the GameState payload.
const ROLE = {
  shooter: null, // the shooter is shot_xy, not a player entry
  att: { team: "att", is_gk: false },
  def: { team: "def", is_gk: false },
  gk: { team: "def", is_gk: true },
};

function defaultMarkers() {
  return [
    { role: "shooter", x: 100, y: 40 },
    { role: "gk", x: 117, y: 40 },
    { role: "def", x: 110, y: 35 },
    { role: "def", x: 110, y: 45 },
    { role: "def", x: 113, y: 40 },
    { role: "att", x: 97, y: 52 },
    { role: "att", x: 95, y: 30 },
  ].map((m) => ({ id: nextId++, ...m }));
}

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

// Convert a pointer event into pitch coordinates via the SVG's transform.
function toPitch(evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return {
    x: clamp(p.x, PITCH.xMin, PITCH.xMax),
    y: clamp(p.y, PITCH.yMin, PITCH.yMax),
  };
}

function render() {
  layer.innerHTML = "";
  for (const m of markers) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", m.x);
    c.setAttribute("cy", m.y);
    c.setAttribute("r", m.role === "shooter" ? 1.9 : 1.6);
    c.setAttribute("class", `marker ${m.role}`);
    c.dataset.id = m.id;
    g.appendChild(c);
    if (m.role === "gk" || m.role === "shooter") {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", m.x);
      t.setAttribute("y", m.y);
      t.setAttribute("class", "marker-label");
      t.textContent = m.role === "gk" ? "GK" : "S";
      g.appendChild(t);
    }
    layer.appendChild(g);
  }
}

function buildGameState() {
  const shooter = markers.find((m) => m.role === "shooter");
  const players = markers
    .filter((m) => m.role !== "shooter")
    .map((m) => ({ xy: [round(m.x), round(m.y)], ...ROLE[m.role] }));
  return { shot_xy: [round(shooter.x), round(shooter.y)], players };
}

function round(v) { return Math.round(v * 10) / 10; }

async function predict() {
  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildGameState()),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    xgValue.textContent = data.xg.toFixed(2);
    xgBar.style.width = `${Math.min(data.xg * 100, 100)}%`;
    statusEl.textContent = `model: ${data.model}`;
  } catch (e) {
    statusEl.textContent = `prediction failed: ${e.message}`;
  }
}

// --- Dragging -------------------------------------------------------------
let dragId = null;

svg.addEventListener("pointerdown", (evt) => {
  const id = evt.target.dataset && evt.target.dataset.id;
  if (id === undefined) return;
  dragId = Number(id);
  evt.target.classList.add("dragging");
  svg.setPointerCapture(evt.pointerId);
});

svg.addEventListener("pointermove", (evt) => {
  if (dragId === null) return;
  const { x, y } = toPitch(evt);
  const m = markers.find((mm) => mm.id === dragId);
  m.x = x; m.y = y;
  render();
});

svg.addEventListener("pointerup", () => {
  if (dragId === null) return;
  dragId = null;
  predict(); // re-score once the player is dropped
});

// --- Controls -------------------------------------------------------------
function addMarker(role, x, y) {
  markers.push({ id: nextId++, role, x, y });
  render(); predict();
}
document.getElementById("add-def").onclick = () => addMarker("def", 105, 40);
document.getElementById("add-att").onclick = () => addMarker("att", 92, 40);
document.getElementById("reset").onclick = () => {
  markers = defaultMarkers(); render(); predict();
};

// --- Init -----------------------------------------------------------------
markers = defaultMarkers();
render();
predict();

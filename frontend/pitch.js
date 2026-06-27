// Interactive xG pitch. SVG user units == StatsBomb pitch units (see viewBox),
// so a marker's (x, y) is already a valid model coordinate.

const PITCH = { xMin: 60, xMax: 120, yMin: 0, yMax: 80 };

// Goal geometry — mirrors xg.data.schema so the on-pitch cone and the live
// stat read-outs match exactly what the model is fed.
const GOAL = { center: [120, 40], left: [120, 36], right: [120, 44] };

// Real field dimensions (yards) mapped onto the 120 x 80 StatsBomb grid, so the
// displayed distances reflect an actual 115 x 75 yd pitch. Length runs along x
// (120 units -> 115 yd), width along y (80 units -> 75 yd). Display only.
const YD_PER_X = 115 / 120;
const YD_PER_Y = 75 / 80;

const SVGNS = "http://www.w3.org/2000/svg";

const svg = document.getElementById("pitch");
const layer = document.getElementById("markers");
const geo = document.getElementById("geometry");
const xgValue = document.getElementById("xg-value");
const xgVerdict = document.getElementById("xg-verdict");
const xgPct = document.getElementById("xg-pct");
const xgBar = document.getElementById("xg-bar");

const stat = {
  dist: document.getElementById("s-dist"),
  angle: document.getElementById("s-angle"),
  cone: document.getElementById("s-cone"),
  near: document.getElementById("s-near"),
  gk: document.getElementById("s-gk"),
  off: document.getElementById("s-off"),
};

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

// Convert a pointer event into pitch coordinates. The markers layer sits inside
// the rotate(-90) group, so its CTM already encodes the rotation — mapping a
// screen point through its inverse yields pitch coordinates directly.
function toPitch(evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(layer.getScreenCTM().inverse());
  return {
    x: clamp(p.x, PITCH.xMin, PITCH.xMax),
    y: clamp(p.y, PITCH.yMin, PITCH.yMax),
  };
}

// --- Shot geometry (mirrors xg.features.build) ----------------------------
function shotAngleDeg(s) {
  const vl = [GOAL.left[0] - s[0], GOAL.left[1] - s[1]];
  const vr = [GOAL.right[0] - s[0], GOAL.right[1] - s[1]];
  const mag = Math.hypot(...vl) * Math.hypot(...vr);
  if (mag === 0) return 0;
  const cos = (vl[0] * vr[0] + vl[1] * vr[1]) / mag;
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

function pointInTriangle(p, a, b, c) {
  const sign = (p1, p2, p3) =>
    (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  const d1 = sign(p, a, b), d2 = sign(p, b, c), d3 = sign(p, c, a);
  const neg = d1 < 0 || d2 < 0 || d3 < 0;
  const pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

function isInCone(m, shooter) {
  if (m.role !== "def") return false;
  return pointInTriangle([m.x, m.y], [shooter.x, shooter.y], GOAL.left, GOAL.right);
}

// --- Rendering ------------------------------------------------------------
function el(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function render() {
  const shooter = markers.find((m) => m.role === "shooter");

  // Shot cone (shooter -> both posts) + sightline to goal center, drawn behind.
  geo.innerHTML = "";
  if (shooter) {
    geo.appendChild(el("path", {
      class: "cone",
      d: `M ${shooter.x} ${shooter.y} L ${GOAL.left[0]} ${GOAL.left[1]} L ${GOAL.right[0]} ${GOAL.right[1]} Z`,
    }));
    geo.appendChild(el("line", {
      class: "sightline",
      x1: shooter.x, y1: shooter.y, x2: GOAL.center[0], y2: GOAL.center[1],
    }));
  }

  layer.innerHTML = "";
  for (const m of markers) {
    const g = el("g", {});
    const cls = `marker ${m.role}` + (shooter && isInCone(m, shooter) ? " in-cone" : "");
    const c = el("circle", {
      cx: m.x, cy: m.y, r: m.role === "shooter" ? 1.9 : 1.6, class: cls,
    });
    c.dataset.id = m.id;
    g.appendChild(c);
    if (m.role === "gk" || m.role === "shooter") {
      // Counter-rotate the label so it stays upright inside the rotated pitch.
      const t = el("text", {
        x: m.x, y: m.y, class: "marker-label",
        transform: `rotate(90 ${m.x} ${m.y})`,
      });
      t.textContent = m.role === "gk" ? "GK" : "S";
      g.appendChild(t);
    }
    layer.appendChild(g);
  }

  updateStats();
}

// --- Live stat read-outs --------------------------------------------------
// Distance in yards between two pitch points, measured center-to-center and
// scaled by the real field dimensions (x and y use different yard-per-unit).
function yardsBetween(a, b) {
  const dx = (a[0] - b[0]) * YD_PER_X;
  const dy = (a[1] - b[1]) * YD_PER_Y;
  return Math.hypot(dx, dy).toFixed(1);
}
function valHTML(num, unit) { return `${num}<span class="unit">${unit}</span>`; }

function updateStats() {
  const shooter = markers.find((m) => m.role === "shooter");
  if (!shooter) return;
  const s = [shooter.x, shooter.y];
  const defs = markers.filter((m) => m.role === "def");
  const gk = markers.find((m) => m.role === "gk");

  const nCone = defs.filter((m) => isInCone(m, shooter)).length;
  const nearest = defs.length
    ? Math.min(...defs.map((m) => Number(yardsBetween(s, [m.x, m.y]))))
    : null;

  stat.dist.innerHTML = valHTML(yardsBetween(s, GOAL.center), "yards");
  stat.angle.innerHTML = valHTML(shotAngleDeg(s).toFixed(0), "°");
  stat.cone.innerHTML = valHTML(nCone, "");
  stat.cone.classList.toggle("warn", nCone > 0);
  stat.near.innerHTML = nearest === null ? "—" : valHTML(nearest.toFixed(1), "yards");
  stat.gk.innerHTML = gk ? valHTML(yardsBetween([gk.x, gk.y], GOAL.center), "yards") : "—";
  stat.off.innerHTML = valHTML(yardsBetween([shooter.x, 40], [shooter.x, shooter.y]), "yards");
}

// --- Verdict tiers --------------------------------------------------------
function verdict(xg) {
  if (xg >= 0.5) return "big chance";
  if (xg >= 0.25) return "good chance";
  if (xg >= 0.1) return "half chance";
  if (xg >= 0.04) return "speculative";
  return "long shot";
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
    const xg = data.xg;

    xgValue.textContent = xg.toFixed(2);
    xgPct.textContent = `≈ ${Math.round(xg * 100)}% chance to score`;
    xgBar.style.width = `${Math.min(xg * 100, 100)}%`;
    xgVerdict.textContent = verdict(xg);
  } catch (e) {
    xgValue.textContent = "—";
    xgVerdict.textContent = `model error`;
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

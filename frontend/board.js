// Coach's tactics whiteboard. SVG user units == pitch units (120 x 80), so a
// token's (x, y) and a drawn stroke's points are in pitch coordinates directly.

const SVGNS = "http://www.w3.org/2000/svg";
const PITCH = { xMin: 0, xMax: 120, yMin: 0, yMax: 80 };

const PALETTE = ["#ffd166", "#ffffff", "#38bdf8", "#e2615a", "#36c275"];

const svg = document.getElementById("board");
const strokeLayer = document.getElementById("strokes");
const tokenLayer = document.getElementById("tokens");
const liveLayer = document.getElementById("live");

let state = { tokens: [], strokes: [] };
let nextId = 0;
let tool = "move";
let color = PALETTE[0];

// --- helpers --------------------------------------------------------------
function el(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}
function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
function colorId(hex) { return "c" + hex.replace("#", ""); }
function round(v) { return Math.round(v * 10) / 10; }

function toPitch(evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return {
    x: clamp(p.x, PITCH.xMin, PITCH.xMax),
    y: clamp(p.y, PITCH.yMin, PITCH.yMax),
  };
}

// --- undo -----------------------------------------------------------------
const undoStack = [];
function snapshot() {
  undoStack.push(JSON.stringify(state));
  if (undoStack.length > 60) undoStack.shift();
}
function undo() {
  if (!undoStack.length) return;
  state = JSON.parse(undoStack.pop());
  render();
}

// --- arrowhead markers (one per palette color) ----------------------------
function buildMarkers() {
  const defs = svg.querySelector("defs");
  for (const hex of PALETTE) {
    // refX at the tip (x=10) so the arrow point sits exactly on the line's end;
    // the head then overlaps the stem instead of the stem poking past the tip.
    // userSpaceOnUse keeps the head a fixed size, independent of stroke width.
    const m = el("marker", {
      id: "ah-" + colorId(hex),
      viewBox: "0 0 10 10", refX: "10", refY: "5",
      markerUnits: "userSpaceOnUse",
      markerWidth: "4.5", markerHeight: "4.5", orient: "auto-start-reverse",
    });
    m.appendChild(el("path", { d: "M 0 1.5 L 10 5 L 0 8.5 z", fill: hex }));
    defs.appendChild(m);
  }
}

// --- rendering ------------------------------------------------------------
function renderToken(t) {
  const g = el("g", { class: "token-g" });
  if (t.type === "ball") {
    const c = el("circle", { cx: t.x, cy: t.y, r: 1.5, class: "token ball" });
    c.dataset.token = t.id;
    g.appendChild(c);
    g.appendChild(el("circle", { cx: t.x, cy: t.y, r: 0.55, class: "ball-pip" }));
  } else if (t.type === "cone") {
    const p = el("path", {
      d: `M ${t.x} ${t.y - 2.1} L ${t.x + 1.8} ${t.y + 1.6} L ${t.x - 1.8} ${t.y + 1.6} Z`,
      class: "token cone",
    });
    p.dataset.token = t.id;
    g.appendChild(p);
  } else {
    const c = el("circle", { cx: t.x, cy: t.y, r: 2.4, class: `token ${t.type}` });
    c.dataset.token = t.id;
    g.appendChild(c);
    const label = el("text", { x: t.x, y: t.y, class: "token-label" });
    label.textContent = t.label;
    g.appendChild(label);
  }
  tokenLayer.appendChild(g);
}

function strokePath(points) {
  return "M " + points.map((p) => `${p[0]} ${p[1]}`).join(" L ");
}

function renderStroke(s) {
  if (s.type === "pen") {
    const path = el("path", {
      d: strokePath(s.points), class: "stroke pen", stroke: s.color,
    });
    path.dataset.stroke = s.id;
    strokeLayer.appendChild(path);
  } else {
    const line = el("line", {
      x1: s.a[0], y1: s.a[1], x2: s.b[0], y2: s.b[1],
      class: "stroke arrow" + (s.dashed ? " dashed" : ""),
      stroke: s.color, "marker-end": `url(#ah-${colorId(s.color)})`,
    });
    line.dataset.stroke = s.id;
    strokeLayer.appendChild(line);
  }
}

function render() {
  strokeLayer.innerHTML = "";
  for (const s of state.strokes) renderStroke(s);
  tokenLayer.innerHTML = "";
  for (const t of state.tokens) renderToken(t);
}

// --- interaction ----------------------------------------------------------
let dragId = null;
let dragMoved = false;
let drawing = null; // { type, color, points } | { type, a, color, dashed }

function setTool(t) {
  tool = t;
  document.querySelectorAll(".tb-tool").forEach((b) =>
    b.classList.toggle("active", b.dataset.tool === t));
  svg.dataset.tool = t;
}

svg.addEventListener("pointerdown", (evt) => {
  const targetTokenId = evt.target.dataset && evt.target.dataset.token;
  const targetStrokeId = evt.target.dataset && evt.target.dataset.stroke;

  if (tool === "erase") {
    if (targetTokenId !== undefined) {
      snapshot();
      state.tokens = state.tokens.filter((t) => String(t.id) !== targetTokenId);
      render();
    } else if (targetStrokeId !== undefined) {
      snapshot();
      state.strokes = state.strokes.filter((s) => String(s.id) !== targetStrokeId);
      render();
    }
    return;
  }

  if (tool === "move") {
    if (targetTokenId === undefined) return;
    dragId = Number(targetTokenId);
    dragMoved = false;
    snapshot();
    svg.setPointerCapture(evt.pointerId);
    return;
  }

  // drawing tools (pen / pass / run)
  const { x, y } = toPitch(evt);
  if (tool === "pen") {
    drawing = { type: "pen", color, points: [[round(x), round(y)]] };
  } else {
    drawing = { type: "arrow", color, dashed: tool === "run", a: [round(x), round(y)], b: [round(x), round(y)] };
  }
  svg.setPointerCapture(evt.pointerId);
});

svg.addEventListener("pointermove", (evt) => {
  if (dragId !== null) {
    const { x, y } = toPitch(evt);
    const m = state.tokens.find((t) => t.id === dragId);
    m.x = round(x); m.y = round(y);
    dragMoved = true;
    render();
    return;
  }
  if (!drawing) return;

  const { x, y } = toPitch(evt);
  liveLayer.innerHTML = "";
  if (drawing.type === "pen") {
    const last = drawing.points[drawing.points.length - 1];
    if (Math.hypot(x - last[0], y - last[1]) > 0.6) drawing.points.push([round(x), round(y)]);
    liveLayer.appendChild(el("path", { d: strokePath(drawing.points), class: "stroke pen", stroke: color }));
  } else {
    drawing.b = [round(x), round(y)];
    liveLayer.appendChild(el("line", {
      x1: drawing.a[0], y1: drawing.a[1], x2: drawing.b[0], y2: drawing.b[1],
      class: "stroke arrow" + (drawing.dashed ? " dashed" : ""),
      stroke: color, "marker-end": `url(#ah-${colorId(color)})`,
    }));
  }
});

function endStroke() {
  if (!drawing) return;
  liveLayer.innerHTML = "";
  const ok = drawing.type === "pen"
    ? drawing.points.length > 1
    : Math.hypot(drawing.b[0] - drawing.a[0], drawing.b[1] - drawing.a[1]) > 1.5;
  if (ok) {
    snapshot();
    state.strokes.push({ id: nextId++, ...drawing });
    render();
  }
  drawing = null;
}

svg.addEventListener("pointerup", () => {
  if (dragId !== null) {
    // A click that didn't move shouldn't litter the undo stack.
    if (!dragMoved) undoStack.pop();
    dragId = null;
    return;
  }
  endStroke();
});
svg.addEventListener("pointercancel", () => { dragId = null; endStroke(); });

// --- controls -------------------------------------------------------------
document.getElementById("tools").addEventListener("click", (e) => {
  const b = e.target.closest("[data-tool]");
  if (b) setTool(b.dataset.tool);
});
document.getElementById("toggle-numbers").addEventListener("change", (e) => {
  svg.dataset.numbers = e.target.checked ? "on" : "off";
});

document.getElementById("undo").onclick = undo;
document.getElementById("clear-draw").onclick = () => {
  if (!state.strokes.length) return;
  snapshot();
  state.strokes = [];
  render();
};
document.getElementById("reset").onclick = () => { snapshot(); resetBoard(); };

document.getElementById("formats").addEventListener("click", (e) => {
  const b = e.target.closest("[data-format]");
  if (!b) return;
  teamSize = Number(b.dataset.format);
  document.querySelectorAll("#formats .tb-fmt").forEach((f) =>
    f.classList.toggle("active", f === b));
  buildShapes(); // team size changed → repopulate both formation menus
  snapshot();
  resetBoard();
});

function buildColors() {
  const wrap = document.getElementById("colors");
  for (const hex of PALETTE) {
    const b = document.createElement("button");
    b.className = "swatch" + (hex === color ? " active" : "");
    b.style.background = hex;
    b.title = hex;
    b.onclick = () => {
      color = hex;
      document.querySelectorAll(".swatch").forEach((s) =>
        s.classList.toggle("active", s.style.background === hexToRgb(hex)));
      if (tool === "move" || tool === "erase") setTool("pen");
    };
    wrap.appendChild(b);
  }
}
// style.background normalises hex to rgb(); compare against that.
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}

// --- formations -----------------------------------------------------------
// Each formation is its outfield shape from defence -> attack (the GK is added
// separately); the counts sum to teamSize - 1. The first formation per size is
// the default. Positions are generated for the home half (attacking right) and
// mirrored across halfway for the away side.
const FORMATIONS = {
  11: {
    "4-3-3":   [4, 3, 3],
    "4-2-3-1": [4, 2, 3, 1],
    "4-4-2":   [4, 4, 2],
    "4-5-1":   [4, 5, 1],
    "3-5-2":   [3, 5, 2],
    "5-4-1":   [5, 4, 1],
    "5-3-2":   [5, 3, 2],
  },
  9: {
    "3-2-3": [3, 2, 3],
    "3-3-2": [3, 3, 2],
    "2-3-3": [2, 3, 3],
    "3-4-1": [3, 4, 1],
    "2-4-2": [2, 4, 2],
  },
  7: {
    "2-3-1":   [2, 3, 1],
    "3-2-1":   [3, 2, 1],
    "2-1-2-1": [2, 1, 2, 1],
    "3-1-2":   [3, 1, 2],
    "2-1-3":   [2, 1, 3],
  },
};

// Shirt numbers per slot, in the SAME order outfieldPositions() emits them:
// band by band from defence to attack, and within a band top -> bottom, which
// (home attacking right) reads left -> right. So a back four is
// [LB 3, LCB 5, RCB 4, RB 2]. Canonical: 1 GK, 2 RB, 3 LB, 4/5 CB, 6 CDM,
// 7 RW, 8 CM, 9 ST, 10 CAM, 11 LW; smaller formats reuse the lower numbers.
const LABELS = {
  11: {
    "4-3-3":   [3, 5, 4, 2,  8, 6, 10,  11, 9, 7],
    "4-2-3-1": [3, 5, 4, 2,  6, 8,  11, 10, 7,  9],
    "4-4-2":   [3, 5, 4, 2,  11, 8, 6, 7,  10, 9],
    "4-5-1":   [3, 5, 4, 2,  11, 8, 6, 10, 7,  9],
    "3-5-2":   [3, 4, 5,  11, 8, 6, 10, 2,  7, 9],
    "5-4-1":   [3, 5, 4, 6, 2,  11, 8, 10, 7,  9],
    "5-3-2":   [3, 5, 4, 6, 2,  8, 10, 7,  11, 9],
  },
  9: {
    "3-2-3": [3, 4, 2,  6, 8,  5, 9, 7],
    "3-3-2": [3, 4, 2,  5, 6, 7,  8, 9],
    "2-3-3": [3, 2,  4, 6, 8,  5, 9, 7],
    "3-4-1": [3, 4, 2,  5, 6, 8, 7,  9],
    "2-4-2": [3, 2,  4, 6, 8, 7,  5, 9],
  },
  7: {
    "2-3-1":   [3, 2,  4, 6, 5,  7],
    "3-2-1":   [3, 4, 2,  5, 6,  7],
    "2-1-2-1": [3, 2,  6,  4, 5,  7],
    "3-1-2":   [3, 4, 2,  6,  5, 7],
    "2-1-3":   [3, 2,  6,  4, 7, 5],
  },
};

let teamSize = 11;
// Each team picks its own formation — opponents rarely line up the same way.
let homeFormation = Object.keys(FORMATIONS[teamSize])[0];
let awayFormation = Object.keys(FORMATIONS[teamSize])[1] || homeFormation;

// Lay a formation's outfield players into the home half: bands march from the
// back (x=20) toward halfway (x=54). Within a band players are evenly spaced and
// centered on y=40, so a lone striker stays central while a back five fans wide.
function outfieldPositions(bands) {
  const xBack = 20, xFront = 54, spacing = 14;
  const pts = [];
  bands.forEach((count, bi) => {
    const x = bands.length === 1 ? 37 : xBack + (xFront - xBack) * (bi / (bands.length - 1));
    for (let j = 0; j < count; j++) {
      const y = clamp(40 + (j - (count - 1) / 2) * spacing, 10, 70);
      pts.push([round(x), round(y)]);
    }
  });
  return pts;
}

// Add one team's 11/9/7 to the board. The away side is a 180° rotation of the
// home layout (x and y both flipped through the center) so that, facing its own
// goal, its left/right — and therefore its shirt numbers — come out correct.
function addTeam(type, shape, gk) {
  const numbers = LABELS[teamSize][shape];
  state.tokens.push({ id: nextId++, type, x: gk[0], y: gk[1], label: "1" });
  outfieldPositions(FORMATIONS[teamSize][shape]).forEach(([x, y], i) => {
    const px = type === "away" ? 120 - x : x;
    const py = type === "away" ? 80 - y : y;
    state.tokens.push({ id: nextId++, type, x: px, y: py, label: String(numbers[i]) });
  });
}

function resetBoard() {
  state = { tokens: [], strokes: [] };
  nextId = 0;
  addTeam("home", homeFormation, [7, 40]);
  addTeam("away", awayFormation, [113, 40]);
  state.tokens.push({ id: nextId++, type: "ball", x: 60, y: 40, label: "" });
  render();
}

// The formation menus depend on the current team size; rebuilt when it changes.
function buildShapes() {
  const names = Object.keys(FORMATIONS[teamSize]);
  if (!names.includes(homeFormation)) homeFormation = names[0];
  if (!names.includes(awayFormation)) awayFormation = names[1] || names[0];
  fillSelect("home-formation", names, homeFormation);
  fillSelect("away-formation", names, awayFormation);
}

function fillSelect(id, names, selected) {
  const sel = document.getElementById(id);
  sel.innerHTML = "";
  for (const name of names) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    if (name === selected) o.selected = true;
    sel.appendChild(o);
  }
}

document.getElementById("home-formation").addEventListener("change", (e) => {
  homeFormation = e.target.value;
  snapshot();
  resetBoard();
});
document.getElementById("away-formation").addEventListener("change", (e) => {
  awayFormation = e.target.value;
  snapshot();
  resetBoard();
});

// --- init -----------------------------------------------------------------
buildMarkers();
buildColors();
buildShapes();
setTool("move");
resetBoard();
undoStack.length = 0;

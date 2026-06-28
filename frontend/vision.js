// Vision page: photo/video -> 2D pitch.
//
// Flow: load media -> click >=4 landmarks on the reference pitch and their
// matching spots on the source image (building image<->pitch point pairs) ->
// POST to /vision. A photo returns one frame; a video returns a timeline the
// scrubber indexes into. Pitch coordinates are the shared 120x80 grid.

const SVGNS = "http://www.w3.org/2000/svg";
const PITCH_W = 120;
const PITCH_H = 80;

function el(tag, attrs = {}, parent = null) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

// --- Pitch markings (shared by reference + result pitch) ------------------
// Drawn on the 0..120 x 0..80 grid; `cls` namespaces the stroke/fill classes.
function drawMarkings(svg, lineCls, spotCls, turfCls) {
  el("rect", { x: -6, y: -6, width: 132, height: 92, rx: 3, class: turfCls }, svg);
  const L = (a) => el("rect", { ...a, class: lineCls }, svg);
  el("rect", { x: 0, y: 0, width: 120, height: 80, class: lineCls, fill: "none" }, svg);
  el("line", { x1: 60, y1: 0, x2: 60, y2: 80, class: lineCls }, svg);
  el("circle", { cx: 60, cy: 40, r: 10, class: lineCls, fill: "none" }, svg);
  el("circle", { cx: 60, cy: 40, r: 0.6, class: spotCls }, svg);
  // left box
  L({ x: 0, y: 18, width: 18, height: 44 });
  L({ x: 0, y: 30, width: 6, height: 20 });
  el("circle", { cx: 12, cy: 40, r: 0.6, class: spotCls }, svg);
  el("path", { d: "M 18 32 A 10 10 0 0 1 18 48", class: lineCls, fill: "none" }, svg);
  el("line", { x1: 0, y1: 36, x2: 0, y2: 44, class: lineCls, "stroke-width": 1.2 }, svg);
  // right box
  L({ x: 102, y: 18, width: 18, height: 44 });
  L({ x: 114, y: 30, width: 6, height: 20 });
  el("circle", { cx: 108, cy: 40, r: 0.6, class: spotCls }, svg);
  el("path", { d: "M 102 32 A 10 10 0 0 0 102 48", class: lineCls, fill: "none" }, svg);
  el("line", { x1: 120, y1: 36, x2: 120, y2: 44, class: lineCls, "stroke-width": 1.2 }, svg);
}

// Named calibration landmarks the user can pick from. [x, y] in pitch units.
const LANDMARKS = [
  [0, 0], [60, 0], [120, 0],
  [0, 80], [60, 80], [120, 80],
  [60, 40], [60, 30], [60, 50],
  [0, 18], [0, 62], [18, 18], [18, 62], [0, 30], [0, 50], [6, 30], [6, 50], [12, 40],
  [120, 18], [120, 62], [102, 18], [102, 62], [120, 30], [120, 50], [114, 30], [114, 50], [108, 40],
];

// --- State ----------------------------------------------------------------
const state = {
  kind: null, // "image" | "video"
  file: null,
  natW: 0,
  natH: 0,
  pairs: [], // { pitch:[x,y], image:[x,y], mark: <ref circle> }
  armed: null, // { x, y, mark }
  timeline: null, // video: [{t, players, ball}]
};

// --- Elements -------------------------------------------------------------
const fileInput = document.getElementById("file-input");
const uploadText = document.getElementById("upload-text");
const drop = document.getElementById("drop");
const frame = document.getElementById("media-frame");
const img = document.getElementById("media-img");
const video = document.getElementById("media-video");
const overlay = document.getElementById("cal-overlay");
const refPitch = document.getElementById("ref-pitch");
const pitch = document.getElementById("pitch");
const calCount = document.getElementById("cal-count");
const hint = document.getElementById("hint");
const runBtn = document.getElementById("run");
const clearBtn = document.getElementById("clear-cal");
const counts = document.getElementById("counts");
const sourceSub = document.getElementById("source-sub");
const scrubWrap = document.getElementById("scrub-wrap");
const scrub = document.getElementById("scrub");
const timeLabel = document.getElementById("time-label");
const playToggle = document.getElementById("play-toggle");

// --- Build the two pitches ------------------------------------------------
drawMarkings(refPitch, "rp-line", "rp-line", "rp-turf");
const refMarks = el("g", {}, refPitch);
for (const [x, y] of LANDMARKS) {
  const c = el("circle", { cx: x, cy: y, r: 1.8, class: "rp-mark" }, refMarks);
  c.dataset.x = x;
  c.dataset.y = y;
  c.addEventListener("click", () => armLandmark(x, y, c));
}

drawMarkings(pitch, "pz-line", "pz-spot", "pz-turf");
const tokens = el("g", { id: "pz-tokens" }, pitch);

// --- Calibration ----------------------------------------------------------
function armLandmark(x, y, mark) {
  // Re-arming a used landmark drops its old pair so the next image click resets it.
  state.pairs = state.pairs.filter((p) => p.mark !== mark);
  for (const m of refMarks.children) m.classList.remove("armed");
  mark.classList.add("armed");
  mark.classList.remove("used");
  state.armed = { x, y, mark };
  frame.classList.add("armed");
  setHint(`Now click where the marked point is in your image.`);
  syncCalibration();
}

// Pointer -> natural image coordinates via the overlay's own transform.
function overlayPoint(evt) {
  const pt = overlay.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(overlay.getScreenCTM().inverse());
  return [p.x, p.y];
}

overlay.addEventListener("pointerdown", (evt) => {
  if (!state.armed) {
    setHint("Pick a landmark on the small pitch first, then click your image.");
    return;
  }
  const [ix, iy] = overlayPoint(evt);
  const a = state.armed;
  state.pairs.push({ pitch: [a.x, a.y], image: [ix, iy], mark: a.mark });
  a.mark.classList.remove("armed");
  a.mark.classList.add("used");
  state.armed = null;
  frame.classList.remove("armed");
  drawPins();
  syncCalibration();
});

function drawPins() {
  // Pins live in the overlay; size them relative to the image so they read on
  // any resolution.
  [...overlay.querySelectorAll(".cal-pin, .cal-pin-label")].forEach((n) => n.remove());
  const r = Math.max(state.natW, state.natH) * 0.012 || 6;
  state.pairs.forEach((p, i) => {
    el("circle", { cx: p.image[0], cy: p.image[1], r, class: "cal-pin" }, overlay);
    const t = el("text", {
      x: p.image[0], y: p.image[1], class: "cal-pin-label", "font-size": r * 1.3,
    }, overlay);
    t.textContent = i + 1;
  });
}

function syncCalibration() {
  const n = state.pairs.length;
  calCount.textContent = n;
  calCount.classList.toggle("ready", n >= 4);
  clearBtn.disabled = n === 0 && !state.armed;
  runBtn.disabled = !(n >= 4 && state.file);
  if (n >= 4 && !state.armed) setHint("Ready — press Extrapolate to map your image onto the pitch.");
}

clearBtn.addEventListener("click", () => {
  state.pairs = [];
  state.armed = null;
  frame.classList.remove("armed");
  for (const m of refMarks.children) m.classList.remove("used", "armed");
  drawPins();
  syncCalibration();
  setHint("Calibration cleared. Pick landmarks again.");
});

// --- Media loading --------------------------------------------------------
fileInput.addEventListener("change", (e) => loadFile(e.target.files[0]));
["dragover", "drop"].forEach((ev) =>
  frame.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" && e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  })
);

function loadFile(file) {
  if (!file) return;
  resetResults();
  state.file = file;
  state.pairs = [];
  state.armed = null;
  for (const m of refMarks.children) m.classList.remove("used", "armed");
  drawPins();

  const url = URL.createObjectURL(file);
  drop.hidden = true;
  uploadText.textContent = file.name;

  if (file.type.startsWith("video/")) {
    state.kind = "video";
    img.hidden = true;
    video.hidden = false;
    video.src = url;
    video.addEventListener("loadedmetadata", onVideoMeta, { once: true });
  } else {
    state.kind = "image";
    video.hidden = true;
    scrubWrap.hidden = true;
    img.hidden = false;
    img.src = url;
    img.addEventListener("load", () => {
      state.natW = img.naturalWidth;
      state.natH = img.naturalHeight;
      setupOverlay();
      sourceSub.textContent = `${state.natW}×${state.natH} photo`;
    }, { once: true });
  }
  syncCalibration();
  setHint("Click a landmark on the small pitch, then the matching spot in your image.");
}

function onVideoMeta() {
  state.natW = video.videoWidth;
  state.natH = video.videoHeight;
  setupOverlay();
  scrubWrap.hidden = false;
  scrub.max = video.duration || 0;
  scrub.value = 0;
  timeLabel.textContent = "0.0s";
  sourceSub.textContent = `${state.natW}×${state.natH} · ${video.duration.toFixed(1)}s video`;
}

function setupOverlay() {
  overlay.setAttribute("viewBox", `0 0 ${state.natW} ${state.natH}`);
  overlay.hidden = false;
}

// --- Run ------------------------------------------------------------------
runBtn.addEventListener("click", run);

async function run() {
  const calib = state.pairs.map((p) => ({ image: p.image, pitch: p.pitch }));
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("calib", JSON.stringify(calib));

  runBtn.disabled = true;
  const endpoint = state.kind === "video" ? "/vision/video" : "/vision/photo";
  setHint(state.kind === "video"
    ? "Processing video frames… this runs on CPU and may take a moment."
    : "Detecting players and projecting…", "busy");

  try {
    if (state.kind === "video") fd.append("every_seconds", "0.5");
    const r = await fetch(endpoint, { method: "POST", body: fd });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    if (state.kind === "video") {
      state.timeline = data.frames;
      if (!data.frames.length) throw new Error("no frames could be read from this video");
      bindScrub();
      renderAtTime(video.currentTime);
      setHint(`Mapped ${data.count} frames. Scrub the video — the pitch follows.`);
    } else {
      renderFrame(data);
      const ball = data.ball ? " · ball found" : "";
      setHint(`Mapped ${data.players.length} players${ball}.`);
    }
  } catch (e) {
    setHint(`Could not process: ${e.message}`, "err");
  } finally {
    runBtn.disabled = false;
  }
}

// --- Render onto the pitch ------------------------------------------------
function renderFrame(frameData) {
  tokens.innerHTML = "";
  for (const p of frameData.players || []) {
    const team = p.team === 0 ? "team0" : p.team === 1 ? "team1" : "teamnull";
    el("circle", { cx: p.x, cy: p.y, r: 1.6, class: `pz-dot ${team}` }, tokens);
  }
  if (frameData.ball) {
    el("circle", { cx: frameData.ball.x, cy: frameData.ball.y, r: 1.1, class: "pz-ball" }, tokens);
  }
  const n = (frameData.players || []).length;
  counts.textContent = `${n} player${n === 1 ? "" : "s"}${frameData.ball ? " · ball" : ""}`;
}

function resetResults() {
  tokens && (tokens.innerHTML = "");
  counts.textContent = "";
  state.timeline = null;
  scrubWrap.hidden = true;
}

// --- Video scrubbing ------------------------------------------------------
function nearestFrame(t) {
  const tl = state.timeline;
  if (!tl || !tl.length) return null;
  let best = tl[0];
  let bestD = Math.abs(tl[0].t - t);
  for (const f of tl) {
    const d = Math.abs(f.t - t);
    if (d < bestD) { bestD = d; best = f; }
  }
  return best;
}

function renderAtTime(t) {
  const f = nearestFrame(t);
  if (f) renderFrame(f);
}

function bindScrub() {
  scrub.max = video.duration || 0;
  // The video element is the single source of truth for time; the range and the
  // pitch both follow it.
  scrub.oninput = () => { video.currentTime = parseFloat(scrub.value); };
  video.ontimeupdate = () => {
    scrub.value = video.currentTime;
    timeLabel.textContent = `${video.currentTime.toFixed(1)}s`;
    renderAtTime(video.currentTime);
  };
  video.onseeked = () => renderAtTime(video.currentTime);
}

playToggle.addEventListener("click", () => {
  if (video.paused) { video.play(); playToggle.textContent = "⏸"; }
  else { video.pause(); playToggle.textContent = "▶"; }
});
video.addEventListener("ended", () => (playToggle.textContent = "▶"));

// --- Misc -----------------------------------------------------------------
function setHint(text, cls = "") {
  hint.textContent = text;
  hint.className = "vz-hint" + (cls ? " " + cls : "");
}

// Feature availability check.
fetch("/vision/health").then((r) => r.json()).then((d) => {
  if (!d.available) {
    setHint("Vision backend is unavailable — install requirements-vision.txt and restart the server.", "err");
    fileInput.disabled = true;
  }
}).catch(() => {});

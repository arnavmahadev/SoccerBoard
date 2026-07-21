// Forecaster — clean, bracket-first. Dependency-free; talks to the FastAPI backend.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, o) => fetch(p, o).then((r) => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });
  const pct = (x, dp) => (x == null ? "·" : (x * 100).toFixed(dp || 0) + "%");
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  // team -> flag code (flagcdn). England/Scotland use UK subdivision flags.
  const FLAG = {
    "Algeria":"dz","Argentina":"ar","Austria":"at","Jordan":"jo","Australia":"au",
    "Paraguay":"py","Turkey":"tr","United States":"us","Belgium":"be","Egypt":"eg",
    "Iran":"ir","New Zealand":"nz","Bosnia and Herzegovina":"ba","Canada":"ca","Qatar":"qa",
    "Switzerland":"ch","Brazil":"br","Haiti":"ht","Morocco":"ma","Scotland":"gb-sct",
    "Cape Verde":"cv","Saudi Arabia":"sa","Spain":"es","Uruguay":"uy","Colombia":"co",
    "DR Congo":"cd","Portugal":"pt","Uzbekistan":"uz","Croatia":"hr","England":"gb-eng",
    "Ghana":"gh","Panama":"pa","Curaçao":"cw","Ecuador":"ec","Germany":"de",
    "Ivory Coast":"ci","Czech Republic":"cz","Mexico":"mx","South Africa":"za","South Korea":"kr",
    "France":"fr","Iraq":"iq","Norway":"no","Senegal":"sn","Japan":"jp",
    "Netherlands":"nl","Sweden":"se","Tunisia":"tn",
  };
  function flag(name) {
    const c = FLAG[name];
    if (!c) return "";
    return `<img class="flag" src="https://flagcdn.com/h20/${c}.png" ` +
      `srcset="https://flagcdn.com/h40/${c}.png 2x" alt="" loading="lazy" ` +
      `onerror="this.style.display='none'">`;
  }
  const ABBR = {
    "Algeria":"ALG","Argentina":"ARG","Austria":"AUT","Jordan":"JOR","Australia":"AUS",
    "Paraguay":"PAR","Turkey":"TUR","United States":"USA","Belgium":"BEL","Egypt":"EGY",
    "Iran":"IRN","New Zealand":"NZL","Bosnia and Herzegovina":"BIH","Canada":"CAN","Qatar":"QAT",
    "Switzerland":"SUI","Brazil":"BRA","Haiti":"HAI","Morocco":"MAR","Scotland":"SCO",
    "Cape Verde":"CPV","Saudi Arabia":"KSA","Spain":"ESP","Uruguay":"URU","Colombia":"COL",
    "DR Congo":"COD","Portugal":"POR","Uzbekistan":"UZB","Croatia":"CRO","England":"ENG",
    "Ghana":"GHA","Panama":"PAN","Curaçao":"CUW","Ecuador":"ECU","Germany":"GER",
    "Ivory Coast":"CIV","Czech Republic":"CZE","Mexico":"MEX","South Africa":"RSA","South Korea":"KOR",
    "France":"FRA","Iraq":"IRQ","Norway":"NOR","Senegal":"SEN","Japan":"JPN",
    "Netherlands":"NED","Sweden":"SWE","Tunisia":"TUN",
  };
  const named    = (t) => (t ? flag(t) + esc(t) : "TBD");
  const namedShort = (t) => (t ? flag(t) + esc(ABBR[t] || t) : `<span class="bk-tbd">TBD</span>`);

  // short button labels for each timeline stage
  const TL_SHORT = { pre:"Start", md1:"MD 1", md2:"MD 2", group:"Groups", r32:"R32", r16:"R16", qf:"QF", sf:"SF", final:"Final" };
  // the bracket only changes across knockout rounds, so its stepper skips the group matchdays
  const BK_STAGES = [["pre","Pre-knockout"], ["r32","R32"], ["r16","R16"], ["qf","QF"], ["sf","SF"], ["final","Final"]];
  const BK_ROUND = { r32:"Round of 32", r16:"Round of 16", qf:"quarter-finals", sf:"semi-finals" };

  const state = { comp: null, tl: null, byKey: {}, roster: [], rows: {}, oddsIdx: 0, bkIdx: 0 };

  async function init() {
    let comps;
    try { comps = await api("/forecaster/competitions"); }
    catch (e) { $("asof").textContent = "Forecaster artifacts not built. Run: python -m forecaster.build_artifacts"; return; }
    state.comp = comps[0].id;
    $("comp-title").textContent = comps[0].name + " Forecast";

    await Promise.all([loadTimeline(), loadGroups(), loadAccuracy()]);
  }

  // ---- timeline: one payload drives both the title odds and the bracket ------
  async function loadTimeline() {
    let tl;
    try { tl = await api("/forecaster/timeline?competition=" + state.comp); }
    catch (e) { $("tl-caption").textContent = "Timeline not built. Run: python -m forecaster.build_timeline"; return; }
    state.tl = tl;
    tl.checkpoints.forEach((c) => { state.byKey[c.key] = c; });
    $("asof").textContent = `The 2026 World Cup is complete. Champions: ${esc(tl.champion)}.`;

    buildOddsStepper(tl);
    buildBracketStepper();
  }

  // --- title odds -----------------------------------------------------------
  function buildOddsStepper(tl) {
    const cps = tl.checkpoints;
    // Fixed roster: the teams to track down the list, strongest first by their peak
    // title chance across the whole tournament. Rows stay put; the stage buttons
    // only re-order and re-length their bars, reading as a bar-chart race.
    const peak = {};
    cps.forEach((c) => c.teams.forEach((t) => { peak[t.team] = Math.max(peak[t.team] || 0, t.champion || 0); }));
    state.roster = Object.keys(peak).filter((t) => peak[t] > 0).sort((a, b) => peak[b] - peak[a]).slice(0, 12);

    $("odds").innerHTML = state.roster.map((team) =>
      `<div class="odds-item" data-team="${esc(team)}">
         <div class="odds-row">
           <span class="odds-name">${flag(team)}${esc(team)}</span>
           <span class="odds-track"><span class="odds-fill"></span></span>
           <span class="odds-val"></span>
         </div>
       </div>`).join("");
    state.rows = {};
    $("odds").querySelectorAll(".odds-item").forEach((el) => { state.rows[el.dataset.team] = el; });

    $("tl-ticks").innerHTML = cps.map((c, i) =>
      `<button class="stage-btn" data-i="${i}">${esc(TL_SHORT[c.key] || c.label)}</button>`).join("");
    $("tl-ticks").querySelectorAll(".stage-btn").forEach((b) =>
      { b.onclick = () => renderOdds(+b.dataset.i); });
    renderOdds(0);
  }

  function renderOdds(idx) {
    const tl = state.tl; if (!tl) return;
    state.oddsIdx = idx;
    const c = tl.checkpoints[idx];
    $("tl-stage").textContent = c.label;
    $("tl-alive").textContent = c.alive > 1 ? `${c.alive} teams still alive` : "";
    $("tl-caption").textContent = c.caption;
    $("tl-ticks").querySelectorAll(".stage-btn").forEach((b, i) => b.classList.toggle("active", i === idx));

    const byTeam = {}; c.teams.forEach((t) => { byTeam[t.team] = t; });
    const champ = (t) => (byTeam[t] ? byTeam[t].champion || 0 : 0);
    const max = Math.max(...state.roster.map(champ), 0.01);
    const order = state.roster.slice().sort((a, b) => champ(b) - champ(a));
    const leader = order[0];

    order.forEach((team, i) => {
      const item = state.rows[team];
      const p = champ(team);
      item.style.order = String(i);
      item.classList.toggle("out", p <= 0);
      item.classList.toggle("lead-team", team === leader && p > 0);
      item.querySelector(".odds-fill").style.width = (p / max * 100).toFixed(1) + "%";
      item.querySelector(".odds-val").textContent = p > 0 ? pct(p, p < 0.1 ? 1 : 0) : "out";
    });
  }

  // --- bracket: the same run-through, stage by stage ------------------------
  function buildBracketStepper() {
    const stages = BK_STAGES.filter(([k]) => state.byKey[k]);
    $("bk-stages").innerHTML = stages.map(([k, lab], i) =>
      `<button class="stage-btn" data-i="${i}">${esc(lab)}</button>`).join("");
    $("bk-stages").querySelectorAll(".stage-btn").forEach((b) =>
      { b.onclick = () => renderBracket(+b.dataset.i); });
    renderBracket(stages.length - 1); // open on the finished bracket
  }

  function bracketNote(cp) {
    const bk = cp.bracket, champ = esc(bk.champion || "TBD");
    if (cp.key === "pre")
      return `The model's predicted bracket from pre-tournament ratings, before a knockout game was played. Predicted champion: <b>${champ}</b>. Each percentage is that team's chance of winning that game.`;
    if (cp.key === "final")
      return `The finished bracket. The model called <b>${bk.correct} of ${bk.decided}</b> knockout ties right, a ✓ on each correct pick. Champion: <b>${champ}</b>.`;
    return `Results through the ${BK_ROUND[cp.key]} are locked in; the rest is the model's prediction from that point. It has called <b>${bk.correct} of ${bk.decided}</b> played ties right so far. Predicted champion now: <b>${champ}</b>.`;
  }

  const noteTag = (m) => (m && m.note ? ` <span class="bk-tag">${esc(m.note)}</span>` : "");

  function bkRow(name, meta, cls) {
    return `<div class="bk-row ${cls || ""}">
      <span class="bk-name">${namedShort(name)}</span>
      <span class="bk-meta">${meta || ""}</span></div>`;
  }

  // A tie in the run-through: locked to its real result (with a ✓/✗ against the
  // pre-tournament pick) once played, otherwise the model's prediction for it.
  function stageMatch(m) {
    if (m.settled) {
      const aWin = m.winner === m.a, nt = noteTag(m);
      const mark = m.correct == null ? "" :
        (m.correct ? `<span class="mark hit">✓</span> ` : `<span class="mark miss">✗</span> `);
      const sa = `${aWin ? mark : ""}${m.score[0]}${aWin ? nt : ""}`;
      const sb = `${!aWin ? mark : ""}${m.score[1]}${aWin ? "" : nt}`;
      return `<div class="bk-match"><div class="bk-box">
        ${bkRow(m.a, sa, aWin ? "win" : "")}
        ${bkRow(m.b, sb, aWin ? "" : "win")}
      </div></div>`;
    }
    if (!m.a || !m.b) {
      const cls = (t) => (t ? "" : "tbd");
      return `<div class="bk-match"><div class="bk-box">
        ${bkRow(m.a, "", cls(m.a))}${bkRow(m.b, "", cls(m.b))}
      </div></div>`;
    }
    const aWin = m.winner === m.a;
    return `<div class="bk-match"><div class="bk-box">
      ${bkRow(m.a, pct(m.prob_a), aWin ? "win" : "")}
      ${bkRow(m.b, pct(m.prob_b), aWin ? "" : "win")}
    </div></div>`;
  }

  function renderBracket(bkStageIdx) {
    const stages = BK_STAGES.filter(([k]) => state.byKey[k]);
    state.bkIdx = bkStageIdx;
    const [key] = stages[bkStageIdx];
    const cp = state.byKey[key];
    const data = cp.bracket;
    $("bk-stages").querySelectorAll(".stage-btn").forEach((b, i) => b.classList.toggle("active", i === bkStageIdx));
    $("bk-note").innerHTML = bracketNote(cp);

    const byRound = {};
    data.rounds.forEach((r) => { byRound[r.round] = r; });

    // Mirrored two-sided bracket: left half flows inward, right half mirrors.
    const side = (round, which) => {
      const r = byRound[round];
      if (!r) return [];
      const mid = Math.ceil(r.matches.length / 2);
      return which === "l" ? r.matches.slice(0, mid) : r.matches.slice(mid);
    };
    const label = (round) => (byRound[round] ? byRound[round].label : "");
    const halfCol = (round, which) => {
      const ms = side(round, which);
      return `<div class="bk-col ${which}${ms.length <= 1 ? " no-join" : ""}">
        <div class="bk-head">${label(round)}</div>
        <div class="bk-list">${ms.map(stageMatch).join("")}</div></div>`;
    };

    const flow = ["round_of_32", "round_of_16", "quarterfinal", "semifinal"];
    const left  = flow.map((r) => halfCol(r, "l")).join("");
    const right = flow.slice().reverse().map((r) => halfCol(r, "r")).join("");

    const finalRound = byRound.final;
    const finalM = finalRound && finalRound.matches[0];
    const tp = data.third_place || {};
    const champTbd = !data.champion;
    const champLabel = key === "final" ? "Champion" : "Predicted champion";
    const center = `<div class="bk-col center-col no-join">
      <div class="bk-final-content">
        <div class="bk-champ ${champTbd ? "tbd" : ""}">
          <span class="cl">${champLabel}</span>
          <span class="cn">${champTbd ? "TBD" : named(data.champion)}</span>
        </div>
        <div class="bk-center-node">
          <div class="bk-head">${label("final") || "Final"}</div>
          ${finalM ? stageMatch(finalM) : ""}
        </div>
        <div class="bk-bronze">
          <div class="bk-head">Third place</div>
          ${tp.a ? stageMatch(tp) : `<div class="bk-match"><div class="bk-box">
            ${bkRow(null, "", "tbd")}${bkRow(null, "", "tbd")}
          </div></div>`}
        </div>
      </div></div>`;

    const bk = $("bracket");
    bk.classList.add("two");
    bk.innerHTML = left + center + right;
  }

  // ---- groups -------------------------------------------------------------
  async function loadGroups() {
    const g = await api("/forecaster/groups?competition=" + state.comp);
    $("groups").innerHTML = Object.keys(g.groups).sort().map((L) => {
      const rows = g.groups[L].map((r) => {
        const cls = r.advanced ? "adv" : (r.played ? "elim" : "");
        const mark = r.played ? (r.advanced ? "✓" : "✗") : "·";
        const pred = r.forecast_advance == null ? "·" : pct(r.forecast_advance);
        return `<div class="grow ${cls}">
          <span class="gp">${r.position}</span>
          <span class="gname">${flag(r.team)}${esc(r.team)}</span>
          <span class="gpred">${pred}</span>
          <span class="gout">${mark}</span></div>`;
      }).join("");
      return `<div class="gcard"><h3>Group ${L}</h3>
        <div class="gcol-head"><span></span><span>Team</span><span>Pred.</span><span>Res.</span></div>
        ${rows}</div>`;
    }).join("");
  }

  // ---- accuracy -----------------------------------------------------------
  async function loadAccuracy() {
    let mt;
    try { mt = await api("/forecaster/metrics?competition=" + state.comp); } catch (e) { return; }
    if (!mt || !mt.calibration) return;
    renderCalib(mt.calibration.bins);
    const m = mt.model, c = mt.config;
    $("acc-text").innerHTML = `
      <p>The model doesn't just pick winners. It puts a <b>probability</b> on every
      result, and what matters is whether those probabilities are trustworthy. Across
      <b>${c.n_test.toLocaleString()}</b> real matches it had never seen before, it held
      up well: when it said 70%, the favourite won about 70% of the time.</p>
      <p class="acc-foot">Raw win/loss accuracy is a poor way to judge a model like this.
      In a knockout, there are no draws, so just guessing randomly gets you to 50% for free.
      The chart above is more useful: it plots predicted probability against how often that
      result actually happened. The closer to the dashed line, the better. The calibration
      error is <b>${m.ece.toFixed(3)}</b> (0 is perfect), and log-loss is
      <b>${m.log_loss.toFixed(2)}</b> versus ${mt.baseline.log_loss.toFixed(2)} for a naive
      baseline (lower is better).</p>`;
    $("fc-foot").textContent =
      `The scoreline model is a Dixon-Coles fit (bivariate Poisson) trained on about 49,000 ` +
      `international results. Title odds come from 10,000 Monte Carlo simulations. Team strength ` +
      `is set before the tournament starts, so results only decide who advances, not how a team ` +
      `is rated. The timeline replays those odds stage by stage: at each checkpoint it locks in ` +
      `the games already decided and nudges each side's rating by how it had actually played so ` +
      `far, then re-simulates the rest. Live results come from a public community dataset.`;
  }

  function renderCalib(bins) {
    const W = 300, H = 300, pad = 34;
    const X = (p) => pad + p * (W - 2 * pad), Y = (p) => H - pad - p * (H - 2 * pad);
    const maxN = Math.max(...bins.map((b) => b.n), 1);
    const pts = bins.map((b) => `<circle cx="${X(b.p_pred).toFixed(1)}" cy="${Y(b.p_obs).toFixed(1)}" r="${(3 + 5 * Math.sqrt(b.n / maxN)).toFixed(1)}" class="cpt"/>`).join("");
    const path = bins.map((b, i) => (i ? "L" : "M") + X(b.p_pred).toFixed(1) + " " + Y(b.p_obs).toFixed(1)).join(" ");
    const grid = [0, 0.5, 1].map((v) =>
      `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(1)}" class="cax"/><line x1="${X(0)}" y1="${Y(v)}" x2="${X(1)}" y2="${Y(v)}" class="cax"/>`).join("");
    $("acc-chart").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration curve">
        <style>
          .cax{stroke:var(--border);stroke-width:1}
          .cdiag{stroke:var(--text-muted);stroke-width:1.4;stroke-dasharray:5 4;opacity:.7}
          .cline{fill:none;stroke:var(--go);stroke-width:2.4}
          .cpt{fill:var(--go);stroke:var(--surface);stroke-width:1.5}
          .clbl{fill:var(--text-muted);font-size:11px}
        </style>
        ${grid}
        <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" class="cdiag"/>
        <path d="${path}" class="cline"/>${pts}
        <text x="${X(0.5)}" y="${H - 6}" text-anchor="middle" class="clbl">Prediction</text>
        <text x="12" y="${Y(0.5)}" text-anchor="middle" class="clbl" transform="rotate(-90 12 ${Y(0.5)})">Reality</text>
      </svg>`;
  }

  document.addEventListener("DOMContentLoaded", init);
})();

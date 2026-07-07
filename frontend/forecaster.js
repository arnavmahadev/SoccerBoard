// Forecaster — clean, bracket-first. Dependency-free; talks to the FastAPI backend.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, o) => fetch(p, o).then((r) => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });
  const pct = (x, dp) => (x == null ? "·" : (x * 100).toFixed(dp || 0) + "%");
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];
  function niceDate(iso) { const [y, m, d] = iso.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; }

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

  const state = { comp: null, teams: [], bracket: null, view: "prediction", tab: "live", adjust: {}, openNews: new Set(), perf: {}, openForm: new Set() };

  async function init() {
    let comps;
    try { comps = await api("/forecaster/competitions"); }
    catch (e) { $("asof").textContent = "Forecaster artifacts not built. Run: python -m forecaster.build_artifacts"; return; }
    state.comp = comps[0].id;
    $("comp-title").textContent = comps[0].name + " Forecast";

    const t = await api("/forecaster/teams?competition=" + state.comp);
    state.teams = t.teams;
    fillSelect($("h-home"), state.teams, pick(state.teams, "Argentina", 0));
    fillSelect($("h-away"), state.teams, pick(state.teams, "Brazil", 1));
    ["h-home", "h-away"].forEach((id) => $(id).addEventListener("change", headToHead));

    $("bk-toggle").querySelectorAll("button").forEach((btn) => {
      btn.onclick = () => {
        state.view = btn.dataset.view;
        $("bk-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
        renderBracket();
      };
    });

    // Tabs control title odds only — switching reloads odds, not the bracket.
    $("fc-tabs").querySelectorAll(".fc-tab").forEach((btn) => {
      btn.onclick = async () => {
        state.tab = btn.dataset.tab;
        $("fc-tabs").querySelectorAll(".fc-tab").forEach((b) => b.classList.toggle("active", b === btn));
        try { await loadOdds(); }
        catch (e) { console.error("Tab load failed:", state.tab, e); }
      };
    });

    // Delegated once: clicking a team's "news" tag expands its detail panel.
    // (#odds survives innerHTML refreshes, so open panels persist via state.)
    $("odds").addEventListener("click", (e) => {
      const btn = e.target.closest(".news-chip");
      if (!btn) return;
      const team = btn.dataset.team;
      const open = !state.openNews.has(team);
      open ? state.openNews.add(team) : state.openNews.delete(team);
      btn.setAttribute("aria-expanded", open);
      btn.querySelector(".chev").textContent = open ? "▴" : "▾";
      btn.closest(".odds-item").querySelector(".news-panel").classList.toggle("open", open);
    });

    // Same delegation for the "form" tag (in-tournament performance nudge).
    $("odds").addEventListener("click", (e) => {
      const btn = e.target.closest(".form-chip");
      if (!btn) return;
      const team = btn.dataset.team;
      const open = !state.openForm.has(team);
      open ? state.openForm.add(team) : state.openForm.delete(team);
      btn.setAttribute("aria-expanded", open);
      btn.querySelector(".chev").textContent = open ? "▴" : "▾";
      btn.closest(".odds-item").querySelector(".form-panel").classList.toggle("open", open);
    });

    await loadAdjustments(); // before the odds render, so adjusted teams get flagged
    await Promise.all([loadBracket(), loadOdds(), headToHead(), loadGroups(), loadAccuracy()]);
    setInterval(() => { loadBracket(); loadOdds(); }, 120000); // refresh as games are played
  }

  // ---- news overlay (manual injury/squad adjustments) ---------------------
  async function loadAdjustments() {
    let a;
    try { a = await api("/forecaster/adjustments?competition=" + state.comp); }
    catch (e) { return; }
    state.adjust = {};
    (a.teams || []).forEach((r) => { state.adjust[r.team] = r; });
    renderAdjustments(a.teams || []);
  }

  // the little clickable "news" tag next to an adjusted team in the odds list
  function adjChip(team, open) {
    const r = state.adjust[team];
    if (!r) return "";
    // News chips are always red: the tag colour marks an off-pitch absence
    // (injury/suspension), not the size or direction of the rating hit.
    return ` <button class="news-chip" data-team="${esc(team)}" ` +
      `aria-expanded="${open}" title="Injury news factored into these odds. Click for details.">` +
      `news <span class="chev">${open ? "▴" : "▾"}</span></button>`;
  }

  // the little clickable "form" tag: shown when a team's rating was nudged by how
  // it has actually played in the tournament so far (over/under-performance).
  function formChip(team, open) {
    const r = state.perf[team];
    if (!r) return "";
    // Form chips are always green: the tag colour marks an on-pitch form nudge by
    // type, not direction. The expandable panel spells out whether the team has
    // over- or under-performed, so no signal is lost by fixing the colour.
    return ` <button class="form-chip" data-team="${esc(team)}" ` +
      `aria-expanded="${open}" title="Tournament form factored into these odds. Click for details.">` +
      `form <span class="chev">${open ? "▴" : "▾"}</span></button>`;
  }

  // plain-English read of a team's performance nudge, with the actual goal totals
  // vs what the model expected across their tournament games. Attack and defense
  // are described independently, so an over-performing attack alongside a leaky
  // defense reads as exactly that.
  function formPanel(r) {
    const games = `${r.matches} tournament ${r.matches === 1 ? "game" : "games"}`;
    const goals = (n) => `${n} ${n === 1 ? "goal" : "goals"}`;

    // One side of the ball: actual goals vs the model's expected goals, and how
    // that moved the rating. `delta` is the applied nudge (>0 = rating up) and is
    // guaranteed to agree in sign with (actual - expected), since both come from
    // the same pooled tournament totals.
    const clause = (verb, actual, expected, delta, rating) => {
      const gap = actual - expected;
      if (Math.abs(gap) < 0.35)
        return `${verb} <b>${goals(actual)}</b>, right in line with their <b>${expected.toFixed(1)} expected goals</b>, so their ${rating} rating stays put`;
      const side = gap > 0 ? "more" : "fewer";
      const move = delta > 0
        ? `so their ${rating} rating gets a lift`
        : `so their ${rating} rating dips`;
      return `${verb} <b>${goals(actual)}</b>, ${Math.abs(gap).toFixed(1)} ${side} than their ` +
        `<b>${expected.toFixed(1)} expected goals</b>, ${move}`;
    };

    const atk = clause("scored", r.gf, r.xgf, r.att_delta, "attack");
    const def = clause("conceded", r.ga, r.xga, r.def_delta, "defense");

    // Net movement across both sides. A side only counts as a mover if its goal
    // gap cleared the same threshold the clause used to say "in line", so the net
    // line never contradicts what each clause just showed.
    const atkNear = Math.abs(r.gf - r.xgf) < 0.35;
    const defNear = Math.abs(r.ga - r.xga) < 0.35;
    const atkUp = !atkNear && r.att_delta > 0, atkDn = !atkNear && r.att_delta < 0;
    const defUp = !defNear && r.def_delta > 0, defDn = !defNear && r.def_delta < 0;
    let net;
    if ((atkUp && defDn) || (atkDn && defUp)) {
      const up = (r.att_delta + r.def_delta) >= 0;
      net = `Those two tug in opposite directions, so on balance their rating edges <b>${up ? "up" : "down"}</b>.`;
    } else if (atkUp || defUp) {
      net = `Put it together and their rating ticks <b>up</b>.`;
    } else if (atkDn || defDn) {
      net = `Put it together and their rating ticks <b>down</b>.`;
    } else {
      net = `Put it together and their rating barely moves.`;
    }

    return `<div class="form-panel${state.openForm.has(r.team) ? " open" : ""}">` +
      `<p class="news-impact">Over ${games} they've ${atk}. At the back they've ${def}. ${net}</p></div>`;
  }

  const SPARSE_THRESHOLD = 8;

  // plain-English summary of which players are out and by how much
  // Only counts players with enough appearances to trust their delta.
  function impactSentence(r) {
    const reliable = (r.items || []).filter((it) => it.covered && it.n_matches >= SPARSE_THRESHOLD);
    const totalAtt = reliable.reduce((s, it) => s + (it.att_delta || 0), 0);
    const totalDef = reliable.reduce((s, it) => s + (it.def_delta || 0), 0);
    const bits = [];
    if (totalAtt > 0.005) bits.push(`<b>${totalAtt.toFixed(2)} fewer expected goals scored per game</b>`);
    if (totalDef > 0.005) bits.push(`<b>${totalDef.toFixed(2)} more expected goals conceded per game</b>`);
    if (!bits.length) return "";
    return `<p class="news-impact">Based on StatsBomb lineup data, the model estimates ` +
      `${bits.join(" and ")}.</p>`;
  }

  // the expandable detail panel under an adjusted team's row
  function newsPanel(r) {
    const items = (r.items || []).map((it) => {
      let note = "";
      if (!it.covered) {
        note = ` <span class="news-sparse">(no training data)</span>`;
      } else if (it.n_matches < SPARSE_THRESHOLD) {
        note = ` <span class="news-sparse">(only ${it.n_matches} appearances in training data, so impact is not estimated)</span>`;
      }
      return `<li><b>${esc(it.player)}</b> ${esc(it.issue)}${note}</li>`;
    }).join("");
    const links = (r.sources || []).map((s) => s.url
      ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.label)}</a>`
      : esc(s.label)).join(", ");
    return `<div class="news-panel${state.openNews.has(r.team) ? " open" : ""}">` +
      (items ? `<ul class="news-items">${items}</ul>` : "") +
      impactSentence(r) +
      (links ? `<p class="news-src">More on this: ${links}</p>` : "") +
      `</div>`;
  }

  function renderAdjustments(rows) {
    const el = $("adjust-note");
    if (!el) return;
    el.innerHTML = rows.length
      ? `These odds factor in the latest injury news. Teams with a ` +
        `<span class="news-chip static">news ▾</span> tag have someone out. Each missing ` +
        `player's historical contribution is looked up from StatsBomb lineup data and subtracted ` +
        `from that team's rating. Tap a tag to see who's out and by how much.`
      : "";
  }

  const pick = (arr, want, i) => (arr.includes(want) ? want : arr[i] || arr[0]);
  function fillSelect(sel, teams, chosen) {
    sel.innerHTML = teams.map((t) => `<option${t === chosen ? " selected" : ""}>${esc(t)}</option>`).join("");
  }

  // ---- bracket (independent of the odds tab) --------------------------------
  async function loadBracket() {
    const b = await api("/forecaster/bracket?competition=" + state.comp);
    state.bracket = b;
    const n = b.settled_count;
    $("asof").textContent = `Updated through ${niceDate(b.as_of)}. ${n} knockout game${n === 1 ? "" : "s"} played so far.`;
    renderBracket();
  }

  // ---- odds (controlled by the pretournament / live tab) -------------------
  async function loadOdds() {
    const sim = await api("/forecaster/simulation?competition=" + state.comp + "&mode=" + state.tab);
    renderOdds(sim);
  }

  function renderOdds(sim) {
    state.perf = {};
    (sim.performance || []).forEach((r) => { state.perf[r.team] = r; });
    // Injury/squad news only bends the live odds — the pre-tournament odds are the
    // frozen pre-tournament ratings with nothing subtracted — so hide the news
    // chips, panels and legend on the pre-tournament tab (form is already absent
    // there because the pre-tournament sim reports no performance rows).
    const showNews = state.tab === "live";
    $("adjust-note").style.display = showNews ? "" : "none";
    const ranked = sim.teams.slice().sort((a, b) => b.champion - a.champion);
    // On the live tab only teams still alive can win the title, so drop eliminated
    // teams (champion == 0) instead of padding the list to a fixed 10 with sides that
    // are already out. The pre-knockout tab keeps a full field of 10.
    const top = (showNews ? ranked.filter((r) => r.champion > 0) : ranked).slice(0, 10);
    const max = top[0].champion || 1;
    $("odds").innerHTML = top.map((r, i) => {
      const open = state.openNews.has(r.team);
      const fopen = state.openForm.has(r.team);
      const panel = (showNews && state.adjust[r.team]) ? newsPanel(state.adjust[r.team]) : "";
      const fpanel = state.perf[r.team] ? formPanel(state.perf[r.team]) : "";
      return `<div class="odds-item">
        <div class="odds-row ${i === 0 ? "lead-team" : ""}">
          <span class="odds-rank">${i + 1}</span>
          <span class="odds-name">${flag(r.team)}${esc(r.team)}</span>
          <span class="odds-news">${showNews ? adjChip(r.team, open) : ""}</span>
          <span class="odds-form">${formChip(r.team, fopen)}</span>
          <span class="odds-track"><span class="odds-fill" style="width:${(r.champion / max) * 100}%"></span></span>
          <span class="odds-val">${pct(r.champion, r.champion < 0.1 ? 1 : 0)}</span>
        </div>${panel}${fpanel}
      </div>`;
    }).join("");
  }

  // ---- bracket rendering --------------------------------------------------
  function bkRow(name, meta, cls) {
    return `<div class="bk-row ${cls || ""}">
      <span class="bk-name">${namedShort(name)}</span>
      <span class="bk-meta">${meta || ""}</span></div>`;
  }

  function predMatch(m) {
    // A settled tie (already played) is locked to its real score; only ties still
    // to come show a predicted win percentage.
    if (m.settled) {
      const aWin = m.winner === m.a;
      return `<div class="bk-match"><div class="bk-box">
        ${bkRow(m.a, `${m.score[0]}`, aWin ? "win" : "")}
        ${bkRow(m.b, `${m.score[1]}`, aWin ? "" : "win")}
      </div></div>`;
    }
    const aWin = m.winner === m.a, bWin = m.winner === m.b;
    return `<div class="bk-match"><div class="bk-box">
      ${bkRow(m.a, pct(m.prob_a), aWin ? "win" : "")}
      ${bkRow(m.b, pct(m.prob_b), bWin ? "win" : "")}
    </div></div>`;
  }

  function actualMatch(m) {
    if (!m.settled) {
      // not played yet — show the matchup (or TBD if a feeder isn't decided)
      const cls = (t) => (t ? "" : "tbd");
      return `<div class="bk-match"><div class="bk-box">
        ${bkRow(m.a, "", cls(m.a))}
        ${bkRow(m.b, "", cls(m.b))}
      </div></div>`;
    }
    const aWin = m.winner === m.a;
    const mark = m.correct ? `<span class="mark hit">✓</span>` : `<span class="mark miss">✗</span>`;
    const sa = `${aWin ? mark + " " : ""}${m.score[0]}`;
    const sb = `${!aWin ? mark + " " : ""}${m.score[1]}`;
    return `<div class="bk-match"><div class="bk-box">
      ${bkRow(m.a, sa, aWin ? "win" : "")}
      ${bkRow(m.b, sb, aWin ? "" : "win")}
    </div></div>`;
  }

  function renderBracket() {
    const b = state.bracket;
    if (!b) return;
    const data = b[state.view];
    if (!data) return;
    const isPred = state.view !== "actual";

    if (state.view === "pretournament_prediction") {
      $("bk-note").innerHTML = `Pre-tournament predictions using base team ratings, before any games were played and with no injury adjustments. Predicted champion: <b>${esc(data.champion)}</b>.`;
    } else if (state.view === "prediction") {
      $("bk-note").innerHTML = `Confirmed results are locked in; games still to come are predicted using current ratings with injury and in-tournament form adjustments applied. Predicted champion: <b>${esc(data.champion)}</b>. Each percentage is that team's chance of winning that specific upcoming game.`;
    } else {
      const c = data.correct, d = data.decided;
      $("bk-note").innerHTML = d
        ? `The model got <b>${c} of ${d}</b> knockout game${d === 1 ? "" : "s"} right so far. A ✓ marks each correct pick.`
        : `No knockout games have been played yet. This will update as they go.`;
    }

    const renderMatch = isPred ? predMatch : actualMatch;
    const byRound = {};
    data.rounds.forEach((r) => { byRound[r.round] = r; });

    // Mirrored two-sided bracket: left half flows inward, right half mirrors.
    // Splitting each round's matches down the middle gives the two sides.
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
        <div class="bk-list">${ms.map(renderMatch).join("")}</div></div>`;
    };

    const flow = ["round_of_32", "round_of_16", "quarterfinal", "semifinal"];
    const left  = flow.map((r) => halfCol(r, "l")).join("");
    const right = flow.slice().reverse().map((r) => halfCol(r, "r")).join("");

    const finalRound = byRound.final;
    const finalM = finalRound && finalRound.matches[0];
    const tp = data.third_place || {};

    const champTbd = !data.champion;
    const champLabel = state.view === "actual" ? "Champion" : "Predicted champion";
    const center = `<div class="bk-col center-col no-join">
      <div class="bk-final-content">
        <div class="bk-champ ${champTbd ? "tbd" : ""}">
          <span class="trophy" aria-hidden="true">🏆</span>
          <span class="cl">${champLabel}</span>
          <span class="cn">${champTbd ? "TBD" : named(data.champion)}</span>
        </div>
        <div class="bk-center-node">
          <div class="bk-head">${label("final") || "Final"}</div>
          ${finalM ? renderMatch(finalM) : ""}
        </div>
        <div class="bk-bronze">
          <div class="bk-head">Third place</div>
          ${tp.a ? renderMatch(tp) : `<div class="bk-match"><div class="bk-box">
            ${bkRow(null, "", "tbd")}${bkRow(null, "", "tbd")}
          </div></div>`}
        </div>
      </div></div>`;

    const bk = $("bracket");
    bk.classList.add("two");
    bk.innerHTML = left + center + right;
  }

  // ---- head to head -------------------------------------------------------
  async function headToHead() {
    const home = $("h-home").value, away = $("h-away").value;
    const r = await api("/forecaster/match", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ competition: state.comp, home, away, neutral: true }),
    });
    const bar = $("h2h-bar");
    bar.querySelector(".home").style.width = r.prob_home * 100 + "%";
    bar.querySelector(".draw").style.width = r.prob_draw * 100 + "%";
    bar.querySelector(".away").style.width = r.prob_away * 100 + "%";
    $("h-home-p").textContent = pct(r.prob_home);
    $("h-draw-p").textContent = pct(r.prob_draw);
    $("h-away-p").textContent = pct(r.prob_away);
    $("h-home-n").innerHTML = flag(r.home) + esc(r.home);
    $("h-away-n").innerHTML = flag(r.away) + esc(r.away);
    $("h-score").textContent = `${r.most_likely[0]}–${r.most_likely[1]}`;
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
      `is rated. For injuries and suspensions, a second model trained on StatsBomb lineup data ` +
      `from WC 2018 and 2022, the Euros, Copa América, and AFCON estimates each player's ` +
      `contribution to their team's attack and defence. When a player is listed as out, that ` +
      `contribution gets subtracted from their team's rating and flagged under the title odds. ` +
      `Live results come from a public community dataset, so they can sometimes be a few hours behind.`;
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

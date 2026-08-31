// OneBrain dashboard: pure render functions (metrics JSON in, HTML/SVG
// string out). Nothing in this file touches the DOM, the network, or
// storage — app.js does the wiring. Purity is the testing story (ADR 0005:
// no Node test runner in this repo), and it keeps every view a function of
// exactly one metrics document.
//
// Field names follow docs/product.md §1. The schema is additive-stable and
// optionals may be absent: every accessor degrades to "—", never throws.
"use strict";

const ObRender = (() => {
  // ---- tolerant accessors & formatters ----------------------------------

  /** First present (non-null) property among `keys`, else undefined. */
  function pick(obj, ...keys) {
    if (!obj || typeof obj !== "object") return undefined;
    for (const k of keys) {
      if (obj[k] !== undefined && obj[k] !== null) return obj[k];
    }
    return undefined;
  }

  function num(v) {
    return typeof v === "number" && isFinite(v) ? v : undefined;
  }

  /** HTML-escape; every peer/model/user-influenced string goes through it. */
  function esc(s) {
    return String(s === undefined || s === null ? "" : s).replace(
      /[&<>"']/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
    );
  }

  const GIB = 1024 * 1024 * 1024;

  function fmtGiB(bytes) {
    const b = num(bytes);
    if (b === undefined) return "—";
    return (b / GIB).toFixed(1) + " GiB";
  }

  function fmtMs(ms) {
    const m = num(ms);
    if (m === undefined) return "—";
    if (m < 1) return "<1 ms";
    if (m >= 10000) return (m / 1000).toFixed(1) + " s";
    return Math.round(m) + " ms";
  }

  function fmtMbps(mbps) {
    const m = num(mbps);
    if (m === undefined) return "—";
    if (m >= 1000) return (m / 1000).toFixed(1) + " Gbps";
    return Math.round(m) + " Mbps";
  }

  function fmtTps(tps) {
    const t = num(tps);
    if (t === undefined) return "—";
    return (t >= 100 ? Math.round(t) : t.toFixed(1)) + " tok/s";
  }

  /** Loss may arrive as a fraction (0..1) or a percentage; show percent. */
  function fmtLoss(loss) {
    const l = num(loss);
    if (l === undefined || l <= 0) return "";
    const pct = l <= 1 ? l * 100 : l;
    return pct.toFixed(1) + "% loss";
  }

  /** Timestamps: RFC3339 string, unix seconds, or unix milliseconds. */
  function fmtTime(ts) {
    let d;
    if (typeof ts === "string") d = new Date(ts);
    else if (num(ts) !== undefined) d = new Date(ts > 1e12 ? ts : ts * 1000);
    if (!d || isNaN(d.getTime())) return "—";
    return d.toLocaleTimeString();
  }

  /** Memory as {usable, total} bytes from either short or _bytes names. */
  function memPair(mem) {
    return {
      usable: num(pick(mem, "usable", "usable_bytes")),
      total: num(pick(mem, "total", "total_bytes")),
    };
  }

  function stateOf(peer) {
    return String(pick(peer, "state") || "Unknown");
  }

  function stateClass(state) {
    const s = state.toLowerCase();
    if (s === "connected") return "state-connected";
    if (s === "suspect") return "state-suspect";
    if (s === "down") return "state-down";
    if (s === "draining") return "state-draining";
    return "state-unknown";
  }

  // ---- header: the capacity story (§1.6) --------------------------------

  /**
   * One honest line about pooled capacity. Only usable memory of this node
   * plus Connected peers counts — a Down peer contributes nothing today.
   */
  function capacityLine(m) {
    const peers = Array.isArray(m.peers) ? m.peers : [];
    const connected = peers.filter((p) => stateOf(p).toLowerCase() === "connected");
    // "0.0 GiB" would present an unknown as a measurement (§1.6 honesty):
    // only claim a number once at least one machine reported memory.
    let usable = 0;
    let measured = false;
    for (const mem of [pick(m.node, "memory"), ...connected.map((p) => pick(p, "memory"))]) {
      const u = memPair(mem).usable;
      if (u !== undefined) {
        usable += u;
        measured = true;
      }
    }
    const machines = 1 + connected.length;

    const count = machines + (machines === 1 ? " machine" : " machines");
    let line = measured
      ? fmtGiB(usable) + " usable memory pooled across " + count
      : count + " — memory not measured yet";
    const model = pick(m.plan, "model");
    if (model) line += " · serving " + esc(model);
    if (machines === 1) line += ". Pair another machine to pool more.";
    return line;
  }

  // ---- advisor (server-side findings, rendered verbatim) ----------------

  function sevClass(sev) {
    const s = String(sev || "").toLowerCase();
    if (s.startsWith("crit") || s === "error" || s === "high") return "sev-crit";
    if (s.startsWith("warn") || s === "medium") return "sev-warn";
    return "sev-info";
  }

  function advisorHtml(advisor) {
    const list = Array.isArray(advisor) ? advisor : [];
    if (list.length === 0) {
      return '<p class="empty">No findings. The advisor only speaks when a measurement gives it something to say.</p>';
    }
    const items = list
      .map((a) => {
        const sev = String(pick(a, "severity") || "info");
        return (
          '<li class="' + sevClass(sev) + '"><span class="sev">' +
          esc(sev.toLowerCase()) + "</span> " + esc(pick(a, "text")) + "</li>"
        );
      })
      .join("");
    return '<ul class="advisor-list">' + items + "</ul>";
  }

  // ---- topology: hub-and-spoke, self left, peers in a column right ------
  // (A radial ring puts link labels on top of node captions for vertical
  // links; left-to-right spokes keep every label in clear space and scale
  // to any peer count by growing downward.)

  function topologySvg(m) {
    const W = 640;
    const peers = Array.isArray(m.peers) ? m.peers : [];
    const selfName = pick(m.node, "name") || "this machine";
    const n = peers.length;
    const SPACING = 100;
    const H = Math.max(210, 70 + n * SPACING);
    const sx = n === 0 ? W / 2 : 120;
    const sy = H / 2 - (n === 0 ? 20 : 0);

    let out =
      '<svg viewBox="0 0 ' + W + " " + H + '" role="img" ' +
      'aria-label="Cluster topology" class="topo">';

    const px = W - 150;
    const firstY = H / 2 - ((n - 1) * SPACING) / 2;
    const pos = peers.map((_, i) => ({ x: px, y: firstY + i * SPACING }));

    // Links first so nodes draw on top of them.
    peers.forEach((p, i) => {
      const { x, y } = pos[i];
      const cls = stateClass(stateOf(p));
      out +=
        '<line class="link ' + cls + '" x1="' + sx + '" y1="' + sy +
        '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>';
      // Label the link just above its midpoint: RTT · bandwidth (· loss).
      const parts = [fmtMs(pick(p, "rtt_ms")), fmtMbps(pick(p, "bandwidth_mbps"))];
      const loss = fmtLoss(pick(p, "loss"));
      if (loss) parts.push(loss);
      const mx = (sx + x) / 2;
      const my = (sy + y) / 2 - 8;
      out +=
        '<text class="link-label" x="' + mx.toFixed(1) + '" y="' + my.toFixed(1) +
        '" text-anchor="middle">' + esc(parts.join(" · ")) + "</text>";
    });

    function nodeGlyph(x, y, name, cls, sub) {
      return (
        '<g class="node ' + cls + '"><circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) +
        '" r="26"/><text x="' + x.toFixed(1) + '" y="' + (y + 44).toFixed(1) +
        '" text-anchor="middle" class="node-name">' + esc(name) + "</text>" +
        (sub
          ? '<text x="' + x.toFixed(1) + '" y="' + (y + 60).toFixed(1) +
            '" text-anchor="middle" class="node-sub">' + esc(sub) + "</text>"
          : "") +
        "</g>"
      );
    }

    out += nodeGlyph(sx, sy, selfName, "state-self", "this machine");
    peers.forEach((p, i) => {
      out += nodeGlyph(pos[i].x, pos[i].y, pick(p, "name") || "?", stateClass(stateOf(p)), stateOf(p));
    });
    out += "</svg>";

    if (peers.length === 0) {
      out +=
        '<p class="empty">Just this machine so far. Pairing another adds its memory to the pool.</p>';
    }
    return out;
  }

  // ---- plan: layer ranges per node, in stage order ----------------------

  /** Layer range as {start, end} from any of the plausible §1 spellings. */
  function layerRange(a) {
    const arr = pick(a, "layers", "layer_range");
    if (Array.isArray(arr) && arr.length >= 2) {
      return { start: num(arr[0]), end: num(arr[1]) };
    }
    if (arr && typeof arr === "object") {
      return { start: num(pick(arr, "start")), end: num(pick(arr, "end")) };
    }
    return {
      start: num(pick(a, "layer_start", "first_layer")),
      end: num(pick(a, "layer_end", "last_layer")),
    };
  }

  function planHtml(plan) {
    if (!plan) {
      return '<p class="empty">No active plan — load a model with <code>onebrain run &lt;model&gt;</code>.</p>';
    }
    const assignments = (Array.isArray(plan.assignments) ? plan.assignments : [])
      .map((a, i) => ({
        node: pick(a, "node", "name") || "?",
        stage: num(pick(a, "stage", "stage_index", "order")),
        range: layerRange(a),
        idx: i,
      }))
      .sort((x, y) => (x.stage ?? x.idx) - (y.stage ?? y.idx));

    let head =
      '<p class="plan-head"><strong>' + esc(pick(plan, "model") || "?") + "</strong>" +
      (pick(plan, "strategy") ? " · " + esc(plan.strategy) : "") +
      (num(pick(plan, "epoch")) !== undefined ? " · epoch " + plan.epoch : "") +
      "</p>";
    const tpt = num(pick(plan, "predicted_tpt_ms"));
    const pre = num(pick(plan, "predicted_prefill_ms"));
    if (tpt !== undefined || pre !== undefined) {
      const bits = [];
      if (tpt !== undefined) bits.push("≈" + fmtMs(tpt) + "/token");
      if (pre !== undefined) bits.push("prefill ≈" + fmtMs(pre));
      head += '<p class="plan-pred">predicted ' + bits.join(" · ") + "</p>";
    }

    if (assignments.length === 0) return head + '<p class="empty">Plan has no assignments.</p>';

    // Scale every bar against the largest layer index in the plan so the
    // rows read as one pipeline.
    const totalLayers = Math.max(
      1,
      ...assignments.map((a) => a.range.end ?? 0)
    );
    const rows = assignments
      .map((a, i) => {
        const s = a.range.start ?? 0;
        const e = a.range.end ?? s;
        const left = (100 * s) / totalLayers;
        const width = Math.max(2, (100 * (e - s)) / totalLayers);
        const label =
          a.range.start !== undefined && a.range.end !== undefined
            ? "L" + s + "–" + e
            : "?";
        return (
          '<div class="plan-row"><span class="plan-node">' +
          (i + 1) + ". " + esc(a.node) + "</span>" +
          '<div class="plan-track"><div class="plan-range" style="left:' +
          left.toFixed(1) + "%;width:" + width.toFixed(1) + '%">' +
          esc(label) + "</div></div></div>"
        );
      })
      .join("");
    return head + '<div class="plan-rows">' + rows + "</div>";
  }

  // ---- per-node cards ---------------------------------------------------

  function badge(text, cls) {
    return '<span class="badge ' + cls + '">' + esc(text) + "</span>";
  }

  function cardHtml(name, stateLabel, cls, mem, profile, extras) {
    const { usable, total } = memPair(mem);
    const pct =
      usable !== undefined && total ? Math.min(100, (100 * usable) / total) : 0;
    const memLine =
      usable !== undefined || total !== undefined
        ? fmtGiB(usable) + " of " + fmtGiB(total) + " usable"
        : "memory —";
    const dec = pick(profile, "decode_tps");
    const pre = pick(profile, "prefill_tps");
    const perf =
      num(dec) !== undefined || num(pre) !== undefined
        ? "measured: " + fmtTps(dec) + " decode · " + fmtTps(pre) + " prefill"
        : "not profiled yet";
    return (
      '<article class="card ' + cls + '"><h3>' + esc(name) +
      ' <span class="node-state">' + esc(stateLabel) + "</span></h3>" +
      '<div class="membar"><div class="membar-fill" style="width:' +
      pct.toFixed(1) + '%"></div></div>' +
      '<p class="mem-line">' + memLine + "</p>" +
      '<p class="perf-line">' + perf + "</p>" +
      (extras ? '<p class="badges">' + extras + "</p>" : "") +
      "</article>"
    );
  }

  /** Battery/draining/sleep/version badges shared by self and peer cards. */
  function nodeBadges(n) {
    const out = [];
    const battery = pick(n, "battery");
    const draining = pick(n, "draining", "on_battery_draining") === true ||
      pick(battery, "draining") === true;
    if (draining) out.push(badge("on battery · draining", "badge-warn"));
    else if (battery !== undefined && battery !== null) {
      const pctv = num(battery) !== undefined ? num(battery) : num(pick(battery, "percent"));
      out.push(badge(pctv !== undefined ? "battery " + Math.round(pctv) + "%" : "on battery", "badge-info"));
    }
    if (pick(n, "sleep_inhibited") === true) out.push(badge("sleep inhibited", "badge-info"));
    const ver = pick(n, "version");
    const build = pick(n, "engine_build_id", "engine_build");
    if (ver || build) {
      out.push(
        badge(
          (ver ? "v" + ver : "") + (ver && build ? " · " : "") +
          (build ? "engine " + String(build).slice(0, 12) : ""),
          "badge-dim"
        )
      );
    }
    return out.join(" ");
  }

  function nodeCardsHtml(m) {
    const n = m.node || {};
    let out = cardHtml(
      pick(n, "name") || "this machine",
      pick(n, "platform") || "this machine",
      "state-self",
      pick(n, "memory"),
      pick(n, "profile"),
      nodeBadges(n)
    );
    for (const p of Array.isArray(m.peers) ? m.peers : []) {
      const st = stateOf(p);
      const extras = [nodeBadges(p)];
      const idp = pick(p, "id_prefix");
      if (idp) extras.push(badge(String(idp), "badge-dim"));
      out += cardHtml(
        pick(p, "name") || "?",
        st,
        stateClass(st),
        pick(p, "memory"),
        pick(p, "profile"),
        extras.filter(Boolean).join(" ")
      );
    }
    return out;
  }

  // ---- request log ------------------------------------------------------

  function requestsHtml(requests) {
    const list = Array.isArray(requests) ? requests : [];
    if (list.length === 0) {
      return '<p class="empty">No requests yet — point a client at this daemon\'s API and they will appear here (never any prompt text).</p>';
    }
    // Ring buffer arrives oldest-first; show newest on top when timestamps
    // allow, otherwise just reverse the given order.
    const rows = list
      .slice()
      .reverse()
      .map((r) => {
        const drafted = num(pick(r, "drafted"));
        const accepted = num(pick(r, "accepted"));
        const draft =
          drafted !== undefined ? accepted + "/" + drafted : "—";
        return (
          "<tr><td>" + esc(fmtTime(pick(r, "timestamp", "ts"))) + "</td>" +
          "<td>" + esc(pick(r, "dialect", "api_dialect") || "—") + "</td>" +
          "<td>" + esc(pick(r, "model") || "—") + "</td>" +
          "<td>" + (num(pick(r, "prompt_tokens")) ?? "—") + " → " +
          (num(pick(r, "completion_tokens")) ?? "—") + "</td>" +
          "<td>" + fmtMs(pick(r, "ttft_ms")) + "</td>" +
          "<td>" + fmtMs(pick(r, "prefill_ms")) + "</td>" +
          "<td>" + fmtMs(pick(r, "decode_ms")) + "</td>" +
          "<td>" + esc(draft) + "</td>" +
          "<td>" + esc(pick(r, "finish_reason") || "—") + "</td></tr>"
        );
      })
      .join("");
    return (
      '<div class="table-wrap"><table class="req-table"><thead><tr>' +
      "<th>Time</th><th>API</th><th>Model</th><th>Tokens</th><th>TTFT</th>" +
      "<th>Prefill</th><th>Decode</th><th>Draft acc.</th><th>Finish</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    );
  }

  return {
    pick,
    esc,
    fmtGiB,
    fmtMs,
    fmtMbps,
    fmtTps,
    fmtLoss,
    fmtTime,
    capacityLine,
    advisorHtml,
    topologySvg,
    planHtml,
    nodeCardsHtml,
    requestsHtml,
  };
})();

// No module system on this page (ADR 0005); expose the namespace for
// app.js. The typeof guard keeps the file loadable in any JS engine for
// ad-hoc testing.
if (typeof window !== "undefined") window.ObRender = ObRender;

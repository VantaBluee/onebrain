// OneBrain dashboard: wiring layer. Owns the token flow, the 2 s metrics
// poll, and pushing ObRender output into the DOM. All rendering logic lives
// in render.js (pure functions) — keep it that way (ADR 0005).
"use strict";

(() => {
  const METRICS_URL = "/api/internal/metrics";
  const TOKEN_KEY = "onebrain.dash.token";
  const POLL_MS = 2000;

  const $ = (id) => document.getElementById(id);
  const R = window.ObRender;

  // localStorage can throw (private windows, storage disabled); the
  // dashboard must still work — the user just re-pastes per visit.
  function getToken() {
    try {
      return localStorage.getItem(TOKEN_KEY) || "";
    } catch {
      return "";
    }
  }
  function setToken(t) {
    try {
      localStorage.setItem(TOKEN_KEY, t);
    } catch {
      /* per-visit token only */
    }
  }
  function clearToken() {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing stored */
    }
  }

  let sessionToken = getToken();
  let hasRendered = false;

  function setStatus(text, kind) {
    const el = $("status-line");
    if (!text) {
      el.hidden = true;
      return;
    }
    el.textContent = text;
    el.className = "status " + (kind || "");
    el.hidden = false;
  }

  function showTokenPanel(errorText) {
    $("token-panel").hidden = false;
    const err = $("token-error");
    if (errorText) {
      err.textContent = errorText;
      err.hidden = false;
    } else {
      err.hidden = true;
    }
    // Keep whatever data we already drew visible-but-dim behind the prompt.
    document.body.classList.add("stale");
  }

  function hideTokenPanel() {
    $("token-panel").hidden = true;
  }

  function render(m) {
    $("capacity-line").innerHTML = R.capacityLine(m);
    $("advisor").innerHTML = R.advisorHtml(m.advisor);
    $("topology").innerHTML = R.topologySvg(m);
    $("plan").innerHTML = R.planHtml(m.plan);
    $("nodes").innerHTML = R.nodeCardsHtml(m);
    $("requests").innerHTML = R.requestsHtml(m.requests);
    $("dash-main").hidden = false;
    hasRendered = true;
    document.body.classList.remove("stale");
  }

  function onUnreachable(detail) {
    document.body.classList.add("stale");
    setStatus(
      "Daemon unreachable" + (detail ? " (" + detail + ")" : "") +
        " — is `onebrain up` running? Retrying every 2 s.",
      "err"
    );
    if (!hasRendered) {
      $("capacity-line").textContent = "Waiting for the daemon…";
    }
  }

  async function poll() {
    const headers = {};
    if (sessionToken) headers["Authorization"] = "Bearer " + sessionToken;

    let resp;
    try {
      resp = await fetch(METRICS_URL, { headers, cache: "no-store" });
    } catch {
      onUnreachable();
      return;
    }

    if (resp.status === 401 || resp.status === 403) {
      // Either no token yet (first visit, non-loopback) or a stale one.
      // A rejected token is cleared so a wrong paste can't wedge the page.
      const hadToken = Boolean(sessionToken);
      if (hadToken) {
        clearToken();
        sessionToken = "";
      }
      showTokenPanel(
        hadToken
          ? "The daemon rejected that token (HTTP " + resp.status +
            "). Paste the current one — `onebrain status` prints it."
          : ""
      );
      setStatus("Waiting for an API token.", "warn");
      return;
    }

    if (!resp.ok) {
      onUnreachable("HTTP " + resp.status);
      return;
    }

    let metrics;
    try {
      metrics = await resp.json();
    } catch {
      onUnreachable("unparseable response");
      return;
    }

    hideTokenPanel();
    try {
      render(metrics);
    } catch (e) {
      // A render bug must not kill the poll loop; surface it honestly.
      setStatus("Render error: " + (e && e.message ? e.message : e), "err");
      return;
    }
    setStatus("Updated " + new Date().toLocaleTimeString(), "ok");
  }

  function loop() {
    // setTimeout chain (not setInterval): a slow request never stacks a
    // second one behind it.
    poll().finally(() => setTimeout(loop, POLL_MS));
  }

  $("token-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("token-input").value.trim();
    if (!t) return;
    setToken(t);
    sessionToken = t;
    $("token-input").value = "";
    hideTokenPanel();
    setStatus("Connecting…", "");
    poll();
  });

  // First poll runs immediately — with no token it either succeeds (the
  // daemon's loopback exemption, §M1 rules) or 401s into the token prompt.
  loop();
})();

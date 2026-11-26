/* Dependency-free charts and small UI behaviours for the research dashboard. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function json(id) {
    var node = document.getElementById(id);
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (e) { return null; }
  }

  function el(tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(node);
    return node;
  }

  function drawChart(svg, labels, series, options) {
    if (!svg || !labels || !labels.length) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var box = (svg.getAttribute("viewBox") || "0 0 640 300").split(/\s+/);
    var w = parseFloat(box[2]), h = parseFloat(box[3]);
    var m = { l: 50, r: 18, t: 18, b: 40 };
    var pw = w - m.l - m.r, ph = h - m.t - m.b;
    var all = [];
    series.forEach(function (s) {
      s.values.forEach(function (v) { if (typeof v === "number" && isFinite(v)) all.push(v); });
    });
    if (!all.length) return;
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (hi - lo < 1e-9) { lo -= 0.1; hi += 0.1; }
    var pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    var x = function (i) { return m.l + (labels.length === 1 ? pw / 2 : pw * i / (labels.length - 1)); };
    var y = function (v) { return m.t + ph - (v - lo) / (hi - lo) * ph; };

    // Horizontal grid and y-axis labels.
    for (var g = 0; g <= 4; g++) {
      var gv = lo + (hi - lo) * g / 4, gy = y(gv);
      el("line", { x1: m.l, y1: gy, x2: w - m.r, y2: gy,
        stroke: "#e3e9f2", "stroke-width": 1 }, svg);
      var t = el("text", { x: m.l - 8, y: gy + 4, "text-anchor": "end",
        fill: "#61708a", "font-size": 11 }, svg);
      t.textContent = gv.toFixed(2);
    }
    el("line", { x1: m.l, y1: m.t + ph, x2: w - m.r, y2: m.t + ph,
      stroke: "#cbd5e1", "stroke-width": 1 }, svg);

    // X-axis labels.
    labels.forEach(function (label, i) {
      var tx = el("text", { x: x(i), y: h - 12, "text-anchor": "middle",
        fill: "#61708a", "font-size": 11 }, svg);
      tx.textContent = label;
    });

    // Series lines and points.
    series.forEach(function (s, si) {
      var color = s.color || "#2563eb";
      var pts = s.values.map(function (v, i) {
        return typeof v === "number" && isFinite(v) ? x(i) + "," + y(v) : null;
      }).filter(Boolean).join(" ");
      if (pts) el("polyline", { points: pts, fill: "none", stroke: color,
        "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
      s.values.forEach(function (v, i) {
        if (typeof v !== "number" || !isFinite(v)) return;
        el("circle", { cx: x(i), cy: y(v), r: 3.5, fill: color, stroke: "#fff",
          "stroke-width": 1.2 }, svg);
      });
    });

    // Highlight the chosen point, if provided.
    if (options && typeof options.bestIndex === "number") {
      var s0 = series[0], bv = s0 && s0.values[options.bestIndex];
      if (typeof bv === "number" && isFinite(bv)) {
        el("circle", { cx: x(options.bestIndex), cy: y(bv), r: 7, fill: "none",
          stroke: "#dc2626", "stroke-width": 2 }, svg);
      }
    }
  }

  function readData() {
    return { alpha: json("alpha-data"), noise: json("noise-data") };
  }

  function setupAlpha(data) {
    var select = document.getElementById("alpha-dataset");
    var svg = document.getElementById("stability-chart");
    var image = document.getElementById("alpha-figure");
    if (!select || !svg || !data) return;
    var names = Object.keys(data);

    function update() {
      var d = data[names[parseInt(select.value, 10) || 0]];
      if (!d) return;
      var vals = d.stability || [];
      var best = -1, bestVal = -Infinity;
      vals.forEach(function (v, i) {
        if (typeof v === "number" && isFinite(v) && v > bestVal) { bestVal = v; best = i; }
      });
      drawChart(svg, (d.alphas || []).map(function (a) { return String(a); }),
        [{ name: "Stability", values: vals, color: "#2563eb" }], { bestIndex: best });
      if (image && d.figure) image.setAttribute("src", "/figures/" + encodeURIComponent(d.figure));
    }
    select.addEventListener("change", update);
    update();
  }

  function setupNoise(data) {
    var select = document.getElementById("noise-dataset");
    var svg = document.getElementById("noise-chart");
    var legend = document.getElementById("noise-legend");
    if (!select || !svg || !data) return;
    var names = Object.keys(data);
    var colors = ["#2563eb", "#0f766e", "#b45309", "#7c3aed"];

    function update() {
      var d = data[names[parseInt(select.value, 10) || 0]];
      if (!d) return;
      var levels = (d.levels || []).map(String);
      var series = Object.keys(d.configs || {}).map(function (cfg, i) {
        return { name: cfg, color: colors[i % colors.length],
          values: (d.levels || []).map(function (p) {
            var v = d.configs[cfg][p];
            return typeof v === "number" ? v : null;
          }) };
      });
      drawChart(svg, levels, series, {});
      if (legend) {
        legend.innerHTML = "";
        series.forEach(function (s) {
          var item = document.createElement("span");
          item.className = "legend";
          var sw = document.createElement("span");
          sw.style.cssText = "width:14px;height:3px;border-radius:2px;background:" + s.color;
          item.appendChild(sw);
          item.appendChild(document.createTextNode(" " + s.name));
          legend.appendChild(item);
        });
      }
    }
    select.addEventListener("change", update);
    update();
  }

  function setupNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".nav-inner a"));
    var sections = links.map(function (a) { return document.querySelector(a.getAttribute("href")); });
    function activate() {
      var pos = window.scrollY + 90, current = 0;
      sections.forEach(function (s, i) { if (s && s.offsetTop <= pos) current = i; });
      links.forEach(function (a, i) { a.classList.toggle("active", i === current); });
    }
    window.addEventListener("scroll", activate, { passive: true });
    activate();
  }

  function setupLightbox() {
    var box = document.createElement("div");
    box.className = "lightbox";
    var img = document.createElement("img");
    box.appendChild(img);
    document.body.appendChild(box);
    box.addEventListener("click", function () { box.classList.remove("open"); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") box.classList.remove("open");
    });
    document.querySelectorAll(".gallery-item a").forEach(function (link) {
      link.addEventListener("click", function (e) {
        e.preventDefault();
        img.src = link.getAttribute("href");
        box.classList.add("open");
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var data = readData();
    setupAlpha(data.alpha);
    setupNoise(data.noise);
    setupNav();
    setupLightbox();
  });
})();

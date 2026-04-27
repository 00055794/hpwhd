"use strict";

const DEFAULT_LAT = 43.2567;
const DEFAULT_LON = 76.9286;

const ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const OSM_URL  = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

// Deep-green palette (no orange).
const PALETTE = {
  greenDark:   "#14322B",
  green:       "#1B4332",
  greenMid:    "#2D6A4F",
  greenBright: "#40916C",
  greenSoft:   "#74A57F",
  greenLeaf:   "#95D5B2",
};

const DRIVER_COLORS  = [PALETTE.greenDark, PALETTE.green, PALETTE.greenMid,
                        PALETTE.greenBright, PALETTE.greenSoft, PALETTE.greenLeaf];
const COMFORT_COLORS = [PALETTE.greenMid, PALETTE.greenBright, PALETTE.greenSoft];

let singleMap     = null;
let singleMarker  = null;
let driversChart  = null;
let comfortChart  = null;
let lastResult    = null;
let batchPredictions = null;
let batchLayer    = null;

// ── Map ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initSingleMap();
  document.getElementById("LATITUDE").addEventListener("change", syncMapFromInputs);
  document.getElementById("LONGITUDE").addEventListener("change", syncMapFromInputs);
  bindShareButtons();
});

function makeLayers() {
  return {
    esri: L.tileLayer(ESRI_URL, { attribution: "© Esri", maxZoom: 19 }),
    osm:  L.tileLayer(OSM_URL,  { attribution: "© OpenStreetMap", maxZoom: 19 }),
  };
}
function addMousePos(map) {
  const C = L.Control.extend({
    options: { position: "topright" },
    onAdd(m) {
      const d = L.DomUtil.create("div", "lf-mouse-pos leaflet-control");
      d.innerHTML = "&nbsp;";
      m.on("mousemove", e => { d.textContent = e.latlng.lat.toFixed(6) + "   " + e.latlng.lng.toFixed(6); });
      m.on("mouseout", () => { d.innerHTML = "&nbsp;"; });
      return d;
    },
  });
  new C().addTo(map);
}
function addFullscreen(map) {
  const C = L.Control.extend({
    options: { position: "topright" },
    onAdd() {
      const b = L.DomUtil.create("button", "leaflet-bar leaflet-control lf-fs-btn");
      b.title = "На весь экран"; b.textContent = "[ ]";
      L.DomEvent.disableClickPropagation(b);
      L.DomEvent.on(b, "click", () => {
        const el = map.getContainer();
        if (!document.fullscreenElement) el.requestFullscreen?.();
        else document.exitFullscreen?.();
      });
      return b;
    },
  });
  new C().addTo(map);
}

// ── Custom Nominatim search ───────────────────────────────────
function addCustomSearch(map) {
  const C = L.Control.extend({
    options: { position: "topleft" },
    onAdd() {
      const wrap = L.DomUtil.create("div", "lf-search");
      wrap.innerHTML =
        '<input type="text" class="lf-search-input" placeholder="Поиск адреса (улица, город)…" />' +
        '<div class="lf-search-list"></div>';
      L.DomEvent.disableClickPropagation(wrap);
      L.DomEvent.disableScrollPropagation(wrap);
      const input = wrap.querySelector(".lf-search-input");
      const list  = wrap.querySelector(".lf-search-list");
      let timer = null;

      const closeList = () => { list.innerHTML = ""; list.style.display = "none"; };
      const renderList = (items) => {
        if (!items.length) {
          list.innerHTML = '<div class="lf-search-empty">Адрес не найден</div>';
          list.style.display = "block"; return;
        }
        list.innerHTML = items.map((it, i) =>
          '<div class="lf-search-item" data-idx="' + i + '">' +
            it.display_name + '</div>').join("");
        list.style.display = "block";
        list.querySelectorAll(".lf-search-item").forEach(el => {
          el.addEventListener("click", () => {
            const it = items[parseInt(el.dataset.idx)];
            const lat = parseFloat(it.lat), lon = parseFloat(it.lon);
            if (isNaN(lat) || isNaN(lon)) return;
            input.value = it.display_name.split(",")[0];
            closeList();
            map.setView([lat, lon], 17, { animate: true });
            // Place the marker AFTER the map has finished panning so it is
            // attached to the freshly rendered tiles and visible.
            map.once("moveend", () => setLocation(lat, lon));
            // Fallback in case moveend doesn't fire (e.g. same view):
            setTimeout(() => setLocation(lat, lon), 50);
          });
        });
      };

      const doSearch = async () => {
        const q = input.value.trim();
        if (q.length < 2) { closeList(); return; }
        try {
          const url = "/geocode?q=" + encodeURIComponent(q);
          const resp = await fetch(url, { headers: { "Accept": "application/json" } });
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          const data = await resp.json();
          renderList(data || []);
        } catch (err) {
          list.innerHTML = '<div class="lf-search-empty">Ошибка поиска: ' + err.message + '</div>';
          list.style.display = "block";
        }
      };

      input.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(doSearch, 350);
      });
      input.addEventListener("keydown", e => {
        if (e.key === "Enter") { e.preventDefault(); clearTimeout(timer); doSearch(); }
        else if (e.key === "Escape") { closeList(); input.blur(); }
      });
      document.addEventListener("click", e => {
        if (!wrap.contains(e.target)) closeList();
      });
      return wrap;
    },
  });
  new C().addTo(map);
}

function initSingleMap() {
  const { esri, osm } = makeLayers();
  singleMap = L.map("single-map", {
    center: [DEFAULT_LAT, DEFAULT_LON],
    zoom: 12,
    layers: [esri],
    attributionControl: false,
  });
  L.control.attribution({ prefix: false, position: "bottomright" }).addTo(singleMap);
  L.control.layers({ "Спутник": esri, "Карта": osm }, null,
                   { collapsed: true, position: "topright" }).addTo(singleMap);
  addMousePos(singleMap);
  addFullscreen(singleMap);
  // Custom robust search bar (Nominatim direct fetch)
  addCustomSearch(singleMap);
  singleMap.on("click", e => setLocation(e.latlng.lat, e.latlng.lng));
  singleMarker = L.circle([DEFAULT_LAT, DEFAULT_LON], markerStyle()).addTo(singleMap);

  // Keep aspect-ratio map happy on viewport resize
  if (window.ResizeObserver) {
    new ResizeObserver(() => singleMap.invalidateSize()).observe(
      document.getElementById("single-map")
    );
  }
}
function syncMapFromInputs() {
  const lat = parseFloat(document.getElementById("LATITUDE").value);
  const lon = parseFloat(document.getElementById("LONGITUDE").value);
  if (isNaN(lat) || isNaN(lon) || !singleMap) return;
  singleMarker?.setLatLng([lat, lon]);
  singleMap.setView([lat, lon]);
}
function setLocation(lat, lon) {
  document.getElementById("LATITUDE").value  = lat.toFixed(6);
  document.getElementById("LONGITUDE").value = lon.toFixed(6);
  if (!singleMap) return;
  if (singleMarker && singleMap.hasLayer(singleMarker)) {
    singleMap.removeLayer(singleMarker);
  }
  singleMarker = L.circle([lat, lon], markerStyle()).addTo(singleMap);
  singleMarker.bringToFront();
}

// ── Predict ───────────────────────────────────────────────────
async function runPredict() {
  const payload = {
    ROOMS:        int_("ROOMS"),
    LONGITUDE:    float_("LONGITUDE"),
    LATITUDE:     float_("LATITUDE"),
    TOTAL_AREA:   float_("TOTAL_AREA"),
    FLOOR:        int_("FLOOR"),
    TOTAL_FLOORS: int_("TOTAL_FLOORS"),
    FURNITURE:    int_("FURNITURE"),
    CONDITION:    int_("CONDITION"),
    CEILING:      float_("CEILING"),
    MATERIAL:     int_("MATERIAL"),
    YEAR:         int_("YEAR"),
  };
  if (isNaN(payload.LATITUDE) || isNaN(payload.LONGITUDE)) {
    showError("Укажите координаты или выберите точку на карте.");
    return;
  }
  setLoading(true);
  hide("error-card");
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail || "Ошибка расчёта");
    }
    const data = await resp.json();
    lastResult = { data, payload };
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}
function setLoading(on) {
  const btn = document.getElementById("predict-btn");
  btn.disabled    = on;
  btn.textContent = on ? "Считаем..." : "Оценить квартиру";
  on ? show("loading-card") : hide("loading-card");
}

// ── Render ────────────────────────────────────────────────────
function renderResult(data) {
  show("price-widget");
  show("liv-widget");
  document.getElementById("insights-card").classList.remove("hidden");

  // Price
  animateNumber(document.getElementById("price-value"), data.price_kzt, fmtKzt);
  document.getElementById("price-sub").textContent = fmtKzt(data.price_per_sqm) + " ₸/м²";

  // 90% confidence band
  const cb = data.confidence;
  document.getElementById("conf-text").innerHTML =
    "Ожидаемый диапазон цены: <strong>" +
    fmtKzt(cb.lower) + " — " + fmtKzt(cb.upper) + " ₸</strong>";
  const pct = Math.max(0.05, Math.min(1, (data.price_kzt - cb.lower) / (cb.upper - cb.lower)));
  const fillEl = document.getElementById("conf-fill");
  const markEl = document.getElementById("conf-marker");
  fillEl.style.width = "0%"; markEl.style.left = "0%";
  void fillEl.offsetWidth;
  setTimeout(() => {
    fillEl.style.width = (pct * 100).toFixed(1) + "%";
    markEl.style.left  = (pct * 100).toFixed(1) + "%";
  }, 30);

  // Livability sentence
  const liv = data.livability;
  document.getElementById("liv-text").innerHTML =
    liv.narrative + " <span class=\"liv-score\">" + liv.overall + "/100</span>";

  // Charts + strip
  renderDrivers(data.drivers);
  renderComfort(liv);
  renderProxStrip(data.proximity);

  if (singleMarker) {
    singleMarker.unbindPopup();
    singleMarker.bindPopup(
      "<div style=\"font-weight:700;color:#40916C\">" + fmtKzt(data.price_kzt) + " ₸</div>"
    ).openPopup();
  }
}

// ── Drivers chart (pie + HTML legend) ─────────────────────────
function renderDrivers(drivers) {
  const ctx = document.getElementById("drivers-chart").getContext("2d");
  if (driversChart) driversChart.destroy();
  driversChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: drivers.map(d => d.name),
      datasets: [{
        data: drivers.map(d => d.value),
        backgroundColor: DRIVER_COLORS,
        borderColor: "#171c2a",
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { animateRotate: true, animateScale: true, duration: 900, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => c.label + ": " + c.parsed.toFixed(1) + "%" } },
      },
    },
  });

  // HTML legend
  const html = drivers.map((d, i) =>
    '<div class="lg-row">' +
      '<span class="lg-dot" style="background:' + DRIVER_COLORS[i] + '"></span>' +
      '<span class="lg-name">' + d.name + '</span>' +
      '<span class="lg-val">' + d.value + '%</span>' +
    '</div>'
  ).join("");
  document.getElementById("drivers-legend").innerHTML = html;
}

// ── Comfort doughnut: 3 sub-scores, proportional, centre overall ──
const centerTextPlugin = {
  id: "centerText",
  afterDraw(chart, _, opts) {
    if (!opts || !opts.text) return;
    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    const cx = (chartArea.left + chartArea.right) / 2;
    const cy = (chartArea.top  + chartArea.bottom) / 2;
    ctx.save();
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(45,106,79,0.35)";
    ctx.shadowBlur  = 8;
    ctx.fillStyle = "#2D6A4F";
    ctx.font = "800 26px Inter, Segoe UI, sans-serif";
    ctx.fillText(opts.text, cx, cy - 4);
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#5b6478";
    ctx.font = "700 9px Inter, Segoe UI, sans-serif";
    ctx.fillText("КОМФОРТ /100", cx, cy + 16);
    ctx.restore();
  },
};
Chart.register(centerTextPlugin);

function renderComfort(liv) {
  const ctx = document.getElementById("comfort-chart").getContext("2d");
  if (comfortChart) comfortChart.destroy();

  // Slices proportional to actual sub-scores so the visual matches the data.
  const labels = ["Инфраструктура", "Транспорт", "Состояние"];
  const values = [liv.infrastructure, liv.transport, liv.condition];

  comfortChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: COMFORT_COLORS,
        borderColor: "#171c2a",
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      animation: { animateRotate: true, animateScale: true, duration: 900, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => c.label + ": " + c.parsed + "/100" } },
        centerText: { text: String(liv.overall) },
      },
    },
  });

  const rows = labels.map((l, i) =>
    '<div class="lg-row">' +
      '<span class="lg-dot" style="background:' + COMFORT_COLORS[i] + '"></span>' +
      '<span class="lg-name">' + l + '</span>' +
      '<span class="lg-val">' + values[i] + '%</span>' +
    '</div>'
  ).join("");
  document.getElementById("comfort-legend").innerHTML = rows;
}

// ── Proximity strip ───────────────────────────────────────────
function renderProxStrip(prox) {
  const items = [
    { label: "Аптека",         km: prox.pharmacy.km     },
    { label: "Больница",       km: prox.hospital.km     },
    { label: "Детский сад",    km: prox.kindergarten.km },
    { label: "Главная улица",  km: prox.main_road.km    },
  ];
  const html = items.map(it => {
    const kmText = (it.km == null || isNaN(it.km)) ? "—" : "~" + it.km.toFixed(1);
    return '<div class="prox-cell">' +
             '<div class="pc-label">' + it.label + '</div>' +
             '<div class="pc-km">' + kmText + '<span class="pc-unit">км</span></div>' +
           '</div>';
  }).join("");
  document.getElementById("prox-strip").innerHTML = html;
}

// ── Share ─────────────────────────────────────────────────────
function bindShareButtons() {
  document.querySelectorAll(".share-btn").forEach(btn => {
    btn.addEventListener("click", () => doShare(btn.dataset.share));
  });
}
function buildShareText() {
  if (!lastResult) return {
    text: "ИИ-советник по недвижимости — индикативный ориентир по квартирам в Казахстане.",
    url: window.location.href,
  };
  const { data } = lastResult;
  const text =
    "Посмотрел(а) индикативный ориентир по квартире в ИИ-советнике: " +
    "комфорт проживания — " + data.livability.overall + "/100, " +
    "ориентир рыночной стоимости — около " + fmtKzt(data.price_kzt) + " ₸. " +
    "Оценка справочная, не заменяет отчёт оценщика.";
  return { text, url: window.location.href };
}
async function doShare(kind) {
  const { text, url } = buildShareText();
  try {
    if (kind === "whatsapp") {
      window.open("https://api.whatsapp.com/send?text=" + encodeURIComponent(text + "\n" + url),
                  "_blank", "noopener,noreferrer");
      return;
    }
    if (kind === "instagram") {
      await copyToClipboard(text + "\n" + url);
      showToast("Текст скопирован — откройте Instagram");
      window.open("https://www.instagram.com/", "_blank", "noopener,noreferrer");
      return;
    }
    if (kind === "copy") {
      await copyToClipboard(text + "\n" + url);
      showToast("Ссылка скопирована");
      return;
    }
  } catch (e) { showToast("Не удалось поделиться"); }
}
async function copyToClipboard(s) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(s); return;
  }
  const ta = document.createElement("textarea");
  ta.value = s; ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.focus(); ta.select();
  try { document.execCommand("copy"); } finally { document.body.removeChild(ta); }
}
function showToast(msg) {
  const el = document.getElementById("share-toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.add("hidden"), 2400);
}

// ── Batch ─────────────────────────────────────────────────────
async function handleBatchUpload() {
  const file = document.getElementById("batch-file").files[0];
  if (!file) return;
  hide("batch-results");
  show("batch-loading");
  document.getElementById("upload-zone").classList.add("uploading");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const resp = await fetch("/batch", { method: "POST", body: fd });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail);
    }
    batchPredictions = await resp.json();
    renderBatchResults(batchPredictions);
    document.getElementById("dl-csv-btn").disabled  = false;
    document.getElementById("dl-xlsx-btn").disabled = false;
  } catch (err) {
    alert("Ошибка обработки: " + err.message);
  } finally {
    hide("batch-loading");
    document.getElementById("upload-zone").classList.remove("uploading");
  }
}

const COND_LBL_RU = { 1: "Черновая", 2: "Требует ремонта", 3: "Среднее", 4: "Хорошее", 5: "Свежий ремонт" };
const MAT_LBL_RU  = { 1: "Иной", 2: "Панельный", 3: "Монолитный", 4: "Кирпичный" };
const FUR_LBL_RU  = { 1: "Без мебели", 2: "Частично", 3: "Полностью" };

function renderBatchResults(rows) {
  const banner = document.getElementById("batch-banner");
  banner.innerHTML =
    '<span>Оценено ' + rows.length + ' объектов</span>' +
    '<button class="btn-ghost btn-xs" onclick="showBatchOnMap()">Показать на карте</button>';
  const headers = [
    ["ROOMS",        "Комнаты"],
    ["TOTAL_AREA",   "Площадь"],
    ["FLOOR",        "Этаж"],
    ["TOTAL_FLOORS", "Этажность"],
    ["YEAR",         "Год"],
    ["MATERIAL",     "Материал"],
    ["CONDITION",    "Состояние"],
    ["LATITUDE",     "Широта"],
    ["LONGITUDE",    "Долгота"],
    ["pred_price_kzt", "Оценка, ₸"],
  ];
  let html = '<table class="data-table"><thead><tr>';
  headers.forEach(h => { html += "<th>" + h[1] + "</th>"; });
  html += "</tr></thead><tbody>";
  rows.slice(0, 100).forEach(r => {
    html += "<tr>";
    headers.forEach(h => {
      const k = h[0]; let v = r[k];
      let cls = "";
      if (k === "MATERIAL")  v = MAT_LBL_RU[v]  ?? v ?? "";
      else if (k === "CONDITION") v = COND_LBL_RU[v] ?? v ?? "";
      else if (k === "pred_price_kzt") { v = v == null ? "" : fmtKzt(v); cls = "num"; }
      else if (v == null) v = "";
      html += '<td class="' + cls + '">' + v + "</td>";
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("batch-table-wrap").innerHTML = html;
  show("batch-results");
}

function showBatchOnMap() {
  if (!batchPredictions || !singleMap) return;
  if (batchLayer) { singleMap.removeLayer(batchLayer); batchLayer = null; }
  const pts = [];
  const layer = L.layerGroup();
  batchPredictions.forEach((r, idx) => {
    const lat = parseFloat(r.LATITUDE), lon = parseFloat(r.LONGITUDE);
    if (isNaN(lat) || isNaN(lon)) return;
    pts.push([lat, lon]);
    const price = r.pred_price_kzt;
    L.circleMarker([lat, lon], {
      radius: 6, color: "#40916C", weight: 2,
      fillColor: "#2D6A4F", fillOpacity: 0.85,
    })
    .bindPopup(
      "<strong>#" + (idx + 1) + "</strong><br>" +
      (price != null ? fmtKzt(price) + " ₸" : "—") + "<br>" +
      lat.toFixed(5) + ", " + lon.toFixed(5)
    )
    .addTo(layer);
  });
  layer.addTo(singleMap);
  batchLayer = layer;
  if (pts.length) {
    singleMap.fitBounds(L.latLngBounds(pts).pad(0.15));
  }
  singleMap.getContainer().scrollIntoView({ behavior: "smooth", block: "start" });
}
function downloadCSV() {
  if (!batchPredictions) return;
  const COLS = ["ROOMS","TOTAL_AREA","FLOOR","TOTAL_FLOORS","FURNITURE","CONDITION",
                "CEILING","MATERIAL","YEAR","LATITUDE","LONGITUDE","pred_price_kzt"];
  let csv = COLS.join(",") + "\n";
  batchPredictions.forEach(r => { csv += COLS.map(c => r[c] ?? "").join(",") + "\n"; });
  trigger(new Blob([csv], { type: "text/csv" }), "predictions.csv");
}
async function downloadXLSX() {
  if (!batchPredictions) return;
  try {
    const resp = await fetch("/batch/download/xlsx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batchPredictions),
    });
    trigger(await resp.blob(), "predictions.xlsx");
  } catch (e) { alert("Ошибка скачивания: " + e.message); }
}
function trigger(blob, name) {
  const a = Object.assign(document.createElement("a"),
                          { href: URL.createObjectURL(blob), download: name });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Helpers ───────────────────────────────────────────────────
function int_(id)   { return parseInt(document.getElementById(id).value, 10); }
function float_(id) { return parseFloat(document.getElementById(id).value); }
function fmtKzt(n)  { return Math.round(n).toLocaleString("ru-RU"); }
function show(id)   { document.getElementById(id)?.classList.remove("hidden"); }
function hide(id)   { document.getElementById(id)?.classList.add("hidden"); }
function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  show("error-card");
}
function markerStyle() {
  // L.circle radius is in METRES → scales naturally with zoom.
  // 12 m ≈ small building footprint. White stroke makes the dot stand out
  // on satellite imagery.
  return { radius: 12, color: "#ffffff", fillColor: PALETTE.greenBright,
           fillOpacity: 0.85, weight: 2, opacity: 1 };
}
function animateNumber(el, target, formatter) {
  const dur = 800;
  const t0  = performance.now();
  function tick(t) {
    const k = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    const v = target * eased;
    el.textContent = formatter(v) + " ₸";
    if (k < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

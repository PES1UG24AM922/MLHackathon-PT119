const byId = id => document.getElementById(id);
const map = L.map("map", {preferCanvas: true}).setView([20, 0], 2);
let base = "";
let busy = false;
let ready = false;
let result = null;
let layer = null;

function controls() {
  byId("trip").disabled = busy || !ready;
  byId("run").disabled = busy || !ready || !byId("trip").value;
  byId("connect").disabled = busy;
  byId("api").disabled = busy;
  byId("export").disabled = busy || !result;
}

function clearResult() {
  if (layer) {
    layer.remove();
    layer = null;
  }
  result = null;
  byId("summary").textContent = "Select a trip to begin.";
  byId("metrics").replaceChildren();
}

async function request(path, options = {}, timeout = 8000) {
  const response = await fetch(base + path, {...options, signal: AbortSignal.timeout(timeout)});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const id = response.headers.get("X-Request-ID");
    throw new Error((body.detail || ("HTTP " + response.status)) + (id ? "\nRequest " + id : ""));
  }
  return response;
}

byId("connect").onclick = async () => {
  busy = true;
  ready = false;
  clearResult();
  controls();
  byId("message").textContent = "Connecting…";
  try {
    const url = new URL(byId("api").value);
    if (!["http:", "https:"].includes(url.protocol)) {
      throw new Error("Enter an HTTP or HTTPS address.");
    }
    base = url.origin;
    const health = await (await request("/health")).json();
    if (health.schema_version !== "1.0") throw new Error("Incompatible API version.");
    const trips = await (await request("/trips")).json();
    byId("trip").replaceChildren(...trips.trip_ids.map(id => new Option(id, id)));
    const state = await (await request("/ready")).json();
    ready = true;
    byId("status").textContent = health.mode === "development"
      ? "Student test · Development"
      : "Evaluator mode";
    byId("message").textContent = state.score_kind === "uncalibrated"
      ? "Development model · Candidate scores are uncalibrated."
      : "Connected. Select a trip.";
  } catch (error) {
    byId("status").textContent = "Unavailable";
    byId("message").textContent = "Connection failed. " + error.message +
      "\nCheck the Laptop 1 address and server status.";
  } finally {
    busy = false;
    controls();
  }
};

byId("trip").onchange = () => {
  clearResult();
  byId("message").textContent = "Ready to reconstruct the selected trip.";
  controls();
};

byId("run").onclick = async () => {
  busy = true;
  clearResult();
  controls();
  byId("message").textContent = "Processing selected trip…";
  try {
    const response = await request("/predictions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trip_id: byId("trip").value})
    }, 300000);
    const data = await response.json();
    layer = L.featureGroup().addTo(map);
    L.polyline(data.gps.map(point => [point.lat, point.lon]), {color: "#93a4b8", weight: 2}).addTo(layer);
    for (const segment of data.route_segments) {
      L.polyline(segment.map(point => [point.lat, point.lon]), {color: "#13b8a6", weight: 5}).addTo(layer);
    }
    map.fitBounds(layer.getBounds(), {padding: [20, 20], maxZoom: 17});
    byId("summary").textContent = "Trip " + data.trip_id + " · " + data.gps.length +
      " GPS points · " + data.route_segments.length + " route segments";
    byId("message").textContent = (data.duration_ms / 1000).toFixed(2) + " s · " +
      (data.score_kind === "uncalibrated" ? "Development model" : data.model_version) +
      "\nRequest " + data.request_id +
      (data.warnings.length ? "\n" + data.warnings.map(w => w.toLowerCase().replaceAll("_", " ")).join(", ") : "");
    for (const [key, value] of Object.entries(data.metrics)) {
      const metric = document.createElement("p");
      const label = {
        ordered_route_recall: "Reference route recovered",
        ordered_route_precision: "Predicted route overlap",
        ordered_route_f1: "Ordered route F1"
      }[key];
      metric.textContent = label + " · " + (100 * value).toFixed(1) + "%";
      byId("metrics").appendChild(metric);
    }
    result = data;
  } catch (error) {
    byId("message").textContent = error.message;
  } finally {
    busy = false;
    controls();
  }
};

byId("export").onclick = async () => {
  busy = true;
  controls();
  try {
    const response = await request("/maps/" + encodeURIComponent(result.request_id));
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "pathsnap-" + result.trip_id + ".html";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    byId("message").textContent = error.message;
  } finally {
    busy = false;
    controls();
  }
};

byId("api").value = location.pathname.startsWith("/ui/") ? location.origin : "http://127.0.0.1:8000";
if (location.pathname.startsWith("/ui/")) byId("connect").click();

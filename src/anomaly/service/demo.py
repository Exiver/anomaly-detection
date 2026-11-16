_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual anomaly detection</title>
<style>
  :root { color-scheme: light dark; --ok: #1a7f37; --bad: #cf222e; --muted: #6e7781; }
  body { font-family: system-ui, sans-serif; margin: 0 auto; padding: 1.5rem; max-width: 880px; }
  h1 { font-size: 1.4rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: end; margin-bottom: 1rem; }
  label { display: flex; flex-direction: column; font-size: .85rem; gap: .25rem; }
  #drop { border: 2px dashed var(--muted); border-radius: 8px; padding: 1.5rem; text-align: center; }
  #drop.over { border-color: var(--ok); }
  #status { margin: 1rem 0; color: var(--muted); min-height: 1.2em; }
  #verdict { font-size: 1.3rem; font-weight: 600; }
  .bar { position: relative; height: 14px; background: rgba(128,128,128,.25); border-radius: 7px; margin: .5rem 0 1rem; }
  .fill { height: 100%; border-radius: 7px; }
  .mark { position: absolute; left: 50%; top: -3px; width: 2px; height: 20px; background: currentColor; }
  .images { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .images img { width: 100%; border-radius: 6px; }
  figcaption { font-size: .8rem; color: var(--muted); }
</style>
</head>
<body>
<h1>Visual anomaly detection</h1>
<p>Pick a product category, then upload a PNG or JPEG image. The score is compared with a threshold calibrated on normal images.</p>
<div class="row">
  <!--API_ROW-->
  <label>API key (if required)<input id="key" type="password" autocomplete="off"></label>
  <label>Category<select id="category"></select></label>
  <button id="reload" type="button">Connect / reload</button>
</div>
<div id="drop"><p>Drop an image here or choose a file</p><input id="file" type="file" accept="image/png,image/jpeg"></div>
<div id="status" role="status"></div>
<section id="result" hidden>
  <div id="verdict"></div>
  <div class="bar"><div class="fill" id="fill"></div><div class="mark" title="threshold"></div></div>
  <div id="numbers"></div>
  <div class="images">
    <figure><img id="original" alt="Uploaded image"><figcaption>Uploaded image</figcaption></figure>
    <figure><img id="overlay" alt="Anomaly heatmap overlay"><figcaption>Anomaly overlay (resized and cropped as the model sees it)</figcaption></figure>
  </div>
</section>
<script>
const $ = (id) => document.getElementById(id); /*API_BASE_JS*/
const headers = () => { const key = $("key").value.trim(); return key ? { "X-API-Key": key } : {}; };
const setStatus = (text) => { $("status").textContent = text; };

async function loadCategories() {
    if ($("api") && !apiBase()) { setStatus("Enter the API URL, then press Connect."); return; }
  setStatus("Loading categories...");
  try {
    const response = await fetch("/v1/models", { headers: headers() });
    if (!response.ok) {
      setStatus(response.status === 401 ? "Enter a valid API key, then reload." : "Could not load categories (HTTP " + response.status + ").");
      return;
    }
    const models = await response.json();
    const select = $("category");
    select.replaceChildren();
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.category;
      option.textContent = model.category;
      select.appendChild(option);
    }
    setStatus(models.length ? "" : "No categories available.");
  } catch (error) {
    setStatus("Network error: " + error.message);
  }
}

async function analyze(file) {
  if (!file) return;
  if (!$("category").value) { setStatus("Choose a category first."); return; }
  setStatus("Analyzing...");
  $("result").hidden = true;
  const form = new FormData();
  form.append("file", file);
  const url = apiBase() + "/v1/predict?category=" + encodeURIComponent($("category").value) + "&include_heatmap=true";
  try {
    const response = await fetch(apiBase() + "/v1/models", { headers: headers() });
    const body = await response.json();
    if (!response.ok) {
      setStatus(typeof body.detail === "string" ? body.detail : "Request failed (HTTP " + response.status + ").");
      return;
    }
    show(file, body);
    setStatus("");
  } catch (error) {
    setStatus("Network error: " + error.message);
  }
}

function show(file, body) {
  const ratio = body.anomaly_score / body.threshold;
  const verdict = $("verdict");
  verdict.textContent = body.is_anomalous ? "Anomaly detected" : "Looks normal";
  verdict.style.color = body.is_anomalous ? "var(--bad)" : "var(--ok)";
  const fill = $("fill");
  fill.style.width = Math.min(ratio / 2, 1) * 100 + "%";
  fill.style.background = body.is_anomalous ? "var(--bad)" : "var(--ok)";
  $("numbers").textContent = "Score " + body.anomaly_score.toFixed(3) + " vs threshold " + body.threshold.toFixed(3) +
    " (ratio " + ratio.toFixed(2) + "), " + body.latency_ms + " ms, model " + body.model_version;
  $("original").src = URL.createObjectURL(file);
  $("overlay").src = "data:image/png;base64," + body.heatmap_png_base64;
  $("result").hidden = false;
}

$("file").addEventListener("change", (event) => analyze(event.target.files[0]));
$("reload").addEventListener("click", loadCategories);
const drop = $("drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => { event.preventDefault(); drop.classList.remove("over"); analyze(event.dataTransfer.files[0]); });
loadCategories();
</script>
</body>
</html>
"""

def render_demo(static: bool = False) -> str:
    """static=True adds an API URL field, for hosting the page away from the API."""
    if static:
        api_row = '<label>API URL<input id="api" type="url" placeholder="https://your-api.example.com"></label>'
        base_js = r'const apiBase = () => $("api").value.trim().replace(/\/$/, "");'
    else:
        api_row = ""
        base_js = 'const apiBase = () => "";'
    return _TEMPLATE.replace("<!--API_ROW-->", api_row).replace("/*API_BASE_JS*/", base_js)


DEMO_HTML = render_demo()
"""
Web lokal untuk model Keras deteksi katarak.

Jalankan:
    pip install flask pillow numpy tensorflow
    python app.py
Buka: http://localhost:5001
"""
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from tensorflow import keras

# ====================== SESUAIKAN BAGIAN INI ======================
MODEL_PATH = "cataract_model.keras"  # file model, taruh satu folder dengan app.py
RESCALE = 1.0 / 255.0                 # set None jika model sudah punya layer Rescaling/preprocess sendiri
CLASS_NAMES = ["Katarak", "Normal"]   # urutan alfabetis folder: cataract=0, normal=1. Jika hasil tertukar, balik urutannya
IMG_SIZE = (224, 224)                 # sesuai input model (lebar, tinggi)
# Output sigmoid (1 neuron): nilai = probabilitas CLASS_NAMES[1]
# Output softmax (>1 neuron): urutan sesuai CLASS_NAMES
# ==================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH, compile=False)

if IMG_SIZE is None:
    shape = model.input_shape  # (None, tinggi, lebar, kanal)
    if len(shape) == 4 and shape[1] and shape[2]:
        IMG_SIZE = (int(shape[2]), int(shape[1]))
    else:
        IMG_SIZE = (224, 224)
        print("Ukuran input tidak terbaca dari model, pakai 224x224. Isi IMG_SIZE manual jika perlu.")

print("Model dimuat. Input shape:", model.input_shape, "| Output shape:", model.output_shape)
print("Ukuran gambar yang dipakai (lebar, tinggi):", IMG_SIZE)


def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if RESCALE is not None:
        arr = arr * RESCALE
    return np.expand_dims(arr, axis=0)


def predict(file_bytes: bytes) -> dict:
    x = preprocess(file_bytes)
    out = model.predict(x, verbose=0)[0]

    if out.shape[0] == 1:  # sigmoid biner
        p = float(out[0])
        probs = [1.0 - p, p]
    else:                   # softmax
        probs = [float(v) for v in out]

    idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[idx],
        "confidence": probs[idx],
        "probabilities": {CLASS_NAMES[i]: probs[i] for i in range(len(probs))},
    }


PAGE = """
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deteksi Katarak</title>
<style>
  :root { --bg:#f5f6f8; --card:#fff; --text:#1c1f24; --muted:#6b7280; --accent:#0d9488; --border:#e5e7eb; --bad:#dc2626; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111418; --card:#1a1e24; --text:#e8eaed; --muted:#9aa0a6; --border:#2c323a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--text); display:flex; justify-content:center; padding:32px 16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:28px; width:100%; max-width:480px; }
  h1 { font-size:1.3rem; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 20px; font-size:.9rem; }
  .drop { display:block; border:2px dashed var(--border); border-radius:12px; padding:20px; text-align:center; cursor:pointer; color:var(--muted); }
  .drop.over { border-color:var(--accent); }
  img#preview { max-width:100%; border-radius:10px; margin-top:16px; display:none; }
  button { width:100%; margin-top:16px; padding:12px; border:0; border-radius:10px; background:var(--accent); color:#fff; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  #result { margin-top:20px; display:none; }
  .label { font-size:1.4rem; font-weight:700; }
  .bar { height:10px; background:var(--border); border-radius:6px; overflow:hidden; margin:6px 0 12px; }
  .bar > div { height:100%; background:var(--accent); }
  .row { display:flex; justify-content:space-between; font-size:.9rem; }
  .note { margin-top:16px; font-size:.78rem; color:var(--muted); }
  .err { color:var(--bad); margin-top:12px; }
</style>
</head>
<body>
<div class="card">
  <h1>Deteksi Katarak</h1>
  <p class="sub">Unggah foto mata untuk diprediksi oleh model.</p>

  <label class="drop" id="drop">
    <input type="file" id="file" accept="image/*" hidden>
    <span id="dropText">Klik atau seret gambar ke sini</span>
  </label>
  <img id="preview" alt="Pratinjau">
  <button id="btn" disabled>Prediksi</button>

  <div id="result">
    <div class="label" id="label"></div>
    <div class="row"><span>Keyakinan</span><span id="conf"></span></div>
    <div class="bar"><div id="confBar" style="width:0"></div></div>
    <div id="probs"></div>
  </div>
  <div class="err" id="err"></div>
  <p class="note">Hasil ini bukan diagnosis medis. Konsultasikan dengan dokter mata.</p>
</div>

<script>
const fileInput = document.getElementById('file');
const drop = document.getElementById('drop');
const preview = document.getElementById('preview');
const btn = document.getElementById('btn');
const result = document.getElementById('result');
const err = document.getElementById('err');
let current = null;

function setFile(f) {
  if (!f) return;
  current = f;
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
  document.getElementById('dropText').textContent = f.name;
  btn.disabled = false;
  result.style.display = 'none';
  err.textContent = '';
}
fileInput.addEventListener('change', e => setFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));

btn.addEventListener('click', async () => {
  if (!current) return;
  btn.disabled = true; btn.textContent = 'Memproses...';
  err.textContent = '';
  const fd = new FormData();
  fd.append('image', current);
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Gagal memprediksi');
    document.getElementById('label').textContent = data.label;
    document.getElementById('conf').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('confBar').style.width = (data.confidence * 100) + '%';
    document.getElementById('probs').innerHTML = Object.entries(data.probabilities)
      .map(([k, v]) => `<div class="row"><span>${k}</span><span>${(v*100).toFixed(1)}%</span></div>`).join('');
    result.style.display = 'block';
  } catch (e) {
    err.textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Prediksi';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict_route():
    f = request.files.get("image")
    if f is None or f.filename == "":
        return jsonify(error="Tidak ada gambar yang diunggah."), 400
    try:
        return jsonify(predict(f.read()))
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Gagal memproses gambar: {e}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

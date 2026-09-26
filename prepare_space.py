"""Build a ready-to-upload Hugging Face Space in .\\space\\ (your own project files are not changed).
    python prepare_space.py
Copies only what the web app needs: app.py, the sv_*.py code, the trained heads and two result files.
Never copies recordings, clones, voiceprints, data\\ or raw\\.

The copy of app.py is adjusted for a public demo:
- it opens on "Upload a recording" (the team demo calls need recordings that stay private),
- it listens on the Space's address instead of opening a local browser,
- uploaded audio is deleted from the server after about 10 minutes,
- the speed it shows is measured on the Space itself (reports/latency.json holds our RTX 5050 timing,
  so it is left out).
"""
import shutil, sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "space"
FILES = ["app.py", "sv_audio.py", "sv_heads.py", "sv_models.py", "sv_engine.py", "sv_views.py",
         "models/spk_head.pt", "models/deepfake/all.pt", "models/deepfake/calib.json",
         *[f"models/deepfake/fold_s0{i}.pt" for i in range(1, 7)],
         "reports/speaker_test.json", "reports/deepfake_loso.json"]
NEVER = ["models/voiceprints.npz", "data", "raw", "reports/latency.json"]


def ver(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def write(path, text):
    """Always Linux line endings: on Windows a plain write adds \\r, and the Space's apt-get then looks
    for a package called "ffmpeg\\r"."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def patch(text, old, new, what, required=True):
    if old in text:
        return text.replace(old, new, 1), True
    msg = f"  !! could not apply: {what}"
    if required:
        raise SystemExit(msg + " - send this message to Claude")
    print(msg + " (skipped; the Space still works)")
    return text, False


# ------------------------------------------------------------------ copy
if OUT.exists():
    shutil.rmtree(OUT)
missing = []
for f in FILES:
    src = ROOT / f
    if not src.exists():
        missing.append(f)
        continue
    (OUT / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, OUT / f)
for f in missing:
    print("  not found, skipped:", f)
if any(f in missing for f in ("app.py", "models/spk_head.pt", "models/deepfake/all.pt")):
    raise SystemExit("Run this inside satyavaani_app: core files are missing.")

# ------------------------------------------------------------------ app.py for a public demo
try:
    import inspect, gradio
    can_delete_cache = "delete_cache" in inspect.signature(gradio.Blocks.__init__).parameters
except Exception:
    can_delete_cache = False

app = (OUT / "app.py").read_text(encoding="utf-8")
app, _ = patch(app, 'server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True',
               'server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False',
               "listen on the Space's address")
app, _ = patch(app, 'mode = gr.Radio(MODES, value=MODES[0], label="Audio source")',
               'mode = gr.Radio(MODES[1:], value=MODES[1], label="Audio source")',
               'open on "Upload a recording"')
app, _ = patch(app, "with gr.Column(visible=True) as demo_box:",
               "with gr.Column(visible=False) as demo_box:", "hide the team demo calls")
app, _ = patch(app, "with gr.Column(visible=False) as upload_box:",
               "with gr.Column(visible=True) as upload_box:", "show the upload box first")
deleted = False
if can_delete_cache:
    app, deleted = patch(app, 'with gr.Blocks(title="SATYAVAANI", **blocks_kw) as demo:',
                         'with gr.Blocks(title="SATYAVAANI", delete_cache=(300, 300), **blocks_kw) as demo:',
                         "delete uploads after about 10 minutes", required=False)
privacy = ('"**Privacy (this public demo):** audio is processed in memory'
           + (" and uploaded files are deleted from the server within about 10 minutes" if deleted else "")
           + '; the only thing kept per enrolled person, until the Space restarts, "')
app, _ = patch(app, '"**Privacy:** runs on-premise, audio is processed in memory and dropped; the only thing stored per person "',
               privacy, "privacy line for the demo", required=False)
write(OUT / "app.py", app)

# ------------------------------------------------------------------ sv_audio.py: no shared temp file
sva = (OUT / "sv_audio.py").read_text(encoding="utf-8")
old = '''        tmp = ROOT / "tmp_decode.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cannot read {path.name}: {r.stderr.strip()}")
        w, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
'''
new = '''        import os, tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        try:
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                                "-ac", "1", "-ar", str(SR), tmp], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"cannot read {path.name}: {r.stderr.strip()}")
            w, sr = sf.read(tmp, dtype="float32", always_2d=True)
        finally:
            os.remove(tmp)
'''
sva, _ = patch(sva, old, new, "decode each upload into its own temp file and delete it", required=False)
write(OUT / "sv_audio.py", sva)

# ------------------------------------------------------------------ Space settings
gver, tver, trver = ver("gradio"), ver("torch"), ver("transformers")
tver = tver.split("+")[0] if tver else None
py = f"{sys.version_info.major}.{sys.version_info.minor}"
py = py if py in ("3.10", "3.11", "3.12") else "3.11"

# The Space installs Gradio and these packages in one go. Gradio 6.27+ needs huggingface_hub>=1.16, which
# transformers 4.x refuses (it needs <1.0), so with a newer Gradio the Space gets transformers 5.
# The app only uses Wav2Vec2Model from it, which works the same in both.
new_gradio = bool(gver) and int(gver.split(".")[0]) >= 6
if trver and not (new_gradio and trver.startswith("4.")):
    trf = f"transformers=={trver}"
else:
    trf = "transformers>=5,<6"
write(OUT / "requirements.txt", "\n".join([
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    f"torch=={tver}" if tver else "torch",
    trf,
    "numpy", "scipy", "soundfile", ""]))
write(OUT / "packages.txt", "ffmpeg\n")
write(OUT / "README.md", f"""---
title: SATYAVAANI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
{f'sdk_version: {gver}' if gver else ''}
python_version: "{py}"
app_file: app.py
pinned: false
---

# SATYAVAANI: real-time detection of AI voice clones on calls

Team-Cognify, Smart India Hackathon, problem statement SIH26104.

Upload a recording or use your microphone. Every 3 seconds the app checks whether the voice is AI-made
and, if you pick an enrolled caller, whether it matches that person. Result: PASS, VERIFY or HOLD.

- This demo runs on a free CPU, so each 3-second check is slower than on our RTX 5050 laptop GPU
  (about 18 ms there). The header shows the speed measured here.
- To try the caller check, enrol yourself on the Enrolment tab first (10-20 seconds of speech).
  Enrolled voices are kept only until the Space restarts, and their names are visible to other
  visitors in the caller list, so use a nickname.
- Our team's demo calls aren't included: our teammates' recordings stay private.

Code, trained heads and the full findings report: https://github.com/asmit-1101/satyavaani
""")

# ------------------------------------------------------------------ checks and housekeeping
for f in NEVER:
    if (OUT / f).exists():
        raise SystemExit(f"refusing to continue: {f} ended up in space\\ - delete the space folder")
gi = ROOT / ".gitignore"
if gi.exists() and "space/" not in gi.read_text(encoding="utf-8").split():
    with gi.open("a") as fh:
        fh.write("\nspace/\n")

files = [p for p in OUT.rglob("*") if p.is_file()]
size = sum(p.stat().st_size for p in files) / 1e6
print(f"space\\ ready: {len(files)} files, {size:.1f} MB, gradio {gver or 'latest'}, torch {tver or 'latest'} (CPU), python {py}")
print("uploads deleted after ~10 min:", "yes" if deleted else "no (this Gradio version can't; the Space still works)")
print("next: hf upload YOUR-HF-USERNAME/satyavaani .\\space . --repo-type=space")

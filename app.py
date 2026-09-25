"""SATYAVAANI web app (Gradio).

    python app.py               # http://127.0.0.1:7860

Tabs: Live call (team demo calls, an upload, or the live mic), Model insights, Enrolment, API.
"""
import time
import numpy as np
import gradio as gr
from sv_audio import (SR, SPEAKERS, NAME, TEST_CLIPS, real_clip, xtts_clip, stitch, telephonize,
                      load16, rms_norm, to16k, StreamResampler)
from sv_engine import Engine, CONTEXT_PRIORS, HOP_SEC
import sv_views as V

ENGINE = Engine.load()
TEAM = [sid for sid, *_ in SPEAKERS]
CLIPS_PER_CALL = 8                                   # ~35 s demo call from sentences 011-018
MODES = ["Demo call (team recordings)", "Upload a recording", "Microphone (live)"]
KINDS = ["Real voice", "AI clone (XTTS)", "Real, then switches to AI clone"]
GR6 = int(gr.__version__.split(".")[0]) >= 6

CSS = V.CSS + """
body, .gradio-container{background:#f4f6f9 !important}
.gradio-container{max-width:1480px !important}
#sv-top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:#0f1b2d;color:#fff;border-radius:12px;
  padding:14px 18px;font-family:Inter,Segoe UI,system-ui,sans-serif}
#sv-top .brand{font-weight:800;font-size:19px;letter-spacing:.08em}
#sv-top .sub{color:#cbd5e1;font-size:13px}
#sv-top .chips{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
#sv-top .chip{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.18);border-radius:999px;
  padding:4px 11px;font-size:12px;color:#e2e8f0}
.sv-panel{background:#fff !important;border:1px solid #e4e7ec !important;border-radius:12px !important;padding:16px !important}
.sv-panel-title{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#667085;margin:0 0 4px}
"""
THEME = gr.themes.Default(primary_hue="blue", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"])

def top_bar():
    lat = (ENGINE.info.get("latency") or {}).get("median_ms", ENGINE.info.get("startup_ms"))
    chips = [f"Models loaded &middot; {ENGINE.info.get('gpu', 'CPU')}",
             f"{lat:.0f} ms per 3 s window" if lat else "", f"{len(ENGINE.voiceprints)} enrolled voices"]
    logo = ('<svg width="30" height="30" viewBox="0 0 30 30"><rect width="30" height="30" rx="8" fill="#2a78d6"/>'
            '<g stroke="#fff" stroke-width="2.4" stroke-linecap="round"><line x1="8" y1="12" x2="8" y2="18"/>'
            '<line x1="12.5" y1="8" x2="12.5" y2="22"/><line x1="17" y1="11" x2="17" y2="19"/><line x1="21.5" y1="13" x2="21.5" y2="17"/></g></svg>')
    return (f'<div id="sv-top">{logo}<div><div class="brand">SATYAVAANI</div>'
            f'<div class="sub">Call risk console &middot; AI voice-clone and caller-identity check, every 3 seconds</div></div>'
            f'<div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in chips if c)}</div></div>')

def caller_choices():
    return [("Unknown number", "none")] + [(n, vid) for vid, (n, _) in ENGINE.voiceprints.items()]
def claimed_of(v): return None if v in (None, "", "none") else v
def prior_of(ctx): return float(sum(CONTEXT_PRIORS.get(c, 0.0) for c in (ctx or [])))
def pcm16(w): return (np.clip(w, -1, 1) * 32767).astype(np.int16)

# ------------------------------------------------------------------ building the audio of a call
def build_demo_call(speaker, kind):
    ids = [f"{speaker}_{k:03d}" for k in list(TEST_CLIPS)[:CLIPS_PER_CALL]]
    h = len(ids) // 2
    if kind == KINDS[0]:   paths = [real_clip(c) for c in ids]
    elif kind == KINDS[1]: paths = [xtts_clip(c) for c in ids]
    else:                  paths = [real_clip(c) for c in ids[:h]] + [xtts_clip(c) for c in ids[h:]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise gr.Error(f"Missing {len(missing)} clips (e.g. {missing[0]}). Run steps 1 and 2 first.")
    audio, marks = stitch(paths)
    truth = (f"real voice of {NAME[speaker]}" if kind == KINDS[0] else
             f"AI clone of {NAME[speaker]}" if kind == KINDS[1] else
             f"real {NAME[speaker]}, AI clone from {marks[h][0]:.0f} s")
    return audio, truth

def render(pack, t=0.0, live=False, ended=False):
    st = pack["st"]
    scored = any(h.get("scored") for h in st.history)
    verdict = st.verdict if scored else "LISTENING"
    reason = st.history[-1]["reason"] if st.history else ""
    return (V.call_card(pack["caller"], pack["channel"], t, live, ended, pack.get("truth"), verdict),
            V.phone_screen(pack["caller"], pack["channel"], t, live, ended, pack.get("truth"), verdict,
                           st.risk if scored else None, st.history, st.alert),
            V.risk_ring(st.risk if scored else None, verdict, reason, st.alert, history=st.history, thr=ENGINE.thr),
            V.timeline(st.history, t_now=t),
            V.window_log(st.history))

def empty_views():
    pack = {"st": ENGINE.new_call(), "caller": None, "channel": "-", "truth": None}
    return render(pack)

# ------------------------------------------------------------------ demo / upload calls
def prepare_call(mode, caller, speaker, kind, phone, ctx, upload):
    if mode == MODES[2]:
        raise gr.Error("Microphone mode: just press the record button in the microphone box.")
    if mode == MODES[1]:
        if not upload: raise gr.Error("Upload a recording first.")
        audio, truth, source = rms_norm(load16(upload)), None, None
    else:
        audio, truth = build_demo_call(speaker, kind)
        source = speaker                     # scored by the fold head that never trained on this voice
    if phone: audio = telephonize(audio)
    claimed = claimed_of(caller)
    pack = {"audio": audio, "st": ENGINE.new_call(claimed, source, prior_of(ctx)),
            "caller": ENGINE._name(claimed) if claimed else None,
            "channel": "Phone line - G.711, 8 kHz" if phone else "Wideband 16 kHz", "truth": truth}
    return ((SR, pcm16(audio)), pack, *render(pack, 0.0, live=True))

def run_call(pack):
    """Plays the call in real time: feeds 1 s at a time, the engine scores every 3 s."""
    if not pack or "audio" not in pack: return
    audio, st = pack["audio"], pack["st"]
    total, fed, t, t0 = len(audio) / SR, 0, 0.0, time.time()
    while t < total:
        t = min(total, t + 1.0)
        wait = t - (time.time() - t0)
        if wait > 0: time.sleep(wait)
        end = int(t * SR)
        ENGINE.push(st, audio[fed:end]); fed = end
        yield render(pack, t, live=True)
    yield render(pack, total, ended=True)

def hang_up(pack):
    if not pack: return empty_views()
    return render(pack, pack["st"].n / SR, ended=True)

# ------------------------------------------------------------------ live microphone
def mic_start(caller, ctx, phone=True):
    claimed = claimed_of(caller)
    pack = {"st": ENGINE.new_call(claimed, None, prior_of(ctx), phone=phone),
            "caller": ENGINE._name(claimed) if claimed else None,
            "channel": "Microphone (live) - phone line G.711, 8 kHz" if phone else "Microphone (live) - wideband 16 kHz",
            "truth": None}
    return (pack, *render(pack, 0.0, live=True))

def mic_chunk(chunk, pack, caller, ctx, phone=True):
    if not pack or "audio" in pack: pack = mic_start(caller, ctx, phone)[0]
    if chunk is not None:
        sr, data = chunk
        x = np.asarray(data)
        x = x.astype("float32") / 32768.0 if x.dtype == np.int16 else x.astype("float32")
        if x.ndim == 2: x = x.mean(1)
        prev = pack.get("last_raw")
        if prev is not None and len(x) > len(prev) and np.array_equal(x[: len(prev)], prev):
            new = x[len(prev):]                      # browser sent everything so far: keep only the new part
        else:
            new = x
        pack["last_raw"] = x
        if pack.get("rs") is None or pack["rs"].sr != int(sr): pack["rs"] = StreamResampler(sr)
        ENGINE.push(pack["st"], pack["rs"].push(new))
    return (pack, *render(pack, pack["st"].n / SR, live=True))

def mic_stop(pack):
    return hang_up(pack)

# ------------------------------------------------------------------ enrol + API
def do_enrol(name, path):
    name = (name or "").strip()
    if not name: raise gr.Error("Type a name.")
    if not path: raise gr.Error("Record or upload 10-20 s of their voice.")
    w = load16(path)
    if len(w) < 5 * SR: raise gr.Error("Need at least 5 seconds of speech.")
    vid, n = ENGINE.enroll(name, w)
    msg = f"Enrolled **{name}** as `{vid}` from {n} speech windows. Stored: one 192-number voiceprint, no audio."
    return msg, V.enrolled_list(ENGINE.voiceprints), gr.Dropdown(choices=caller_choices()), gr.Dropdown(choices=caller_choices())

def api_score(path, caller, ctx, phone=False):
    if not path: raise gr.Error("Give an audio file.")
    w = rms_norm(load16(path))
    if phone: w = telephonize(w)
    out = ENGINE.score_audio(w, claimed=claimed_of(caller), prior=prior_of(ctx))
    out["channel"] = "phone line G.711 8 kHz" if phone else "wideband 16 kHz"
    return out

API_DOC = """
**REST API** - run `python api_server.py` next to this app, then open **http://127.0.0.1:8000/docs**.
```
curl -F "file=@call.wav" -F "claimed_id=s01" http://127.0.0.1:8000/v1/score
```
Returns `final_risk`, `verdict`, `deepfake_prob_mean`, `speaker_cos` and every 3 s window.
The button on the left calls the same engine through Gradio's own API (`api_name="score"`).
"""

# ------------------------------------------------------------------ layout
def on_mode(mode):
    return (gr.Column(visible=mode == MODES[0]), gr.Column(visible=mode == MODES[1]),
            gr.Column(visible=mode == MODES[2]), gr.Button(visible=mode != MODES[2]))

blocks_kw = {} if GR6 else {"css": CSS, "theme": THEME}
default_caller = "s01" if "s01" in ENGINE.voiceprints else "none"

with gr.Blocks(title="SATYAVAANI", **blocks_kw) as demo:
    gr.HTML(top_bar())
    with gr.Tabs():
        with gr.Tab("Live call"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=300, elem_classes="sv-panel"):
                    gr.HTML('<div class="sv-panel-title">Call setup</div>')
                    mode = gr.Radio(MODES, value=MODES[0], label="Audio source")
                    caller = gr.Dropdown(caller_choices(), value=default_caller, label="Caller ID says it is")
                    with gr.Column(visible=True) as demo_box:
                        speaker = gr.Dropdown([(NAME[s], s) for s in TEAM], value="s01", label="Who is really speaking")
                        kind = gr.Radio(KINDS, value=KINDS[0], label="Voice")
                    with gr.Column(visible=False) as upload_box:
                        upload = gr.Audio(sources=["upload"], type="filepath", label="Recording to check")
                    with gr.Column(visible=False) as mic_box:
                        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                                       label="Press record and talk")
                    phone = gr.Checkbox(value=True, label="Send it through a phone line (8 kHz G.711)")
                    ctx = gr.CheckboxGroup(list(CONTEXT_PRIORS), label="Call context (raises the prior)")
                    with gr.Row():
                        start = gr.Button("Start call", variant="primary")
                        stop = gr.Button("Hang up", variant="stop")
                    player = gr.Audio(label="Call audio", autoplay=True, interactive=False, type="numpy")
                v0 = empty_views()
                with gr.Column(scale=3, min_width=320):
                    phone_v = gr.HTML(v0[1])
                with gr.Column(scale=5, min_width=360):
                    card_v = gr.HTML(v0[0])
                    ring_v = gr.HTML(v0[2])
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, min_width=360):
                    tl_v = gr.HTML(v0[3])
                with gr.Column(scale=5, min_width=360):
                    log_v = gr.HTML(v0[4])
            pack = gr.State(None)

        with gr.Tab("Model insights"):
            gr.HTML(V.layer_butterfly(ENGINE.info["df_layers"], ENGINE.info["spk_layers"]))
            gr.HTML(V.metrics_panel(ENGINE.info))
            gr.Markdown(
                f"**Pipeline per {HOP_SEC:.0f} s:** energy VAD on the raw audio -> one frozen wav2vec2-base pass -> "
                "13 hidden layers, each pooled to mean+std -> the deepfake head and the speaker head each take their "
                "own learned mix of those layers -> risk = P(synthetic OR not the claimed person) + call-context prior, "
                "smoothed over time.  \n"
                f"**Speaker head:** `{ENGINE.info.get('spk_arch', '')}`  \n"
                f"**Deepfake heads loaded:** one per held-out teammate ({', '.join(ENGINE.info.get('folds', []))}) + one on all.  \n"
                "**Privacy:** runs on-premise, audio is processed in memory and dropped; the only thing stored per person "
                "is a 192-number voiceprint (768 bytes).")

        with gr.Tab("Enrolment"):
            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="sv-panel"):
                    gr.HTML('<div class="sv-panel-title">Enrol a voice</div>')
                    en_name = gr.Textbox(label="Name")
                    en_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                        label="10-20 s of their normal voice")
                    en_btn = gr.Button("Enrol voiceprint", variant="primary")
                    en_msg = gr.Markdown()
                with gr.Column():
                    en_list = gr.HTML(V.enrolled_list(ENGINE.voiceprints))

        with gr.Tab("API"):
            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="sv-panel"):
                    gr.HTML('<div class="sv-panel-title">Score a file</div>')
                    api_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (3 s or more)")
                    api_caller = gr.Dropdown(caller_choices(), value="none", label="Caller claims to be")
                    api_ctx = gr.CheckboxGroup(list(CONTEXT_PRIORS), label="Call context")
                    api_phone = gr.Checkbox(value=False, label="Send it through a phone line (8 kHz G.711)")
                    api_btn = gr.Button("Score", variant="primary")
                    gr.Markdown(API_DOC)
                with gr.Column():
                    api_out = gr.JSON(label="Response")

    views = [card_v, phone_v, ring_v, tl_v, log_v]
    mode.change(on_mode, mode, [demo_box, upload_box, mic_box, start])
    ev1 = start.click(prepare_call, [mode, caller, speaker, kind, phone, ctx, upload], [player, pack] + views)
    ev2 = ev1.then(run_call, [pack], views)
    stop.click(hang_up, [pack], views, cancels=[ev1, ev2])
    mic.start_recording(mic_start, [caller, ctx, phone], [pack] + views)
    try:
        mic.stream(mic_chunk, [mic, pack, caller, ctx, phone], [pack] + views, stream_every=0.5, time_limit=900)
    except TypeError:                                  # older Gradio without stream_every
        mic.stream(mic_chunk, [mic, pack, caller, ctx, phone], [pack] + views)
    mic.stop_recording(mic_stop, [pack], views)
    en_btn.click(do_enrol, [en_name, en_audio], [en_msg, en_list, caller, api_caller])
    api_btn.click(api_score, [api_audio, api_caller, api_ctx, api_phone], api_out, api_name="score")
    demo.load(None, None, None, js="() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }")

if __name__ == "__main__":
    launch_kw = {"css": CSS, "theme": THEME} if GR6 else {}
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, **launch_kw)

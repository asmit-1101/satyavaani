"""SATYAVAANI web app (Gradio).

    python app.py               # http://127.0.0.1:7860

Tabs: Live call (team demo calls, an upload, or the live mic), Under the hood, Enrol, API.
"""
import time
import numpy as np
import gradio as gr
from sv_audio import (SR, SPEAKERS, NAME, TEST_CLIPS, real_clip, xtts_clip, stitch, telephonize,
                      load16, rms_norm, to16k)
from sv_engine import Engine, CONTEXT_PRIORS, HOP_SEC
import sv_views as V

ENGINE = Engine.load()
TEAM = [sid for sid, *_ in SPEAKERS]
CLIPS_PER_CALL = 8                                   # ~35 s demo call from sentences 011-018
MODES = ["Demo call (team recordings)", "Upload a recording", "Microphone (live)"]
KINDS = ["Real voice", "AI clone (XTTS)", "Real, then switches to AI clone"]
GR6 = int(gr.__version__.split(".")[0]) >= 6

CSS = V.CSS + """
.gradio-container{max-width:1440px !important}
#sv-head{padding:6px 2px 2px}
#sv-head h1{margin:0;font-size:30px;letter-spacing:.04em}
#sv-head p{margin:4px 0 0;color:#94a3b8}
"""
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

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
    reason = st.history[-1]["reason"] if st.history else "Waiting for the first 3 seconds of speech ..."
    return (V.call_card(pack["caller"], pack["channel"], t, live, ended, pack.get("truth"), verdict),
            V.risk_ring(st.risk if scored else None, verdict, reason, st.alert),
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
        ENGINE.push(pack["st"], to16k(x, sr))
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

def api_score(path, caller, ctx):
    if not path: raise gr.Error("Give an audio file.")
    return ENGINE.score_audio(rms_norm(load16(path)), claimed=claimed_of(caller), prior=prior_of(ctx))

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
    gr.HTML('<div id="sv-head"><h1>SATYAVAANI</h1>'
            '<p>Real-time AI voice-clone detection for live calls &middot; one wav2vec2 pass, two heads</p></div>')
    with gr.Tabs():
        with gr.Tab("Live call"):
            with gr.Row():
                with gr.Column(scale=4, min_width=320):
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
                        stop = gr.Button("Hang up")
                    player = gr.Audio(label="Call audio", autoplay=True, interactive=False, type="numpy")
                with gr.Column(scale=5, min_width=380):
                    card_v, ring_v, tl_v, log_v = [gr.HTML(v) for v in empty_views()]
            pack = gr.State(None)

        with gr.Tab("Under the hood"):
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

        with gr.Tab("Enrol"):
            with gr.Row():
                with gr.Column():
                    en_name = gr.Textbox(label="Name")
                    en_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                        label="10-20 s of their normal voice")
                    en_btn = gr.Button("Enrol voiceprint", variant="primary")
                    en_msg = gr.Markdown()
                with gr.Column():
                    en_list = gr.HTML(V.enrolled_list(ENGINE.voiceprints))

        with gr.Tab("API"):
            with gr.Row():
                with gr.Column():
                    api_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio (3 s or more)")
                    api_caller = gr.Dropdown(caller_choices(), value="none", label="Caller claims to be")
                    api_ctx = gr.CheckboxGroup(list(CONTEXT_PRIORS), label="Call context")
                    api_btn = gr.Button("Score", variant="primary")
                    gr.Markdown(API_DOC)
                with gr.Column():
                    api_out = gr.JSON(label="Response")

    views = [card_v, ring_v, tl_v, log_v]
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
    api_btn.click(api_score, [api_audio, api_caller, api_ctx], api_out, api_name="score")
    demo.load(None, None, None, js="() => { document.body.classList.add('dark'); }")

if __name__ == "__main__":
    launch_kw = {"css": CSS, "theme": THEME} if GR6 else {}
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, **launch_kw)

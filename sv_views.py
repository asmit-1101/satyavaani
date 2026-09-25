"""HTML/SVG pieces for the app: call header, risk gauge, signals, timeline, window log, model insights."""
import html
import math
import numpy as np

# status colours are reserved for the verdict and always come with an icon + word
COL = {"PASS": "#15803d", "VERIFY": "#b45309", "HOLD": "#b91c1c", "LISTENING": "#64748b"}
TINT = {"PASS": "#ecfdf3", "VERIFY": "#fff7e6", "HOLD": "#fef2f2", "LISTENING": "#f1f5f9"}
ICON = {"PASS": "&#10003;", "VERIFY": "!", "HOLD": "&#9632;", "LISTENING": "&#8230;"}
TITLE = {"PASS": "PASS", "VERIFY": "VERIFY", "HOLD": "HOLD", "LISTENING": "LISTENING"}
ACTION = {
    "PASS": "No sign of a cloned or wrong voice. Continue the call normally.",
    "VERIFY": "Something is off. Ask a security question or call back on the registered number before acting.",
    "HOLD": "Do not process payments or share OTPs. Escalate and call back on a saved number.",
    "LISTENING": "Waiting for the first 3 seconds of speech.",
}
DF_COL, SPK_COL = "#eb6834", "#2a78d6"            # deepfake head, speaker head (same in every chart)
INK, INK2, MUTED, LINE, SURF = "#0f1b2d", "#475467", "#667085", "#e4e7ec", "#ffffff"
FONT = "Inter,Segoe UI,system-ui,sans-serif"

CSS = f"""
.sv-card{{background:{SURF};border:1px solid {LINE};border-radius:12px;padding:16px 18px;color:{INK};
  font-family:{FONT};box-shadow:0 1px 2px rgba(16,24,40,.04)}}
.sv-h{{font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:{MUTED};margin:0 0 12px}}
.sv-muted{{color:{MUTED}}} .sv-small{{font-size:13px}}
.sv-callhead{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.sv-avatar{{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:18px;font-weight:700;color:#fff;flex:none}}
.sv-callname{{font-size:20px;font-weight:700;line-height:1.2}}
.sv-status{{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.sv-pill{{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:12px;
  background:#f2f4f7;color:{INK2};border:1px solid {LINE}}}
.sv-dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
.sv-verdict{{display:flex;gap:12px;align-items:flex-start;border-radius:10px;padding:12px 14px;margin-top:4px}}
.sv-vicon{{width:28px;height:28px;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;
  font-weight:800;flex:none;font-size:14px}}
.sv-vtitle{{font-weight:800;letter-spacing:.08em;font-size:15px}}
.sv-vtext{{font-size:13px;color:{INK2};margin-top:2px;line-height:1.45}}
.sv-alert{{margin-top:10px;border:1px solid #fecdca;background:#fef3f2;color:#912018;border-radius:10px;
  padding:9px 12px;font-size:13px;font-weight:600}}
.sv-meter{{margin-top:14px}}
.sv-meter .row{{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}}
.sv-meter .row b{{font-weight:600}}
.sv-track{{position:relative;height:8px;border-radius:999px;background:#f2f4f7}}
.sv-fill{{position:absolute;left:0;top:0;bottom:0;border-radius:999px}}
.sv-tick{{position:absolute;top:-3px;bottom:-3px;width:2px;background:{INK};opacity:.55;border-radius:1px}}
.sv-note{{font-size:12px;color:{MUTED};margin-top:4px}}
.sv-table{{width:100%;border-collapse:collapse;font-size:13px}}
.sv-table th{{color:{MUTED};font-weight:600;text-align:left;padding:7px 8px;border-bottom:1px solid {LINE};font-size:12px}}
.sv-table td{{padding:7px 8px;border-bottom:1px solid #f2f4f7;color:{INK}}}
.sv-table td.num{{font-variant-numeric:tabular-nums}}
.sv-tag{{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:999px;font-weight:700;font-size:11px}}
.sv-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.sv-metric{{border:1px solid {LINE};border-radius:10px;padding:14px;background:#fcfcfd}}
.sv-metric .v{{font-size:26px;font-weight:700;color:{INK};font-variant-numeric:tabular-nums}}
.sv-metric .k{{font-size:13px;font-weight:600;color:{INK2};margin-top:2px}}
.sv-metric .d{{font-size:12px;color:{MUTED};margin-top:6px;line-height:1.45}}
.sv-legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:{INK2};margin-top:8px}}
.sv-legend span{{display:inline-flex;align-items:center;gap:6px}}
"""

def _e(s): return html.escape(str(s))
def _initials(name):
    parts = [p for p in str(name).replace("(", " ").split() if p[:1].isalpha()]
    return "".join(p[0] for p in parts[:2]).upper() or "?"
def _mmss(t): t = int(max(t, 0)); return f"{t // 60:02d}:{t % 60:02d}"
def _tag(verdict, text=None):
    c = COL.get(verdict, COL["LISTENING"])
    return (f'<span class="sv-tag" style="background:{TINT.get(verdict, "#f1f5f9")};color:{c};border:1px solid {c}33">'
            f'{ICON.get(verdict, "")} {_e(text or verdict)}</span>')

# ------------------------------------------------------------------ call header
def call_card(caller_name, channel, t=0.0, live=False, ended=False, truth=None, verdict="LISTENING"):
    who = _e(caller_name) if caller_name else "Unknown number"
    if live:    status = f'<span class="sv-pill"><span class="sv-dot" style="background:#12b76a"></span>Live {_mmss(t)}</span>'
    elif ended: status = f'<span class="sv-pill">Call ended &middot; {_mmss(t)}</span>'
    else:       status = '<span class="sv-pill">Ready</span>'
    pills = status + f'<span class="sv-pill">{_e(channel)}</span>' + _tag(verdict, TITLE.get(verdict, verdict))
    demo = f'<div class="sv-note">Demo ground truth: {_e(truth)}</div>' if truth else ""
    bg = COL.get(verdict, COL["LISTENING"])
    return (f'<div class="sv-card"><div class="sv-callhead">'
            f'<div class="sv-avatar" style="background:{bg}">{_e(_initials(who)) if caller_name else "?"}</div>'
            f'<div><div class="sv-muted sv-small">Incoming call &middot; caller ID</div><div class="sv-callname">{who}</div>{demo}</div>'
            f'<div class="sv-status">{pills}</div></div></div>')

# ------------------------------------------------------------------ risk gauge + verdict + signals
def _arc(cx, cy, r, a0, a1):
    """SVG arc on the upper half circle; angles in degrees, 180 = left end, 0 = right end."""
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy - r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy - r * math.sin(math.radians(a1))
    return f"M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}"

def _gauge(risk, verdict, w=260):
    cx, cy, r, sw = w / 2, w * 0.55, w * 0.40, 16
    ang = lambda v: 180 - 180 * v / 100
    s = [f'<svg width="100%" viewBox="0 0 {w} {w * 0.68:.0f}" style="display:block;max-width:300px;margin:0 auto">']
    for lo, hi, v in ((0, 45, "PASS"), (45, 75, "VERIFY"), (75, 100, "HOLD")):   # the three bands, faint
        s.append(f'<path d="{_arc(cx, cy, r, ang(lo), ang(hi) + (0 if hi == 100 else 1.2))}" stroke="{COL[v]}" '
                 f'stroke-opacity=".16" stroke-width="{sw}" fill="none"/>')
    if risk is not None and risk > 0.5:
        s.append(f'<path d="{_arc(cx, cy, r, 180, ang(min(risk, 99.9)))}" stroke="{COL.get(verdict, INK)}" '
                 f'stroke-width="{sw}" fill="none" stroke-linecap="round"/>')
    for v in (45, 75):
        a = math.radians(ang(v))
        s.append(f'<text x="{cx + (r + 20) * math.cos(a):.1f}" y="{cy - (r + 20) * math.sin(a):.1f}" text-anchor="middle" '
                 f'font-size="11" fill="{MUTED}" font-family="{FONT}">{v}</text>')
    num = "&#8211;" if risk is None else f"{risk:.0f}"
    s.append(f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" font-size="46" font-weight="700" fill="{INK}" '
             f'font-family="{FONT}">{num}</text>'
             f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="12" fill="{MUTED}" font-family="{FONT}">risk score / 100</text>'
             f'<text x="{cx - r}" y="{cy + 22}" text-anchor="middle" font-size="11" fill="{MUTED}" font-family="{FONT}">0</text>'
             f'<text x="{cx + r}" y="{cy + 22}" text-anchor="middle" font-size="11" fill="{MUTED}" font-family="{FONT}">100</text></svg>')
    return "".join(s)

def _meter(label, value_text, frac, color, tick=None, note=""):
    fill = "" if frac is None else f'<div class="sv-fill" style="width:{100 * max(0, min(1, frac)):.1f}%;background:{color}"></div>'
    tk = "" if tick is None else f'<div class="sv-tick" style="left:{100 * tick:.1f}%"></div>'
    return (f'<div class="sv-meter"><div class="row"><span>{label}</span><b>{value_text}</b></div>'
            f'<div class="sv-track">{fill}{tk}</div><div class="sv-note">{note}</div></div>')

def risk_ring(risk=None, verdict="LISTENING", reason="", alert="", size=260, history=None, thr=None, claimed=None):
    c, tint = COL.get(verdict, COL["LISTENING"]), TINT.get(verdict, "#f1f5f9")
    banner = (f'<div class="sv-verdict" style="background:{tint}"><div class="sv-vicon" style="background:{c}">{ICON.get(verdict, "")}</div>'
              f'<div><div class="sv-vtitle" style="color:{c}">{TITLE.get(verdict, verdict)}</div>'
              f'<div class="sv-vtext">{ACTION.get(verdict, "")}</div></div></div>')
    alert_html = f'<div class="sv-alert">&#9888; {_e(alert)}</div>' if alert else ""
    last = next((h for h in reversed(history or []) if h.get("scored")), None)
    if last:
        p = last["p_synth"]
        m1 = _meter("Synthetic voice", f"{100 * p:.0f}%", p, DF_COL, 0.5,
                    "Median of the last three 3 s windows. Line = 50%.")
        cos = last.get("cos")
        if cos is None:
            m2 = _meter("Voice match to caller ID", "no caller ID", None, SPK_COL, None,
                        "Pick who the caller ID says to check the voice.")
        else:
            ok = thr is not None and cos >= thr
            m2 = _meter("Voice match to caller ID", f"{cos:.2f} &middot; {'match' if ok else 'no match'}",
                        max(0.0, cos), SPK_COL, thr, f"Cosine similarity to the enrolled voiceprint. Line = threshold {thr:.2f}."
                        if thr is not None else "")
        signals = m1 + m2
    else:
        signals = ('<div class="sv-note" style="margin-top:14px">Signals appear after the first 3 s window with enough speech.</div>')
    parts = [_e(x.strip()) for x in str(reason).split("|") if x.strip()]
    why = (f'<div class="sv-note" style="margin-top:12px;padding-top:10px;border-top:1px solid {LINE}">'
           f'{" &middot; ".join(parts)}</div>') if parts else ""
    return (f'<div class="sv-card"><div class="sv-h">Risk</div>{_gauge(risk, verdict, size)}{banner}{alert_html}'
            f'{signals}{why}</div>')

# ------------------------------------------------------------------ timeline
def timeline(history, t_now=0.0, width=720, height=230, min_span=30.0):
    pad_l, pad_r, pad_t, pad_b = 34, 12, 12, 26
    span = max(min_span, t_now, max([h["t"] for h in history], default=0))
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda t: pad_l + W * t / span
    Y = lambda v: pad_t + H * (1 - v / 100)
    s = [f'<svg width="100%" viewBox="0 0 {width} {height}" style="display:block" font-family="{FONT}">']
    for lo, hi, v in ((0, 45, "PASS"), (45, 75, "VERIFY"), (75, 100, "HOLD")):
        s.append(f'<rect x="{pad_l}" y="{Y(hi):.1f}" width="{W}" height="{Y(lo) - Y(hi):.1f}" fill="{COL[v]}" opacity="0.06"/>')
    for v in (0, 45, 75, 100):
        s.append(f'<line x1="{pad_l}" x2="{pad_l + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{LINE}"/>'
                 f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}" font-size="12">{v}</text>')
    step = 5 if span <= 40 else 10 if span <= 90 else 30
    for t in np.arange(0, span + 0.01, step):
        s.append(f'<text x="{X(t):.1f}" y="{height - 7}" text-anchor="middle" fill="{MUTED}" font-size="12">{int(t)}s</text>')
    sc = [h for h in history if h.get("scored")]
    rk = [h for h in history if h.get("scored") or h.get("eased")]
    if sc:
        ps = " ".join(f"{X(h['t']):.1f},{Y(100 * h['p_synth']):.1f}" for h in sc)
        s.append(f'<polyline points="{ps}" fill="none" stroke="{DF_COL}" stroke-width="2" stroke-dasharray="5 4"/>')
    if rk:
        pr = " ".join(f"{X(h['t']):.1f},{Y(h['risk']):.1f}" for h in rk)
        s.append(f'<polyline points="{pr}" fill="none" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
        for h in rk:
            s.append(f'<circle cx="{X(h["t"]):.1f}" cy="{Y(h["risk"]):.1f}" r="4.5" fill="{COL[h["verdict"]]}" '
                     f'stroke="{SURF}" stroke-width="2"/>')
    if t_now:
        s.append(f'<line x1="{X(t_now):.1f}" x2="{X(t_now):.1f}" y1="{pad_t}" y2="{pad_t + H}" stroke="{SPK_COL}" '
                 f'stroke-width="1.5" opacity="0.5"/>')
    s.append("</svg>")
    legend = (f'<div class="sv-legend"><span><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="{INK}" stroke-width="2.5"/></svg>Risk</span>'
              f'<span><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="{DF_COL}" stroke-width="2" stroke-dasharray="5 4"/></svg>Synthetic-voice probability</span>'
              f'<span>Bands: PASS &lt; 45 &middot; VERIFY 45&ndash;75 &middot; HOLD &ge; 75</span></div>')
    return f'<div class="sv-card"><div class="sv-h">Risk over the call</div>{"".join(s)}{legend}</div>'

# ------------------------------------------------------------------ window log
def window_log(history, last=8):
    rows = []
    for h in history[-last:][::-1]:
        if not h.get("scored"):
            note = "silence, score held" if not h.get("eased") else "silence, risk easing"
            rows.append(f'<tr><td class="num">{h["t"]:.0f}s</td><td class="num">{100 * h["speech"]:.0f}%</td>'
                        f'<td colspan="3" class="sv-muted">{note}</td><td></td></tr>')
            continue
        cos = "&ndash;" if h.get("cos") is None else f'{h["cos"]:.2f}'
        rows.append(f'<tr><td class="num">{h["t"]:.0f}s</td><td class="num">{100 * h["speech"]:.0f}%</td>'
                    f'<td class="num">{100 * h["p_synth"]:.0f}%</td><td class="num">{cos}</td>'
                    f'<td class="num">{h["ms"]:.0f} ms</td><td>{_tag(h["verdict"], h["verdict"] + " " + format(h["risk"], ".0f"))}</td></tr>')
    body = "".join(rows) or '<tr><td colspan="6" class="sv-muted">No windows yet</td></tr>'
    return (f'<div class="sv-card"><div class="sv-h">Event log &middot; every 3 s window</div><table class="sv-table">'
            f'<tr><th>Time</th><th>Speech</th><th>Synthetic</th><th>Voice match</th><th>Compute</th><th>Risk</th></tr>'
            f'{body}</table></div>')

# ------------------------------------------------------------------ model insights
def layer_butterfly(df_w, spk_w, width=720, row=24):
    """13 layers down the middle; deepfake-head weight grows left, speaker-head weight grows right."""
    df_w, spk_w = np.asarray(df_w, float), np.asarray(spk_w, float)
    L, top, mid = len(df_w), 44, width / 2
    maxv, half = max(df_w.max(), spk_w.max(), 1e-9), mid - 80
    h = top + L * row + 30
    s = [f'<svg width="100%" viewBox="0 0 {width} {h}" style="display:block" font-family="{FONT}">',
         f'<text x="{mid - 34}" y="{top - 14}" text-anchor="end" fill="{DF_COL}" font-size="13" font-weight="700">&#9664; Deepfake head</text>',
         f'<text x="{mid + 34}" y="{top - 14}" fill="{SPK_COL}" font-size="13" font-weight="700">Speaker head &#9654;</text>']
    for i in range(L):
        y = top + i * row
        wl, wr = half * df_w[i] / maxv, half * spk_w[i] / maxv
        s.append(f'<rect x="{mid - 26 - wl:.1f}" y="{y + 4}" width="{wl:.1f}" height="{row - 8}" rx="3" fill="{DF_COL}"/>')
        s.append(f'<rect x="{mid + 26}" y="{y + 4}" width="{wr:.1f}" height="{row - 8}" rx="3" fill="{SPK_COL}"/>')
        s.append(f'<text x="{mid}" y="{y + row / 2 + 4}" text-anchor="middle" fill="{INK2}" font-size="12" font-weight="600">L{i}</text>')
        if df_w[i] >= 0.02:
            s.append(f'<text x="{mid - 32 - wl:.1f}" y="{y + row / 2 + 4}" text-anchor="end" fill="{MUTED}" font-size="10">{df_w[i]:.2f}</text>')
        if spk_w[i] >= 0.02:
            s.append(f'<text x="{mid + 32 + wr:.1f}" y="{y + row / 2 + 4}" fill="{MUTED}" font-size="10">{spk_w[i]:.2f}</text>')
    s.append(f'<text x="{mid}" y="{top + L * row + 20}" text-anchor="middle" fill="{MUTED}" font-size="12">'
             f'L0 = right after the CNN &#183; L12 = top transformer layer &#183; each head learns its own mix</text></svg>')
    return (f'<div class="sv-card"><div class="sv-h">One wav2vec2 pass, two heads: which layers each one uses</div>'
            f'{"".join(s)}</div>')

def _metric(value, key, detail=""):
    return f'<div class="sv-metric"><div class="v">{value}</div><div class="k">{key}</div><div class="d">{detail}</div></div>'

def _pct(x, digits=1):
    try:
        return "n/a" if x is None or x != x else f"{100 * float(x):.{digits}f}%"
    except Exception:
        return "n/a"

def metrics_panel(info):
    df, sp, lat = info.get("deepfake", {}) or {}, info.get("speaker", {}) or {}, info.get("latency", {}) or {}
    ms = lat.get("median_ms", info.get("startup_ms"))
    std = (f' <span style="font-size:14px;color:{MUTED}">&plusmn;{100 * df["eer_std"]:.1f}</span>'
           if df.get("eer_std") is not None else "")
    cards = [
        _metric(_pct(df.get("eer_mean")) + std, "Deepfake EER, unseen speaker",
                "Leave-one-speaker-out: trained on five teammates and their XTTS clones, tested on the sixth."),
        _metric(_pct(df.get("eer_phone_mean")), "Deepfake EER on phone line", "Same test, 8 kHz G.711 copies only."),
        _metric(_pct(df.get("silence_eer_mean")), "Pause-only check", "A head trained on the pauses alone. Near 50% means it listens to the voice, not the room."),
        _metric(_pct(sp.get("clone_pass_rate")), "Clones that pass the voice check", "Why a speaker check alone isn't enough."),
        _metric(_pct(sp.get("eer")), "Speaker EER on our team",
                f'Genuine accepted {_pct(sp.get("genuine_accept"), 0)}, others rejected {_pct(sp.get("impostor_reject"), 0)}.'),
        _metric("n/a" if ms is None else f"{ms:.0f} ms", "Per 3 s window, both heads", f'{_e(info.get("gpu", ""))}. Only a 768-byte voiceprint is stored per person.'),
    ]
    return f'<div class="sv-card"><div class="sv-h">Measured on our data</div><div class="sv-grid">{"".join(cards)}</div></div>'

def enrolled_list(voiceprints):
    items = "".join(f'<span class="sv-pill" style="margin:0 6px 6px 0">{_e(n)} &middot; {_e(i)}</span>'
                    for i, (n, _) in voiceprints.items())
    return f'<div class="sv-card"><div class="sv-h">Enrolled voiceprints</div>{items or "<span class=sv-muted>None yet</span>"}</div>'

# ------------------------------------------------------------------ phone-style call screen
PHONE_CSS = """
.sv-phone{width:300px;max-width:100%;margin:0 auto;background:#0b1220;border-radius:40px;padding:14px 14px 22px;
  border:8px solid #1d2939;box-shadow:0 10px 30px rgba(16,24,40,.25);color:#fff;font-family:Inter,Segoe UI,system-ui,sans-serif}
.sv-notch{width:90px;height:20px;background:#1d2939;border-radius:0 0 14px 14px;margin:-14px auto 8px}
.sv-sbar{display:flex;justify-content:space-between;font-size:11px;color:#98a2b3;padding:0 8px}
.sv-pcenter{text-align:center;margin-top:18px}
.sv-pav{width:84px;height:84px;border-radius:50%;margin:0 auto;display:flex;align-items:center;justify-content:center;
  font-size:30px;font-weight:700;color:#fff;box-shadow:0 0 0 6px rgba(255,255,255,.06)}
.sv-pname{font-size:22px;font-weight:700;margin-top:12px}
.sv-psub{font-size:12px;color:#98a2b3;margin-top:3px}
.sv-pbadge{display:inline-flex;align-items:center;gap:8px;margin-top:16px;padding:8px 14px;border-radius:999px;
  font-weight:700;font-size:14px}
.sv-prisk{font-size:12px;color:#cbd5e1;margin-top:8px}
.sv-pmsg{margin:14px 6px 0;padding:10px 12px;border-radius:14px;font-size:12.5px;line-height:1.45;text-align:left}
.sv-wave{display:flex;gap:4px;justify-content:center;align-items:center;height:36px;margin-top:16px}
.sv-wave i{display:block;width:4px;border-radius:2px;background:#475467;height:6px}
.sv-wave.on i{animation:svw 1s ease-in-out infinite;background:#98a2b3}
.sv-wave.on i:nth-child(2n){animation-delay:.15s}.sv-wave.on i:nth-child(3n){animation-delay:.3s}
.sv-wave.on i:nth-child(5n){animation-delay:.45s}
@keyframes svw{0%,100%{height:6px}50%{height:30px}}
.sv-pbtns{display:flex;justify-content:center;gap:34px;margin-top:22px}
.sv-pbtn{width:54px;height:54px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff}
"""
CSS += PHONE_CSS

PHONE_BADGE = {"PASS": "Voice verified", "VERIFY": "Verify the caller", "HOLD": "Likely fraud call", "LISTENING": "Checking voice..."}

def phone_screen(caller_name, channel, t=0.0, live=False, ended=False, truth=None, verdict="LISTENING",
                 risk=None, history=None, alert=""):
    c = COL.get(verdict, COL["LISTENING"])
    who = _e(caller_name) if caller_name else "Unknown number"
    state = f"On call &middot; {_mmss(t)}" if live else (f"Call ended &middot; {_mmss(t)}" if ended else "Incoming call")
    last = next((h for h in reversed(history or []) if h.get("scored")), None)
    badge = PHONE_BADGE.get(verdict, verdict)
    if verdict == "HOLD" and last is not None:
        badge = "Likely AI-cloned voice" if last.get("p_synth", 0) >= 0.5 else "Not the caller ID's voice"
    msg = {"PASS": ("#0e3b25", "#a6f4c5", "The voice sounds human and matches the caller ID."),
           "VERIFY": ("#3d2a08", "#fedf89", "Ask something only the real caller would know, or call back on a saved number."),
           "HOLD": ("#4a1212", "#fecdca", alert or "Don't share OTPs or approve payments. Hang up and call back on a saved number."),
           "LISTENING": ("#1d2939", "#d0d5dd", "Listening. The first check takes 3 seconds of speech.")}[verdict if verdict in COL else "LISTENING"]
    bars = "".join("<i></i>" for _ in range(17))
    live_tag = '<span style="color:#f04438">&#9679;</span> LIVE' if live else ""
    rk = "" if risk is None else f'<div class="sv-prisk">Risk {risk:.0f} / 100 &middot; {_e(channel)}</div>'
    demo = f'<div class="sv-psub" style="margin-top:10px">Demo: {_e(truth)}</div>' if truth else ""
    return (f'<div class="sv-phone"><div class="sv-notch"></div>'
            f'<div class="sv-sbar"><span>SATYAVAANI</span><span>{live_tag}</span></div>'
            f'<div class="sv-pcenter"><div class="sv-pav" style="background:{c}">{_e(_initials(who)) if caller_name else "?"}</div>'
            f'<div class="sv-pname">{who}</div><div class="sv-psub">{state}</div>'
            f'<div class="sv-pbadge" style="background:{c}">{ICON.get(verdict, "")} {_e(badge)}</div>{rk}'
            f'<div class="sv-wave{" on" if live else ""}">{bars}</div>'
            f'<div class="sv-pmsg" style="background:{msg[0]};color:{msg[1]}">{_e(msg[2])}</div>{demo}'
            f'<div class="sv-pbtns"><div class="sv-pbtn" style="background:#344054">mute</div>'
            f'<div class="sv-pbtn" style="background:#d92d20">end</div><div class="sv-pbtn" style="background:#344054">keypad</div></div>'
            f'</div></div>')

"""SATYAVAANI - HTML/SVG views for the app (pure Python, no Gradio needed to render)."""
import html
import numpy as np

COL = {"PASS": "#22c55e", "VERIFY": "#f59e0b", "HOLD": "#ef4444", "LISTENING": "#64748b"}
LABEL = {"PASS": "PASS", "VERIFY": "VERIFY", "HOLD": "HOLD", "LISTENING": "LISTENING"}
DF_COL, SPK_COL = "#f97316", "#38bdf8"

CSS = """
.sv-card{background:#0f172a;border:1px solid #1e293b;border-radius:18px;padding:18px 20px;color:#e2e8f0;
  font-family:Inter,Segoe UI,system-ui,sans-serif}
.sv-muted{color:#94a3b8}.sv-small{font-size:13px}.sv-h{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin:0 0 10px}
.sv-caller{display:flex;align-items:center;gap:16px}
.sv-avatar{width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:24px;font-weight:700;color:#0f172a;flex:none}
.sv-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;margin:6px 6px 0 0;
  background:#1e293b;color:#cbd5e1;border:1px solid #334155}
.sv-live{color:#22c55e;font-weight:600}.sv-ended{color:#94a3b8;font-weight:600}
.sv-ringwrap{display:flex;flex-direction:column;align-items:center;gap:6px}
.sv-chip{padding:6px 18px;border-radius:999px;font-weight:800;letter-spacing:.12em;font-size:15px;color:#0f172a}
.sv-reason{font-size:14px;line-height:1.5;text-align:center;color:#cbd5e1;max-width:520px}
.sv-alert{margin-top:12px;background:#450a0a;border:1px solid #ef4444;color:#fecaca;border-radius:12px;
  padding:10px 14px;font-weight:600;font-size:14px}
.sv-table{width:100%;border-collapse:collapse;font-size:13px}
.sv-table th{color:#94a3b8;font-weight:600;text-align:left;padding:6px 8px;border-bottom:1px solid #1e293b}
.sv-table td{padding:6px 8px;border-bottom:1px solid #1e293b;color:#e2e8f0}
.sv-tag{padding:2px 8px;border-radius:999px;font-weight:700;font-size:11px;color:#0f172a}
.sv-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.sv-metric{background:#0b1220;border:1px solid #1e293b;border-radius:14px;padding:14px}
.sv-metric .v{font-size:28px;font-weight:800;color:#f8fafc}.sv-metric .k{font-size:12px;color:#94a3b8;margin-top:2px}
.sv-metric .d{font-size:12px;color:#64748b;margin-top:6px;line-height:1.4}
"""

def _e(s): return html.escape(str(s))
def _initials(name):
    parts = [p for p in str(name).replace("(", " ").split() if p[:1].isalpha()]
    return "".join(p[0] for p in parts[:2]).upper() or "?"
def _mmss(t): t = int(max(t, 0)); return f"{t // 60:02d}:{t % 60:02d}"

# ------------------------------------------------------------------ call card
def call_card(caller_name, channel, t=0.0, live=False, ended=False, truth=None, verdict="LISTENING"):
    status = (f'<span class="sv-live">&#9679; Live {_mmss(t)}</span>' if live else
              f'<span class="sv-ended">Call ended {_mmss(t)}</span>' if ended else
              '<span class="sv-muted">Ready</span>')
    who = _e(caller_name) if caller_name else "Unknown number"
    pills = f'<span class="sv-pill">{_e(channel)}</span>'
    if truth: pills += f'<span class="sv-pill">Demo truth: {_e(truth)}</span>'
    return (f'<div class="sv-card"><div class="sv-caller">'
            f'<div class="sv-avatar" style="background:{COL.get(verdict, "#64748b")}">{_e(_initials(who))}</div>'
            f'<div><div class="sv-muted sv-small">Incoming call &middot; caller ID says</div>'
            f'<div style="font-size:24px;font-weight:700">{who}</div>'
            f'<div class="sv-small">{status}</div><div>{pills}</div></div></div></div>')

# ------------------------------------------------------------------ risk ring
def risk_ring(risk=None, verdict="LISTENING", reason="Waiting for the first 3 seconds of speech ...", alert="", size=230):
    c = COL.get(verdict, "#64748b")
    r, sw = size / 2 - 16, 16
    circ = 2 * np.pi * r
    frac = 0 if risk is None else max(0.0, min(1.0, risk / 100))
    num = "--" if risk is None else f"{risk:.0f}"
    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
           f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#1e293b" stroke-width="{sw}"/>'
           f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round"'
           f' stroke-dasharray="{circ * frac:.1f} {circ:.1f}" transform="rotate(-90 {size/2} {size/2})"/>'
           f'<text x="50%" y="47%" text-anchor="middle" dominant-baseline="middle" fill="#f8fafc" '
           f'font-size="{size*0.27:.0f}" font-weight="800" font-family="Inter,Segoe UI,sans-serif">{num}</text>'
           f'<text x="50%" y="66%" text-anchor="middle" fill="#94a3b8" font-size="14" '
           f'font-family="Inter,Segoe UI,sans-serif">RISK / 100</text></svg>')
    parts = [_e(p.strip()) for p in str(reason).split("|")]
    alert_html = f'<div class="sv-alert">&#9888; {_e(alert)}</div>' if alert else ""
    return (f'<div class="sv-card"><div class="sv-ringwrap">{svg}'
            f'<div class="sv-chip" style="background:{c}">{LABEL.get(verdict, verdict)}</div>'
            f'<div class="sv-reason">{"<br>".join(parts)}</div></div>{alert_html}</div>')

# ------------------------------------------------------------------ timeline
def timeline(history, t_now=0.0, width=720, height=210, min_span=30.0):
    pad_l, pad_r, pad_t, pad_b = 38, 12, 14, 28
    span = max(min_span, t_now, max([h["t"] for h in history], default=0))
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda t: pad_l + W * t / span
    Y = lambda v: pad_t + H * (1 - v / 100)
    s = [f'<svg width="100%" viewBox="0 0 {width} {height}" style="display:block">']
    for lo, hi, col in ((0, 45, "#22c55e"), (45, 75, "#f59e0b"), (75, 100, "#ef4444")):
        s.append(f'<rect x="{pad_l}" y="{Y(hi):.1f}" width="{W}" height="{Y(lo) - Y(hi):.1f}" fill="{col}" opacity="0.07"/>')
    for v in (0, 45, 75, 100):
        s.append(f'<line x1="{pad_l}" x2="{pad_l + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#1e293b"/>'
                 f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#64748b" font-size="11">{v}</text>')
    step = 5 if span <= 40 else 10 if span <= 90 else 30
    for t in np.arange(0, span + 0.01, step):
        s.append(f'<text x="{X(t):.1f}" y="{height - 8}" text-anchor="middle" fill="#64748b" font-size="11">{int(t)}s</text>')
    sc = [h for h in history if h.get("scored")]
    if sc:
        ps = " ".join(f"{X(h['t']):.1f},{Y(100 * h['p_synth']):.1f}" for h in sc)
        s.append(f'<polyline points="{ps}" fill="none" stroke="{DF_COL}" stroke-width="2" stroke-dasharray="5 4" opacity="0.9"/>')
        pr = " ".join(f"{X(h['t']):.1f},{Y(h['risk']):.1f}" for h in sc)
        s.append(f'<polyline points="{pr}" fill="none" stroke="#f8fafc" stroke-width="3" stroke-linejoin="round"/>')
        for h in sc:
            s.append(f'<circle cx="{X(h["t"]):.1f}" cy="{Y(h["risk"]):.1f}" r="5" fill="{COL[h["verdict"]]}" stroke="#0f172a" stroke-width="2"/>')
    if t_now:
        s.append(f'<line x1="{X(t_now):.1f}" x2="{X(t_now):.1f}" y1="{pad_t}" y2="{pad_t + H}" stroke="#38bdf8" stroke-width="1.5" opacity="0.6"/>')
    s.append("</svg>")
    legend = (f'<div class="sv-small sv-muted" style="margin-top:6px">'
              f'<span style="color:#f8fafc">&#9473;&#9473;</span> risk &nbsp;&nbsp; '
              f'<span style="color:{DF_COL}">&#9476;&#9476;</span> synthetic-voice probability (deepfake head) &nbsp;&nbsp; '
              f'bands: PASS &lt;45 &middot; VERIFY 45-75 &middot; HOLD &ge;75</div>')
    return f'<div class="sv-card"><div class="sv-h">Risk over the call</div>{"".join(s)}{legend}</div>'

# ------------------------------------------------------------------ window log
def window_log(history, last=8):
    rows = []
    for h in history[-last:][::-1]:
        if not h.get("scored"):
            rows.append(f'<tr><td>{h["t"]:.0f}s</td><td>{100*h["speech"]:.0f}%</td><td colspan="4" class="sv-muted">too little speech - skipped</td></tr>')
            continue
        cos = "-" if h.get("cos") is None else f'{h["cos"]:.2f}'
        rows.append(f'<tr><td>{h["t"]:.0f}s</td><td>{100*h["speech"]:.0f}%</td><td>{100*h["p_synth"]:.0f}%</td>'
                    f'<td>{cos}</td><td>{h["ms"]:.0f} ms</td><td><span class="sv-tag" style="background:{COL[h["verdict"]]}">'
                    f'{h["verdict"]} {h["risk"]:.0f}</span></td></tr>')
    body = "".join(rows) or '<tr><td colspan="6" class="sv-muted">No windows yet</td></tr>'
    return (f'<div class="sv-card"><div class="sv-h">Every 3 s window</div><table class="sv-table">'
            f'<tr><th>at</th><th>speech</th><th>synthetic</th><th>voice match</th><th>compute</th><th>risk</th></tr>'
            f'{body}</table></div>')

# ------------------------------------------------------------------ under the hood
def layer_butterfly(df_w, spk_w, width=720, row=24):
    """13 layers down the middle; deepfake-head weight grows left, speaker-head weight grows right."""
    df_w, spk_w = np.asarray(df_w, float), np.asarray(spk_w, float)
    L = len(df_w)
    top, mid = 70, width / 2
    maxv = max(df_w.max(), spk_w.max(), 1e-9)
    half = mid - 70
    h = top + L * row + 60
    s = [f'<svg width="100%" viewBox="0 0 {width} {h}" style="display:block">',
         f'<rect x="{mid - 150}" y="6" width="300" height="34" rx="10" fill="#1e293b" stroke="#334155"/>',
         f'<text x="{mid}" y="28" text-anchor="middle" fill="#e2e8f0" font-size="14" font-weight="700">'
         f'3 s audio &#8594; wav2vec2-base (frozen) &#8212; ONE pass</text>',
         f'<text x="{mid - 40}" y="{top - 10}" text-anchor="end" fill="{DF_COL}" font-size="13" font-weight="700">&#9664; DEEPFAKE HEAD weight</text>',
         f'<text x="{mid + 40}" y="{top - 10}" fill="{SPK_COL}" font-size="13" font-weight="700">SPEAKER HEAD weight &#9654;</text>']
    for i in range(L):
        y = top + i * row
        wl, wr = half * df_w[i] / maxv, half * spk_w[i] / maxv
        s.append(f'<rect x="{mid - 30 - wl:.1f}" y="{y + 4}" width="{wl:.1f}" height="{row - 8}" rx="3" fill="{DF_COL}" opacity="{0.35 + 0.65 * df_w[i] / maxv:.2f}"/>')
        s.append(f'<rect x="{mid + 30}" y="{y + 4}" width="{wr:.1f}" height="{row - 8}" rx="3" fill="{SPK_COL}" opacity="{0.35 + 0.65 * spk_w[i] / maxv:.2f}"/>')
        s.append(f'<text x="{mid}" y="{y + row / 2 + 4}" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">L{i}</text>')
        s.append(f'<text x="{mid - 36 - wl:.1f}" y="{y + row / 2 + 4}" text-anchor="end" fill="#94a3b8" font-size="10">{df_w[i]:.2f}</text>')
        s.append(f'<text x="{mid + 36 + wr:.1f}" y="{y + row / 2 + 4}" fill="#94a3b8" font-size="10">{spk_w[i]:.2f}</text>')
    yb = top + L * row + 22
    s.append(f'<text x="{mid}" y="{yb}" text-anchor="middle" fill="#94a3b8" font-size="12">'
             f'L0 = just after the CNN &#183; L12 = top transformer layer &#183; each head learned its own softmax over the 13 layers</text>')
    s.append("</svg>")
    return f'<div class="sv-card"><div class="sv-h">One pass, two heads &#8212; which layers each head listens to</div>{"".join(s)}</div>'

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
    cards = [
        _metric(_pct(df.get("eer_mean")) + (f' <span style="font-size:14px;color:#94a3b8">&plusmn;{100*df["eer_std"]:.1f}</span>' if df.get("eer_std") is not None else ""),
                "Deepfake EER, unseen speaker", "6-fold leave-one-speaker-out: trained on 5 teammates' voices + XTTS clones, tested on the 6th."),
        _metric(_pct(df.get("eer_phone_mean")), "Deepfake EER on phone line", "Same test, only the 8 kHz G.711 copies."),
        _metric(_pct(df.get("silence_eer_mean")), "Silence-only EER", "Head trained on the pauses only. ~50% = it reads the voice, not the room."),
        _metric(_pct(sp.get("clone_pass_rate")), "XTTS clones that pass the voice check",
                "Why a speaker check alone is not enough - the deepfake head catches these."),
        _metric(_pct(sp.get("eer")), "Speaker EER on our team", f'Genuine accepted {_pct(sp.get("genuine_accept"), 0)}, others rejected {_pct(sp.get("impostor_reject"), 0)}.'),
        _metric("n/a" if ms is None else f"{ms:.0f} ms", "Per 3 s window, both heads", f'{info.get("gpu", "")}; VRAM {lat.get("vram_gb", "n/a")} GB.'),
        _metric("768 B", "Stored per person", "One 192-number voiceprint. No audio is kept."),
    ]
    return f'<div class="sv-card"><div class="sv-h">Measured on our data</div><div class="sv-grid">{"".join(cards)}</div></div>'

def enrolled_list(voiceprints):
    items = "".join(f'<span class="sv-pill">{_e(n)} &middot; {_e(i)}</span>' for i, (n, _) in voiceprints.items())
    return f'<div class="sv-card"><div class="sv-h">Enrolled voiceprints</div>{items or "<span class=sv-muted>none yet</span>"}</div>'

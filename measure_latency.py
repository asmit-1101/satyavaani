"""Measure the live pipeline on your GPU: one 3 s window -> wav2vec2 pass + deepfake head + speaker head.
    python measure_latency.py
Writes reports\\latency.json (shown in the app's 'Under the hood' tab)."""
import json, time
import numpy as np, torch
from sv_audio import WIN, REPORTS
from sv_engine import Engine

def main():
    eng = Engine.load()
    cuda = torch.cuda.is_available()
    w = (np.random.default_rng(0).standard_normal(WIN) * 0.05).astype("float32")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    for _ in range(5): eng.speaker_embed(eng.featurize([w]))
    if cuda: torch.cuda.reset_peak_memory_stats()
    tot, encm = [], []
    for _ in range(50):
        sync(); t0 = time.perf_counter()
        f = eng.featurize([w]); sync(); t1 = time.perf_counter()
        eng.deepfake_logit(f, None); eng.speaker_embed(f); sync(); t2 = time.perf_counter()
        tot.append((t2 - t0) * 1000); encm.append((t1 - t0) * 1000)
    rep = {"median_ms": float(np.median(tot)), "p95_ms": float(np.percentile(tot, 95)),
           "encoder_ms": float(np.median(encm)), "heads_ms": float(np.median(tot) - np.median(encm)),
           "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
           "device": eng.info.get("gpu"), "realtime_factor": float(3000 / np.median(tot))}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "latency.json").write_text(json.dumps(rep, indent=2))
    print(f"\nper 3 s window: {rep['median_ms']:.1f} ms median (p95 {rep['p95_ms']:.1f})  = "
          f"encoder {rep['encoder_ms']:.1f} + both heads {rep['heads_ms']:.1f}")
    print(f"{rep['realtime_factor']:.0f}x faster than real time, peak VRAM {rep['vram_gb']} GB on {rep['device']}")
    print("saved reports\\latency.json.  Next: python app.py")

if __name__ == "__main__":
    main()

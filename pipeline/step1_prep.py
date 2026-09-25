"""Step 1: turn raw phone recordings into clean 16 kHz clips.

    python pipeline/step1_prep.py

Put every recording in raw/. Files longer than 20 s are reference passages (matched to a person by
name, or through reports/reference_map.txt); shorter ones are sentences, numbered as in
sv_audio.SPEAKERS. Writes data/real/, data/ref/ and reports/mapping_sNN.txt.
"""
import re, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import (RAW, REAL_DIR, REF_DIR, REPORTS, SPEAKERS, SR, load16, duration, trim_silence,
                      rms_norm, save16, load_sentences)

N_SENT, REF_MIN_SEC = 30, 20.0
MAPFILE = REPORTS / "reference_map.txt"
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".flac", ".3gp", ".amr", ".wma"}
MAXNUM = max(hi for *_, hi in SPEAKERS)

def file_number(p):
    ok = [int(x) for x in re.findall(r"\d+", p.stem) if 1 <= int(x) <= MAXNUM]
    return ok[-1] if ok else None

def read_map():
    out = {}
    if MAPFILE.exists():
        for line in MAPFILE.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                left, right = line.split("=", 1)
                if right.strip(): out[left.split()[0]] = right.strip()
    return out

def write_map(unclaimed, missing, durs):
    lines = ["# Which reference passage belongs to whom?",
             "# Play the long files listed at the bottom, write the right file name after each '=',",
             "# save this file, then run pipeline/step1_prep.py again.", ""]
    for sid, name, *_ in SPEAKERS:
        guess = unclaimed[missing.index(sid)].name if (sid in missing and len(unclaimed) == len(missing)) else ""
        lines.append(f"{sid} {name:<8} = {guess}")
    lines += ["", "# long files in raw\\ :"] + [f"#   {p.name}   ({durs[p]:.0f} s)" for p in unclaimed]
    MAPFILE.parent.mkdir(parents=True, exist_ok=True)
    MAPFILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    if not RAW.exists(): sys.exit("No raw\\ folder next to this script.")
    files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    if not files: sys.exit("raw\\ has no audio files.")
    REPORTS.mkdir(exist_ok=True)
    print(f"{len(files)} audio files in raw\\ - measuring lengths ...")
    durs = {p: duration(p) for p in files}
    longs = [p for p in files if durs[p] >= REF_MIN_SEC]
    shorts = [p for p in files if durs[p] < REF_MIN_SEC]
    print(f"  {len(shorts)} sentence files, {len(longs)} reference passages (> {REF_MIN_SEC:.0f} s)\n")
    problems = []

    # references: person's name in the file/folder name, else reports\reference_map.txt
    refmap, unclaimed = {}, []
    for p in longs:
        hay = (p.stem + " " + p.parent.name).lower()
        hit = [sid for sid, name, *_ in SPEAKERS if name.lower() in hay]
        if len(hit) == 1: refmap[hit[0]] = p
        else: unclaimed.append(p)
    for sid, fname in read_map().items():
        q = next((p for p in longs if p.name.lower() == fname.lower()), None)
        if q is None: problems.append(f"reference_map.txt: no long file called '{fname}'"); continue
        refmap[sid] = q
        if q in unclaimed: unclaimed.remove(q)
    missing = [sid for sid, *_ in SPEAKERS if sid not in refmap]
    if missing and unclaimed:
        write_map(unclaimed, missing, durs)
        print("PAUSED - can't tell whose reference passage is whose.")
        print(f"  1. notepad {MAPFILE}")
        print("  2. play the long files, fix the file name after each '=' (a guess is filled in)")
        print("  3. save, run  python pipeline/step1_prep.py  again")
        return
    for sid in missing: problems.append(f"{sid}: no reference passage found")

    # sentences: bucket by number range, sort, hand out as 1..30
    buckets = {sid: [] for sid, *_ in SPEAKERS}
    for p in shorts:
        n = file_number(p)
        if n is None: problems.append(f"no number in '{p.name}' - skipped"); continue
        buckets[next(s for s, _, lo, hi in SPEAKERS if lo <= n <= hi)].append((n, p))

    for sid, name, lo, hi in SPEAKERS:
        got = sorted(buckets[sid]); nums = [n for n, _ in got]
        if len(got) != N_SENT:
            problems.append(f"{sid} {name}: {len(got)} files in {lo}-{hi} (want {N_SENT}); found {nums}")
        dup = sorted({n for n in nums if nums.count(n) > 1})
        if dup: problems.append(f"{sid} {name}: number used twice: {dup}")
        ref_d = 0.0
        if sid in refmap:
            w = rms_norm(trim_silence(load16(refmap[sid]))); save16(REF_DIR / f"{sid}.wav", w); ref_d = len(w) / SR
        texts, rows, ds = load_sentences(sid), [], []
        for k, (n, src) in enumerate(got[:N_SENT], 1):
            cid = f"{sid}_{k:03d}"
            w = load16(src); raw_d = len(w) / SR
            w = rms_norm(trim_silence(w)); d = len(w) / SR; ds.append(d)
            save16(REAL_DIR / f"{cid}.wav", w)
            txt = texts.get(cid, ("", "?"))[1]
            rows.append(f"{cid}  <- file {n:>3}  {src.name:<28} {raw_d:5.1f}s -> {d:4.1f}s | {txt}")
            if d < 1.8: problems.append(f"{cid} (file {n}): only {d:.1f}s of speech - re-record?")
            if d > 15:  problems.append(f"{cid} (file {n}): {d:.1f}s long - is it one sentence?")
        (REPORTS / f"mapping_{sid}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        rname = refmap[sid].name if sid in refmap else "MISSING"
        print(f"{sid} {name:<8} {len(got):2d} clips (files {nums[0] if nums else '-'}-{nums[-1] if nums else '-'})  "
              f"speech {min(ds, default=0):.1f}-{max(ds, default=0):.1f}s   ref {ref_d:4.1f}s <- {rname}")

    print("\nCHECK: notepad reports\\mapping_s06.txt, then play data\\real\\s06_004.wav and s06_005.wav -")
    print("       the words in each must match the sentence printed on its line.")
    if problems:
        print("\nFix these:"); [print("   -", p) for p in problems]
    else:
        print("\nAll good. Next: python pipeline/step2_clone.py")

if __name__ == "__main__":
    main()

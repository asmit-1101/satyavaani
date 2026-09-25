# SATYAVAANI

Real-time detection of AI voice clones on phone calls. Built for Smart India Hackathon, problem statement 21064.

Every 3 seconds we run one pass of a frozen wav2vec2 model over the call audio and feed it to two small heads. One asks *is this voice synthetic?*, the other asks *is this really the person on the caller ID?* Together they give one risk score: **PASS**, **VERIFY** or **HOLD**.

```mermaid
flowchart LR
    A["Call audio, 3 s windows"] --> B["wav2vec2-base (frozen)"]
    B --> C["Deepfake head"]
    B --> D["Speaker head"]
    C --> E["Risk: synthetic OR not the caller"]
    D --> E
    E --> F["PASS / VERIFY / HOLD"]
```

## Results

| | |
|---|---|
| Deepfake detection on a teammate the model never heard | **4.6% ± 3.3% EER** (4.7% on phone-line audio) |
| At the app's threshold | 94.7% of clones caught, 7.1% of real voices flagged |
| Speaker check on our team | 10.9% EER |
| XTTS clones that fool the speaker check alone, over a phone line | 43%, which is why we need both heads |
| Trained on one generator (XTTS), tested on another (kNN-VC) | ~40% EER, our main open problem |
| Trained on both generators (LibriSpeech, 20 unseen speakers) | kNN-VC drops from 40.3% to 24.7% EER |
| Time per 3 s window, RTX 5050 laptop GPU | 18.3 ms |

Everything we tried, including what didn't work, is written up in [docs/SATYAVAANI_experimental_findings.pdf](docs/SATYAVAANI_experimental_findings.pdf).

![EER for heads trained on one or two generators](docs/chart_generators.png)

Which wav2vec2 layers each deepfake head learned to rely on. Each generator leaves its traces at a different depth:

![Layer weights per training set](docs/chart_layers.png)

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python app.py          # http://127.0.0.1:7860
python api_server.py   # http://127.0.0.1:8000/docs
```

The trained heads are in `models/`, and wav2vec2-base downloads on first run. You can upload a recording, use the live microphone, or enrol new voices. The built-in demo calls need our team's recordings, which aren't in this repo.

## Train it on your own voices

Put your phone recordings in `raw/`, describe who read what in `SPEAKERS` at the top of `sv_audio.py`, then:

```powershell
python pipeline/step1_prep.py        # clean the recordings
python pipeline/step2_clone.py       # XTTS-v2 clone of every sentence
python pipeline/step3_features.py    # wav2vec2 features, same augmentation for real and fake
python pipeline/step4_deepfake.py    # leave-one-speaker-out training and test
python pipeline/step5_speaker.py     # enrol voiceprints and test the speaker check
python pipeline/measure_latency.py
```

The speaker head was trained separately on LibriSpeech train-clean-100 (`ref_code/train_speaker.py`).

## What's where

```
app.py, api_server.py     the web app and the REST API
sv_*.py                   audio processing, models, scoring engine, UI pieces
pipeline/                 steps 1-5 and the latency test
experiments/              LibriSpeech generalisation, two generators, voiceprint comparison
tools/                    export layer weights, redraw the layer chart
models/                   trained heads (see models/README.md)
reports/, libri_exp/      results as JSON
docs/                     findings report and charts
```

## Not in this repo

- **Our teammates' recordings, their AI clones and their voiceprints.** Real voices of named people next to working clones of them is exactly what an impersonator would want.
- **Features, LibriSpeech audio and generated clips.** They're big and can be rebuilt. LibriSpeech is at [openslr.org/12](https://www.openslr.org/12).
- **Third-party models.** [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) (Apache-2.0), [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (Coqui Public Model License, non-commercial) and [kNN-VC](https://github.com/bshall/knn-vc) (MIT) download from their own sources.

## Known limitations

- Six speakers is small, so per-speaker results vary a lot.
- A detector trained on one clone generator mostly misses others. Training on two helps but raises false alarms (25% at the default threshold), and it still needs testing on a third generator it has never seen.
- The speaker head learned from English audiobooks, so its threshold had to be retuned for Indian-accented phone speech.

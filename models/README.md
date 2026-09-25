# Trained weights

| File | What it is | Trained on |
|---|---|---|
| `spk_head.pt` | Speaker head: learned weights over the 13 wav2vec2 layers → LayerNorm → 1536→256 → ReLU → 256→192 voiceprint (AAM-softmax, m 0.25, s 30) | LibriSpeech train-clean-100 (211 speakers), every clip clean and through a G.711 phone line. 3.96% EER on 40 held-out speakers |
| `deepfake/all.pt` | Deepfake head (human vs synthetic), 3-seed ensemble, trained on all six teammates | Our team's recordings vs their XTTS-v2 clones, with pause gating, noise, mic tilt, phone line and speed variation |
| `deepfake/fold_s01.pt` … `fold_s06.pt` | The same head trained with one teammate left out; the app scores each teammate's demo call with the head that never heard them | as above, five teammates each |
| `deepfake/calib.json` | Platt calibration: `p = sigmoid(a · logit + b)`, fitted on out-of-fold scores | |

All heads read the same frozen `facebook/wav2vec2-base` features (13 layers, mean + std pooled over 3 s), which download on first run.

**Not included:** `voiceprints.npz`. Those are the voiceprints of our team members, i.e. biometric data. Enrol your own voices in the app's **Enrol** tab or with `step5_speaker.py`.

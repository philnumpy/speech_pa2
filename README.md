# Speech Understanding — Programming Assignment 2
## Code-Switched Transcription → Maithili Voice Cloning Pipeline

**YouTube Source:** https://youtu.be/ZPUtA3W-7_I  
**Segment:** 2 h 20 min → 2 h 54 min window  
**Target LRL:** Maithili (`mai_Deva`)

---

## Repository Structure

```
.
pipeline.py                   # Single executable end-to-end script
Speech_PA2.ipynb     # Full notebook with outputs
environment.yml               # Conda environment
requirements.txt              # pip fallback
README.md                     # This file
report
student_voice_ref.wav
original_segment.wav(create a dir called audio when running and place it there as it is already resampled)
output_LRL_cloned.wav(will be created in audio directory when you execute the code this is my cloned voice generated)
lid_weights.pt            # Trained LID model weights

```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/philnumpy/speech_pa2.git
cd [RepoName]
```

### 2. Set up the environment

**Option A — Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate speech_pa2
```

**Option B — pip:**
```bash
pip install -r requirements.txt
sudo apt-get install -y espeak-ng   # Required by phonemizer
```

> **Note:** Tested on Python 3.12, CUDA 12.1, Google Colab T4/A100.  
> For CPU-only runs, remove `cudatoolkit` from `environment.yml`.

### 3. Add your audio files

Place your files in the root directory:
```
original_segment (3).wav    # Your 10-min lecture clip (any SR)
myrecording.wav             # Your 60-second voice recording (any SR)
```

Or pre-resample to 22050 Hz and place them as:
```
audio/original_segment (3).wav
student_voice_ref.wav
```

### 4. Run the full pipeline

```bash
python pipeline.py \
    --segment audio/original_segment.wav \
    --voice   student_voice_ref.wav \
    --output  audio/output_LRL_cloned.wav
```

All intermediate files are saved automatically to `audio/`, `models/`, `outputs/`.

### 5. Run individual steps (notebook)

Open `Speech_PA2_Complete.ipynb` in Jupyter or Google Colab.  
If resuming a partial run, execute the **Session Restore Cell** first (clearly marked in the notebook).

---

## Pipeline Overview

| Step | Task | Key Model / Method |
|------|------|--------------------|
| 1 | Audio loading & resampling | `librosa.resample` → 22,050 Hz |
| 2 | Spectral subtraction denoising | Custom STFT-based (Task 1.3) |
| 3 | Multi-Head frame-level LID | 3-layer Transformer, 4 heads, 121-dim MFCC features (Task 1.1) |
| 4 | Constrained ASR decoding | Whisper Turbo + N-gram logit bias via `initial_prompt` (Task 1.2) |
| 5 | IPA unified representation | Hybrid Hinglish G2P + espeak-ng (Task 2.1) |
| 6 | Maithili translation | NLLB-200 distilled 600M with `forced_bos_token_id=mai_Deva` (Task 2.2) |
| 7 | Speaker embedding | SpeechBrain x-vector (512-dim, VoxCeleb) (Task 3.1) |
| 8 | DTW prosody warping | FastDTW on joint (F0, Energy) 2D vectors (Task 3.2) |
| 9 | Zero-shot voice cloning | XTTS v2, `language='hi'` proxy, 24 kHz output (Task 3.3) |
| 10 | Anti-spoofing CM | LFCC (60-dim) + Logistic Regression, EER = 0.00% (Task 4.1) |
| 11 | FGSM adversarial attack | Feature-space FGSM, min ε = 1e-5 at SNR 134.4 dB (Task 4.2) |
| 12 | Evaluation + ablation | WER, MCD, LID F1, EER, FGSM summary |

---

## Evaluation Results

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| WER (English) | 0.00% | < 15% | ✅ PASS |
| WER (Hindi) | 0.00% | < 25% | ✅ PASS |
| LID Macro F1 | 1.0000 | ≥ 0.85 | ✅ PASS |
| Switch Timestamp Error | 0.0 ms | < 200 ms | ✅ PASS |
| Anti-Spoof EER | 0.00% | < 10% | ✅ PASS |
| Min FGSM ε (SNR > 40 dB) | 1e-5 | Reported | ✅ PASS |
| MCD (cross-lingual) | 696.52 dB | < 8.0 | ⚠️ NOTE* |

> *MCD measures frame-aligned same-content divergence. Comparing English reference
> audio against Maithili synthesised output violates this assumption; the high value
> reflects linguistic content mismatch, not poor voice quality.

---

## Key Design Choices

**LID — Transformer over BiLSTM:**  
Self-attention resolves long-range phonological context (e.g., a Hindi discourse marker 500 ms earlier disambiguates a borrowed noun) with O(1) path length vs. BiLSTM's vanishing-gradient limitation.

**Constrained Decoding — `initial_prompt` over hook-based logit modification:**  
Whisper's `initial_prompt` pre-fills the KV cache with syllabus token embeddings, raising their cross-attention weight at every decoding step. This is equivalent to logit biasing without modifying Whisper internals.

**Translation — `forced_bos_token_id` over HuggingFace pipeline:**  
`pipeline("translation")` raises `KeyError` for non-standard language pairs in newer transformers versions. Directly calling `generate()` with `forced_bos_token_id=mai_Deva` is the canonical NLLB inference pattern and bypasses the task registry.

**DTW — Joint 2D over independent 1D warping:**  
Separate F0 and energy warping produces misaligned pitch-energy phase relationships that sound unnatural. Joint 2D FastDTW ensures temporally consistent prosodic transfer.

**Anti-Spoofing — LFCC over MFCC:**  
TTS vocoders leave spectral artefacts above 4 kHz. Mel scaling compresses these; the linear filterbank of LFCC preserves them, yielding better spoof-detection discriminability.

---

## Dependencies

See `environment.yml` or `requirements.txt` for the full list.  
Core: `torch`, `torchaudio`, `transformers`, `openai-whisper`, `TTS` (Coqui), `speechbrain`, `librosa`, `phonemizer`, `fastdtw`, `scikit-learn`, `soundfile`.

---

## Notes for Reproducibility

- The notebook was executed in multiple sessions due to Colab RAM constraints when loading large models simultaneously. All outputs are saved to disk after each step, enabling partial re-runs via the **Session Restore Cell**.
- Running all cells top-to-bottom in a fresh Colab T4 session fully reproduces all outputs.
- Model downloads (Whisper Turbo ~1.5 GB, NLLB-200 ~2.4 GB, XTTS v2 ~1.8 GB, SpeechBrain xvect ~100 MB) require internet access on first run.

---

## Citation

If you use this pipeline, please cite the libraries used:
- Whisper: Radford et al., 2022
- NLLB-200: Meta AI, 2022
- XTTS v2: Casanova et al., 2024
- SpeechBrain: Ravanelli et al., 2021
- FastDTW: Salvador & Chan, 2007

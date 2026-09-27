# QuietLab — offline audio denoiser

A local Python app for learning a steady background-noise profile from selected quiet sections, applying adjustable spectral reduction, comparing the result, and exporting a WAV.

## Features

- Upload MP3, WAV, M4A, FLAC, OGG, and AAC (compressed formats may need FFmpeg installed).
- Suggest quiet sections automatically with an adaptive short-time energy detector; inspect and choose suggestions.
- Add and remove noise-only intervals manually.
- Set reduction strength, retained noise floor, and time/frequency smoothing.
- Listen to original and processed audio.
- Compare waveform and average spectrum; inspect a time-frequency spectrogram.
- Export the result as 16-bit PCM WAV.
- Stereo uses a linked reduction mask to help preserve the stereo image.

## Run locally

Requires Python 3.10 or newer.

### Windows

    py -m venv .venv
    .venv\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

### macOS / Linux

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

Streamlit opens the app in your browser, normally at http://localhost:8501. Audio processing runs locally.

## First-use guide

1. Upload a recording.
2. Click **Suggest quiet regions** and review the suggestions. Or enter start/end times to add a region yourself.
3. Select only sections that contain background noise without speech. Multiple regions are useful if the appliance sound changes.
4. Set reduction strength and click **Process audio**.
5. Compare the audio and graphs. Lower strength if the voice sounds watery, metallic, or muffled.
6. Download the cleaned WAV.

## What the controls mean

- **Reduction strength** controls how much estimated noise power is subtracted. Higher settings usually remove more noise and may cause more speech artifacts.
- **Minimum retained level** sets a floor under the attenuation. A higher floor sounds more natural but leaves more background noise.
- **Time/frequency smoothing** softens abrupt changes in the mask and reduces isolated musical-noise tones.

## High-level processing

Audio is divided into overlapping Hann-windowed frames and transformed using an STFT. The app takes a median noise magnitude estimate at each frequency across all selected samples. It then builds a soft Wiener-style gain mask from the ratio of residual signal power to observed power, smooths that mask over time and frequency, applies the same mask to each channel, and reconstructs the waveform by overlap-add.

The quiet-section detector is a convenience, not a voice transcription model. It detects low-energy intervals and may mistake quiet speech or breaths for silence. Review and edit all suggestions. The denoiser assumes noise is reasonably steady; rapidly changing sounds are harder to remove cleanly. Denoising cannot perfectly distinguish speech and noise where they overlap in frequency.

## Limitations

- Output is WAV; MP3 export is intentionally omitted to avoid a separate MP3 encoder dependency.
- There is no destructive editing: the uploaded source remains unchanged.
- Very long recordings consume memory because the app holds the source and processed audio in memory.
- If MP3/M4A/AAC decoding fails, install FFmpeg and retry, or convert the source to WAV.

## Project files

- app.py: Streamlit interface.
- denoiser.py: audio decoding, quiet-region detection, spectral processing, and plot data.
- requirements.txt: Python dependencies.

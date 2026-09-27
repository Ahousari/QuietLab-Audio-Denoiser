"""Offline speech denoiser: Streamlit UI."""
from __future__ import annotations

import io
import hashlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from denoiser import (
    detect_quiet_regions, denoise_audio, frequency_curves, load_audio,
    make_wav_bytes, waveform_envelope,
)

st.set_page_config(page_title="QuietLab · Audio denoiser", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.5rem;}
.hero {padding:1.2rem 1.5rem; border-radius:16px; background:linear-gradient(115deg,#101b35,#243b63); color:#f4f7ff; margin-bottom:1rem}
.hero h1 {margin:0; font-size:2rem}.hero p {margin:.45rem 0 0;opacity:.82}
</style>
<div class="hero"><h1>QuietLab</h1><p>Learn the room tone, reduce it, and compare what changed.</p></div>
""", unsafe_allow_html=True)

def init():
    st.session_state.setdefault("regions", [])
    st.session_state.setdefault("cleaned", None)
    st.session_state.setdefault("cleaned_key", None)
init()

upload = st.file_uploader("Choose an audio file", type=["mp3","wav","m4a","flac","ogg","aac"])
if not upload:
    st.info("Upload an audio file to begin. MP3 decoding may require FFmpeg on some systems; WAV works without it.")
    st.markdown("**Workflow:** select quiet background-only intervals → estimate the noise → preview denoising → export WAV.")
    st.stop()

try:
    audio, sr = load_audio(upload.getvalue())
except Exception as e:
    st.error(f"Could not decode this file: {e}")
    st.caption("Try installing FFmpeg, or convert the source to WAV and upload it.")
    st.stop()

duration = len(audio) / sr
channels = audio.shape[1] if audio.ndim == 2 else 1
st.caption(f"**{upload.name}** · {duration:.1f} s · {sr:,} Hz · {channels} channel{'s' if channels != 1 else ''}")

source_id = (upload.name, hashlib.sha1(upload.getvalue()).hexdigest())
if st.session_state.get("source_id") != source_id:
    st.session_state.source_id = source_id
    st.session_state.regions = []
    st.session_state.cleaned = None
    st.session_state.candidates = []
    for key in ("candidate_choice", "remove_regions", "manual_start", "manual_end"):
        st.session_state.pop(key, None)

left, right = st.columns([1.55, 1])
with left:
    st.subheader("1 · Mark background-only audio")
    c1,c2=st.columns([1,1])
    with c1:
        if st.button("Suggest quiet regions", use_container_width=True):
            st.session_state.candidates = detect_quiet_regions(audio, sr)
    candidates=st.session_state.get("candidates", [])
    if candidates:
        labels=[f"{a:.2f}–{b:.2f} s  ({b-a:.1f} s)" for a,b in candidates]
        chosen=st.multiselect("Choose suggested regions to use as noise samples", labels,
                              default=labels, key="candidate_choice")
        if st.button("Use selected suggestions"):
            st.session_state.regions=[candidates[labels.index(x)] for x in chosen]
            st.session_state.cleaned=None
    with c2:
        st.write("Add a region manually")
        a_col,b_col,add_col=st.columns([1,1,.65])
        with a_col: start=st.number_input("Start (s)",0.0,max_value=float(duration),step=.1,key="manual_start")
        with b_col: end=st.number_input("End (s)",0.0,max_value=float(duration),step=.1,key="manual_end")
        with add_col:
            st.write("")
            add=st.button("Add",use_container_width=True)
        if add:
            if end <= start: st.warning("End must be after start.")
            else:
                st.session_state.regions.append((float(start),float(end)))
                st.session_state.cleaned=None
    if st.session_state.regions:
        st.write("Selected noise samples")
        labels=[f"{a:.2f}–{b:.2f} s ({b-a:.1f} s)" for a,b in st.session_state.regions]
        remove=st.multiselect("Remove regions",labels,key="remove_regions")
        if remove and st.button("Remove selected"):
            st.session_state.regions=[r for i,r in enumerate(st.session_state.regions) if labels[i] not in remove]
            st.session_state.cleaned=None
        st.caption(f"{len(st.session_state.regions)} region(s), {sum(b-a for a,b in st.session_state.regions):.1f} s total")
    else:
        st.warning("Select or add at least one section containing only background noise.")

    st.subheader("2 · Tune reduction")
    strength=st.slider("Reduction strength",0.0,12.0,1.0,.05,help="Higher values suppress more noise, but can make speech sound watery or muffled.")
    floor_db=st.slider("Minimum retained level (dB)",-200,-1,-12,1,help="A higher floor sounds more natural but leaves more noise.")
    smooth=st.slider("Time/frequency smoothing",0.0,1.0,.55,.05)
    c1,c2=st.columns(2)
    with c1:
        if st.button("Process audio",type="primary",use_container_width=True,disabled=not st.session_state.regions):
            with st.spinner("Estimating noise and processing audio…"):
                st.session_state.cleaned=denoise_audio(audio,sr,st.session_state.regions,strength,floor_db,smooth)
                st.session_state.cleaned_key=(st.session_state.source_id,tuple(st.session_state.regions),strength,floor_db,smooth)
    cleaned=st.session_state.cleaned
    with c2:
        if st.button("Reset processed audio",use_container_width=True):
            st.session_state.cleaned=None
            cleaned=None

with right:
    st.subheader("Listen")
    st.audio(make_wav_bytes(audio,sr),format="audio/wav")
    st.caption("Original")
    if st.session_state.cleaned is not None:
        st.audio(make_wav_bytes(st.session_state.cleaned,sr),format="audio/wav")
        st.caption("Processed")
        st.download_button("Download cleaned WAV",make_wav_bytes(st.session_state.cleaned,sr),
                           file_name=f"{upload.name.rsplit('.',1)[0]}_cleaned.wav",mime="audio/wav",use_container_width=True)
    else:
        st.caption("Process audio to hear and download the result.")

st.divider()
st.subheader("3 · Compare in time and frequency")
view=st.radio("Display",["Waveform + spectrum","Spectrogram"],horizontal=True)
plot_a=audio.mean(axis=1) if audio.ndim==2 else audio
plot_b=(st.session_state.cleaned.mean(axis=1) if st.session_state.cleaned is not None and st.session_state.cleaned.ndim==2 else st.session_state.cleaned)
if view=="Waveform + spectrum":
    fig=make_subplots(rows=2,cols=1,shared_xaxes=False,vertical_spacing=.14,
        subplot_titles=("Time domain · waveform (selected noise sections shaded)","Frequency domain · average level"))
    t,y=waveform_envelope(plot_a,sr)
    fig.add_trace(go.Scatter(x=t,y=y,mode="lines",name="Original",line=dict(color="#638bff",width=1)),row=1,col=1)
    if plot_b is not None:
        tb,yb=waveform_envelope(plot_b,sr)
        fig.add_trace(go.Scatter(x=tb,y=yb,mode="lines",name="Processed",line=dict(color="#25c89a",width=1)),row=1,col=1)
    for a,b in st.session_state.regions:
        fig.add_vrect(x0=a,x1=b,fillcolor="#f2aa4c",opacity=.22,line_width=0,row=1,col=1)
    f,orig,noise,proc=frequency_curves(plot_a,sr,st.session_state.regions,plot_b)
    fig.add_trace(go.Scatter(x=f,y=orig,name="Original",line=dict(color="#638bff")),row=2,col=1)
    if noise is not None: fig.add_trace(go.Scatter(x=f,y=noise,name="Estimated noise",line=dict(color="#f2aa4c",dash="dot")),row=2,col=1)
    if proc is not None: fig.add_trace(go.Scatter(x=f,y=proc,name="Processed",line=dict(color="#25c89a")),row=2,col=1)
    fig.update_xaxes(title_text="Time (s)",row=1,col=1)
    fig.update_xaxes(title_text="Frequency (Hz)",type="log",row=2,col=1)
    fig.update_yaxes(title_text="Amplitude",row=1,col=1)
    fig.update_yaxes(title_text="Level (dBFS)",row=2,col=1)
    fig.update_layout(height=650,legend=dict(orientation="h",y=1.08),margin=dict(t=60,b=30))
    st.plotly_chart(fig,use_container_width=True)
else:
    st.caption("Color shows level (dBFS) by frequency and time. Orange shaded regions were selected as the noise sample.")
    # Decimated STFT for fast interactive rendering
    import scipy.signal as sig
    nperseg=min(1024,max(256,2**int(np.floor(np.log2(max(256,sr*.04))))))
    f,t,z=sig.stft(plot_a,fs=sr,nperseg=nperseg,noverlap=nperseg*3//4)
    db=20*np.log10(np.maximum(np.abs(z),1e-7))
    fig=go.Figure(go.Heatmap(x=t,y=f,z=db,zmin=-100,zmax=-25,colorscale="Viridis",colorbar=dict(title="dBFS")))
    for a,b in st.session_state.regions: fig.add_vrect(x0=a,x1=b,fillcolor="#f2aa4c",opacity=.2,line_width=0)
    fig.update_yaxes(title="Frequency (Hz)",range=[0,min(sr/2,10000)])
    fig.update_xaxes(title="Time (s)")
    fig.update_layout(height=500,margin=dict(t=25,b=35))
    st.plotly_chart(fig,use_container_width=True)

with st.expander("How the denoiser works"):
    st.markdown("""
The app analyzes short, overlapping audio frames with an **STFT**. It estimates the typical magnitude at each frequency from your selected noise-only sections, then applies a soft Wiener-style gain to each frame. The frames are overlapped and added back together to produce audio.

The estimated profile is shown as a dotted line in the spectrum view. This assumes the appliance noise is reasonably steady. If it changes over time, select several representative quiet regions. Filtering cannot always distinguish speech from noise at the same frequencies, so compare the result by ear and lower the strength if the voice sounds watery.
""")

import streamlit as st
import sys
import os

# Ensure src is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.fusion_predict import run_fusion_pipeline

st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Real-Time Audio Emotion Recognition")
st.write("Click the button below to record audio and analyze emotion.")

# Button to start pipeline
if st.button("🎤 Record & Analyze Emotion"):

    with st.spinner("Recording and analyzing..."):

        result = run_fusion_pipeline()

        # Extract values safely from result dictionary
        text = result["text"]
        audio_emotion = result["audio_emotion"]
        text_emotion = result["text_emotion"]
        final_emotion = result["final_emotion"]

    st.success("Analysis Complete ✅")

    st.subheader("📝 Transcribed Text")
    st.write(text)

    st.subheader("🎧 Audio Emotion")
    st.write(audio_emotion)

    st.subheader("💬 Text Emotion")
    st.write(text_emotion)

    st.subheader("🎯 Final Emotion")
    st.success(final_emotion)
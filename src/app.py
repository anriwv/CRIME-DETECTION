import os
from pathlib import Path

import streamlit as st

st.set_page_config(layout="centered", page_title="CRIME-DETECTION")
st.title("CRIME DETECTION")
st.caption("Real-time detection • Dataset viewer • Info")


tab1, tab2, tab3 = st.tabs(["Real Time", "Gallery", "Info"])

with tab1:
    st.subheader("Turn on camrra")
    st.write("")


with tab2:
    st.title("Crime Dataset Video Gallery")

    BASE_DIR = Path(__file__).resolve().parent
    ROOT_FOLDERS = ["Train", "Test"]

    root_choice = st.selectbox("Select dataset:", ROOT_FOLDERS)
    root_path = BASE_DIR / root_choice

    subfolders = sorted([f.name for f in root_path.iterdir() if f.is_dir()])
    subfolder_choice = st.selectbox("Select category:", subfolders)

    sub_path = root_path / subfolder_choice

    # Load videos
    videos = sorted([v for v in sub_path.iterdir() if v.suffix.lower() == ".mp4"])

    st.write(f"### {len(videos)} examplevideos")

    cols = st.columns(2)

    for i, video in enumerate(videos):
        with cols[i % 2]:
            st.video(str(video))
            st.caption(video.name)

with tab3:
    st.subheader("ℹ️ Info")
    st.write("")

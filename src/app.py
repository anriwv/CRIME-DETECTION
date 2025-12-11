import os
import time
from pathlib import Path
from posixpath import sep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(layout="centered", page_title="Crime Detection G7")
st.title("Crime Detection")
st.caption("Real-time detection & Dataset viewer & Info")


@st.cache_resource
def load_pipelines():
    pipe1 = pipeline(
        "image-classification", model="dima806/crime_cctv_image_detection", device=0
    )
    pipe2 = pipeline(
        "image-classification",
        model="dima806/crime_type_cctv_image_detection",
        device=0,
    )
    return pipe1, pipe2


def resize(img, target_size=64):
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x = (target_size - img.width) // 2
    y = (target_size - img.height) // 2
    new_img.paste(img, (x, y))
    return new_img


def analyze_frame(frame, pipe1, pipe2):
    new_img = resize(frame)
    res1 = pipe1(new_img)
    crime_score = 0
    for r in res1:
        if r["label"] == "Crime":
            crime_score = r["score"]
            break
    if crime_score > 0.5:
        res2 = pipe2(new_img)
        predicted_label = res2[0]["label"] if len(res2) > 0 else "Unknown"
        return crime_score, predicted_label
    return crime_score, "Normal"


pipe1, pipe2 = load_pipelines()

tab1, tab2, tab3 = st.tabs(["Real Time", "Gallery", "Info"])

with tab1:
    st.subheader("Turn on Camera, prediction with 3 fps")
    run = st.checkbox("Enable Camera", key="rt")
    frame_placeholder = st.empty()

    # container with ph
    container = st.container(border=True)
    with container:
        info_placeholder = st.empty()
        image_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
        else:
            while st.session_state["rt"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frame_64 = resize(pil_frame, 64)
                crime_score, label = analyze_frame(pil_frame, pipe1, pipe2)

                frame_placeholder.image(
                    frame,
                    caption=f"{label} ({crime_score:.2f})",
                    channels="RGB",
                    width=700,
                )

                # upd placeholders
                if label != "Normal":
                    info_placeholder.markdown(
                        f"""<span style='color: orange; font-size: 30px; font-weight: bold;'>
                        Crime Score: {crime_score:.2f},
                        Label: {label}</span>""",
                        unsafe_allow_html=True,
                    )
                else:
                    info_placeholder.write(
                        f"Crime Score: {crime_score:.2f}, Label: {label}"
                    )
                image_placeholder.image(
                    frame_64,
                    caption="64×64 Model Input",
                    width=200,
                )

                time.sleep(1 / 3)
            cap.release()

with tab2:
    st.subheader("Crime Dataset Video Gallery")

    BASE_DIR = Path(__file__).resolve().parent
    ROOT_FOLDERS = ["Train", "Test"]

    root_choice = st.selectbox("Select dataset:", ROOT_FOLDERS)
    root_path = BASE_DIR / root_choice

    subfolders = sorted([f.name for f in root_path.iterdir() if f.is_dir()])
    subfolder_choice = st.selectbox("Select category:", subfolders)

    sub_path = root_path / subfolder_choice

    # videos
    videos = sorted([v for v in sub_path.iterdir() if v.suffix.lower() == ".mp4"])

    st.write(f"### {len(videos)} example videos")

    cols = st.columns(2)

    for i, video in enumerate(videos):
        with cols[i % 2]:
            st.video(str(video))
            st.caption(video.name)


with tab3:
    st.subheader("Dataset Statistics")

    stat = pd.read_csv("stat.csv", sep=";", index_col=0)
    st.dataframe(stat, use_container_width=True)

    train_df = pd.read_csv("train_df.csv", sep=";", index_col=0)
    test_df = pd.read_csv("test_df.csv", sep=";", index_col=0)

    # Samples
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training videos", len(train_df))
    with col2:
        st.metric("Test videos", len(test_df))

    st.markdown("### Train Dataset Crime Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts = train_df["label"].value_counts()
    sns.countplot(
        y="label", data=train_df, order=label_counts.index, ax=ax, palette="viridis"
    )
    ax.set_xlabel("Count", fontsize=14)
    ax.set_ylabel("Crime Type", fontsize=14)
    ax.set_title("Distribution of Crime Types in Training Data", fontsize=18)

    # count labels on bars
    for i, v in enumerate(label_counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # train vs test video counts
    st.markdown("### Train vs Test Class Distribution Comparison")

    train_counts = train_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()

    compare_df = (
        pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0).astype(int)
    )

    st.dataframe(compare_df, use_container_width=True)

    # Train vs Test Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    compare_df.plot(kind="bar", ax=ax, color=["#5ecc62", "#ff7f0e"], alpha=0.5)

    ax.set_xlabel("Crime Type", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Train vs Test Distribution", fontsize=18)
    ax.legend(title="Dataset", fontsize=14)
    ax.grid(axis="y", alpha=0.4)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

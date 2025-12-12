# Deep Learning for Real-World Crime Recognition G7

## Team Members
- Mart Rõbin - [martrobin-inf](https://github.com/martrobin-inf)
- Artjom Geimanen - [Artjom7](https://github.com/Artjom7)
- Anri Sokolov - [anriwv](https://github.com/anriwv)
---

## Motivation and original goal
Crime detection in real-world surveillance footage is a critical task for improving public safety.
Surveillance videos are long, untrimmed, and often contain complex scenes with many people.
Our original goal was therefore ambitious:

1. Train a model to classify activities based on frames.
2. Predict potential criminal activity

---

## Datasets

To achieve this, we used the UCF Crime dataset (14 classes) containing 1,900 videos = ~1.4M frames.
The Kaggle version of the dataset provided:
- extracted every 10th frame
- 64x64 px .png image
- train & test sets

UCF Crime Dataset: <br>
https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset/data

---

## Revised Goal After Experimental Failures

During experimentation, we discovered several major challenges:
- strong class imbalance
- low frame resolution (64×64)
- large variations in video length (from 11 up to 97651 frames)
- some classes dominated by a single video
(total Explosion train frames- 18753, and "Explosion046" 14190). <br>

As a result, all self-trained models (CNN, LSTM, RNN) either underperformed or showed unstable training, despite extensive attempts at tuning.

we trained our models using:
- our own personal PCs
- Kaggle Notebooks
- Google Colab with GPU


## Final Goals:

-   [X] **Use** a model to classify activity types from extracted frames (image-level classification).
-   [X] **Use these frame-level predictions** to fla criminal activity in videos.
-   [X] Do analys of dataset
-   [X] Use models in demo app.


## Repository Structure
- /docs — report
- /notebooks — exploratory analysis
- /src — codebase for streamlit app
- /data — not included

## Report
See: docs/[G7_report.pdf](https://github.com/anriwv/CRIME-DETECTION/blob/main/docs/G7_report.pdf)

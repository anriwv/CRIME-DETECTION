# Deep Learning for Real-World Crime Recognition G7

## Team Members
- Mart Rõbin - [martrobin-inf](https://github.com/martrobin-inf)
- Artjom Geimanen - [Artjom7](https://github.com/Artjom7)
- Anri Sokolov - [anriwv](https://github.com/anriwv)
---

## Motivation and original goal
Crime detection in real-world surveillance footage is a critical task for improving public safety.
Surveillance videos are long, untrimmed, and often contain complex scenes with many people, making manual monitoring inefficient.

Our original goals were ambitious:

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
- large variation in video length (from 11 up to 97651 frames)
- some classes dominated by a single video
  - *Explosion* class: 18,753 training frames
  - *Explosion046* alone: 14,190 frames

As a result, all self-trained models (CNN, LSTM, RNN) either underperformed or showed unstable training, despite extensive tuning.

We trained our models using:
- our own personal PCs
- Kaggle Notebooks
- Google Colab with GPU
---

## Final Goals:
After revising the scope of the project, the final goals became:

-   [x] **Use** a model to classify activity types from extracted frames (image-level classification).
-   [x] **Use frame-level predictions** to flag potential criminal activity in videos
-   [X]  Perform dataset analysis
-   [X] Use models in demo application.
---

## Repository Structure
### `docs/` - Project Report
Contains the final PDF report submitted for Homework 10.

- `[G7_report.pdf](https://github.com/anriwv/CRIME-DETECTION/blob/main/docs/G7_report.pdf).pdf` - Final project report

---

### `notebooks/` - Experiments & Analysis
Contains Jupyter notebooks used for data exploration, preprocessing and model training.
Not all experimental code is included (failed architectures and approaches are omitted), but the notebooks demonstrate the main research workflow.

Key notebooks and files:

- `Distribution.ipynb` - Dataset imbalance analysis and video statistics per class
- `counter.ipynb` - Frame and video count statistics per class
- `ViT.ipynb` - Image and video classification using a pretrained Vision Transformer (Hugging Face)
- `efficientnet_b0_mart.ipynb` - EfficientNet-B0 training
- `fast_reading.ipynb` - Creates dictionaries of image paths per video and outputs sorted `.pkl` files (used frequently)
- `runAllCrimeTrain.ipynb` - Generates predictions for all crime samples in training videos (might be used to train another model || changes to calculate a test_videos to get know model accuracy)
- `format_matters.ipynb` - Tests how input resolution and image format affect predictions
- `tile_images.ipynb` - Combines 16 frames (4×4 grid) into a single image for CNN input (Might be changet to 8*8 images)
  - Related dataset: https://www.kaggle.com/datasets/anrisokolov/ucf-crime-dataset-44-images-for-each-video
- `video_animation.ipynb` - Visualization of videos by name
- `videoutils.py` - Functions for frame-based video visualization
- `examplesForSite.ipynb` - Generates example videos used in the Streamlit demo

---

### `src/` - Streamlit Demo Application
Contains the final runnable application.

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies (used for deployment on Streamlit Community Cloud)
- `packages.txt` - dependencies (used for deployment on Streamlit Community Cloud)
- `Train/`, `Test/` - Example inputs for the demo
- `train_df.csv`, `test_df.csv`, `stat.csv` - Data used by the application

---

## How to Reproduce the Analysis

To replicate the experiments and analysis performed in this project:

1. **Download the dataset** from Kaggle
2. **Run `fast_reading.ipynb`** or download the pre-generated `.pkl` files to prepare sorted frame paths
3. **Train models** using the provided notebooks (`efficientnet_b0_mart.ipynb`)
4. **Run analysis notebooks**:
   - `Distribution.ipynb`
   - `counter.ipynb`
   - `format_matters.ipynb`
   - `video_animation.ipynb`
   - `examplesForSite.ipynb`
5. **Run prediction pipelines**:
   - `ViT.ipynb`
   - `runAllCrimeTrain.ipynb` (+ extend to test videos to calculate accuracy)
6. **Start the demo application**:
   ```streamlit run src/app.py```


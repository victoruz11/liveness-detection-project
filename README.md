# Face Liveness Detection Project

This project follows the pipeline:
- video input
- extract **10 frames per video**
- detect and crop faces automatically
- keep frames grouped per video
- train a **CNN + LSTM** model
- classify as **real** or **fake**

## Folder structure

```text
liveness_detection_project/
├── data/
│   ├── raw_videos/
│   │   ├── Real/                               ← real face videos
│   │   │   └── Selfies/                        ← images only, skipped automatically
│   │   ├── 3D_paper_mask_/                     ← fake
│   │   ├── Cutout_attacks/                     ← fake
│   │   ├── Latex_mask/                         ← fake
│   │   ├── Printouts/                          ← fake
│   │   │   ├── WITH CUTOUT/
│   │   │   └── WITHOUT CUTOUT/
│   │   ├── Replay_display_attacks/             ← fake
│   │   ├── Silicone_mask/                      ← fake
│   │   ├── Textile 3D Face Mask Attack Sample/ ← fake
│   │   └── Wrapped_3D_paper_mask/              ← fake
│   ├── extracted_frames/
│   └── face_sequences/
├── models/
├── src/
├── requirements.txt
└── README.md
```

## How folders are classified

The script `extract_frames.py` uses an explicit mapping to assign each folder to `real` or `fake`.
If you add a new attack folder to `raw_videos/`, you must also add it to the `FOLDER_LABELS`
dictionary at the top of `src/extract_frames.py`, otherwise it will be skipped with a warning.

The `Real/Selfies/` subfolder contains `.jpg` images, not videos. It is skipped automatically.

## Before you start

Make sure Python is installed on your laptop.

## Step-by-step setup on Windows PowerShell

### 1. Open PowerShell in this project folder
Use `cd` to move into the folder that contains this project.

### 2. Create a virtual environment
```powershell
python -m venv venv
```

### 3. Activate it
```powershell
.\venv312\Scripts\Activate
```

If activation is blocked, run this once in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate again.

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

## Run the project step by step

### Step 1. Extract exactly 10 frames from each video
```powershell
python src\extract_frames.py
```

Output goes to:
```text
data/extracted_frames/
├── real/
└── fake/
```

At the end it prints a summary like:
```
Done. Real: 95 | Fake: 112 | Skipped: 3
```

### Step 2. Detect and crop faces using OpenCV
```powershell
python src\crop_faces_opencv.py
```

Output goes to:
```text
data/face_sequences/
├── real/
└── fake/
```

Important:
- only video folders with at least 7 usable face crops are kept
- folders with fewer valid faces are skipped automatically
- each kept folder is padded or trimmed to exactly 10 frames

### Step 3. Check a few output folders manually
Open a few folders in:

```text
data/face_sequences/real/
data/face_sequences/fake/
```

Check that:
- the face is visible
- the crop is not broken
- each kept folder has exactly 10 images
- each folder still represents one single video

### Step 4. Train the model
```powershell
python src\train.py
```

Best model is saved to:
```text
models/best_model.pth
```

Split indices are also saved to:
```text
models/split_indices.json
```

### Step 5. Evaluate the model
```powershell
python src\evaluate.py
```

This prints:
- classification report
- confusion matrix
- APCER
- BPCER
- ACER

### Step 6. Webcam demo
```powershell
python src\webcam_demo.py
```

### Step 7. Launch Streamlit
```powershell
streamlit run app.py
```

Press `q` to quit.

## One-command run
If your dataset is already in place and dependencies are installed, you can run:

```powershell
python run_pipeline.py
```

That will run:
1. frame extraction
2. face cropping
3. training
4. evaluation

## To redo frame extraction and face cropping

If you want to clear the processed data and start fresh:

```powershell
Remove-Item -Recurse -Force "data\extracted_frames\*"
Remove-Item -Recurse -Force "data\face_sequences\*"
```

Then rerun Steps 1 and 2.

## Common issues

### `python` not found
Try:
```powershell
py --version
```
If `py` works, use `py` instead of `python`.

### `ModuleNotFoundError`
Make sure the virtual environment is activated and dependencies were installed.

### A folder is being skipped with a WARNING
If you see `WARNING: Unknown folder '...' — not in FOLDER_LABELS`, open `src/extract_frames.py`
and add the folder name and its label to the `FOLDER_LABELS` dictionary at the top of the file.

### Not enough valid face folders
That usually means face detection failed on many extracted frames.
Check your raw videos and cropped outputs.

### Training starts but does not save a model
Make sure `models/` exists. This starter project already includes that folder.

### `evaluate.py` raises FileNotFoundError for split_indices.json
You must run `train.py` before `evaluate.py`. Training saves the split indices that evaluation depends on.

## Recommended workflow

1. Create and activate `venv`
2. Install requirements
3. Confirm your dataset folders are inside `data/raw_videos/` as shown in the folder structure above
4. Run `python src\extract_frames.py`
5. Run `python src\crop_faces_opencv.py`
6. Inspect the output folders manually
7. Run `python src\train.py`
8. Run `python src\evaluate.py`
9. Try webcam demo only after the model is saved







## AFTER DOWNLOADING THE ZIP


### 1. Activate it
```powershell or terminal
.\venv312\Scripts\Activate
```

If activation is blocked, run this once in PowerShell/terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate again.

### 2. Run webcam_demo.py
```powershell or terminal
 python src\webcam_demo.py      
```

















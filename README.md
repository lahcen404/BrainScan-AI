# BrainScan AI

BrainScan AI is an educational brain MRI image-classification project using a TensorFlow/Keras convolutional neural network (CNN). The model classifies images into four dataset categories:

- Glioma
- Meningioma
- No tumor
- Pituitary

This project is for learning and demonstration only. It is not a clinically validated diagnostic system and must not be used to diagnose patients or guide treatment.

## Project Structure

```text
BrainScanAI/
├── app/
│   └── app.py                    # Streamlit interface
├── data/
│   └── brain_mri/
│       └── Data/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
├── models/
│   └── brainscan_model.h5        # Model used by Streamlit
├── notebooks/
│   └── brainscan_eda.ipynb       # EDA, preprocessing, training, evaluation
├── reports/
├── src/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- pip
- A CPU is sufficient; a GPU is optional

The project uses TensorFlow, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Jupyter, and Streamlit. The exact packages are listed in `requirements.txt`.

## Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Dataset Setup

Place the dataset at:

```text
data/brain_mri/Data/
```

Each class must have its own directory:

```text
data/brain_mri/Data/glioma/
data/brain_mri/Data/meningioma/
data/brain_mri/Data/notumor/
data/brain_mri/Data/pituitary/
```

The notebook uses the relative path `../data/brain_mri/Data` because it is stored in the `notebooks/` directory.

MRI images and trained model files are excluded by `.gitignore`. They must be added locally after cloning or downloading the project.

## Model

The deployed model is:

```text
models/brainscan_model.h5
```

The `.h5` extension is used because the project specification requires `.h5` or `.pt`. This project uses TensorFlow/Keras, so `.h5` is appropriate. PyTorch `.pt` files are not compatible with this application.

The model expects:

- Input size: `224 x 224 x 3`
- Pixel values: normalized to `0-1`
- Output classes: `glioma`, `meningioma`, `notumor`, `pituitary`

The repository may also contain `.keras` model files from training experiments. The Streamlit app specifically loads `models/brainscan_model.h5`.

### CNN Architecture

The model uses:

```text
Input (224 x 224 x 3)
Conv2D(32) -> MaxPooling2D
Conv2D(64) -> MaxPooling2D
Conv2D(128) -> MaxPooling2D
Dropout(0.25) -> Flatten
Dense(128) -> Dropout(0.5)
Dense(4, softmax)
```

It is trained with the Adam optimizer and sparse categorical cross-entropy loss.

## Run the Notebook

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/brainscan_eda.ipynb
```

The notebook includes dataset exploration, image loading and resizing, label encoding, train/validation/test splitting, CNN training, evaluation, confusion matrices, error analysis, and sample prediction.

When saving models from the notebook, use the local models directory:

```python
from pathlib import Path

MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODEL_DIR / "brainscan_model.h5")
```

## Run the Streamlit App

From the project root, with the virtual environment activated:

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

The interface accepts `.jpg`, `.jpeg`, `.png`, and `.bmp` images. Upload an MRI image and select **Analyze MRI** to view the predicted class and class probabilities.

## Prediction Process

The application:

1. Loads the uploaded image.
2. Converts it to RGB.
3. Resizes it to `224 x 224` pixels.
4. Converts pixel values from `0-255` to `0-1`.
5. Runs the image through the CNN.
6. Displays the predicted class and output probabilities.

## Troubleshooting

### Model file not found

Verify that this file exists:

```text
models/brainscan_model.h5
```

Run Streamlit from the project root, not from inside the `app/` directory.

### Dataset not found in the notebook

Verify that the notebook uses:

```python
DATA_DIR = Path("../data/brain_mri/Data")
```

and that all four class directories exist.

### CUDA or GPU warnings

Messages such as `Could not find cuda drivers` mean TensorFlow is using the CPU. They are not application errors. The project can run on CPU.

### Streamlit `use_container_width` warning

This is a Streamlit deprecation warning. It does not prevent the application from running. Newer Streamlit versions prefer `width="stretch"`.

## Git Notes

The following files are intentionally ignored and should not be committed to GitHub:

- MRI dataset files under `data/brain_mri/`
- Trained model files under `models/`
- Generated figures and metrics under `reports/`

Keep a separate copy of the dataset and model file when moving the project to another machine.

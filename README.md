# Microwave Spectral Regression Pipeline

This repository implements an end-to-end pipeline for predicting fruit quality attributes (Brix, TA, Firmness) from microwave spectral data. It supports flexible preprocessing modes, two model types (Random Forest, 1D CNN).

## 📂 Repository Structure

```
damage_detection_pipeline/
├── config/
│   └── config.yaml           # All hyperparameters & settings
├── data/
│   └── raw/                  # Place raw CSV here
├── src/
│   ├── main.py               # Single script for training & validation
│   ├── data_loader.py        # Loads raw CSV
│   ├── preprocessing.py      # Aggregation → feature extraction
│   ├── utils.py              # Metrics & model I/O
│   └── models/
│       ├── random_forest.py  # RF
│       └── oned_cnn.py       # PyTorch 1D-CNN model
└── requirements.txt          # Python dependencies
```

## 🚀 Setup

1. **Clone the repo**

  

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place Data**

   * Copy your spectral CSV (`Assignment_DataScientist_20250502.csv`) into `data/raw/`.

4. **Configure**

   * Edit `config/config.yaml` to adjust:

     * `training.targets` (which targets to predict)
     * `preprocessing.feature_mode` (all\_bands, band\_selection, pca\_only)
     * Model hyperparameters under `model.random_forest` or `model.oned_cnn`
     * Split ratio under `preprocessing.split`

## 🏃 Running the Pipeline

### Basic Usage

From the project root, run:

```bash
python src/main.py
```

This will:

1. Load and preprocess the raw data
2. Split into train/validation
3. Train the specified model on the train set
4. Evaluate on the validation set and print MAE/RMSE

### Command-Line Flags

* **Override targets** (predict only a subset):

  ```bash
  python src/main.py --targets Brix TA
  ```
* **Override model** (choose RF or CNN):

  ```bash
  python src/main.py --model random_forest
  ```


## 📈 Outputs

* **Console logs**: Training loss per epoch (for CNN), validation MAE/RMSE.
* **Model artifact**:

  * RF: saved as `models/random_forest.pkl`
  * CNN: saved as `models/oned_cnn.pt`







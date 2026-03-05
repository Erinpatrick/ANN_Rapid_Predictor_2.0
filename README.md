# ANN Rapid Predictor 2.0

Computational models are powerful tools that can enable the optimization of deep brain stimulation (DBS). To enhance the clinical practicality of these models, their computational expense and required technical expertise must be minimized. An important aspect of DBS models is the prediction of neural activation in response to electrical stimulation. Existing rapid predictors of activation simplify implementation and reduce prediction runtime, but at the expense of accuracy. We sought to address this issue by leveraging the speed and generalization abilities of artificial neural networks (ANNs) to create a novel predictor of neural fiber activation in response to DBS.

<img width="1916" height="1042" alt="image" src="https://github.com/user-attachments/assets/a006e257-c8ec-404d-bf50-9a2603bc3709" />
DRTT fibers activated by a bipolar electrode

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/BenNorbom/ANN_Rapid_Predictor_2.0.git
   cd ANN_Rapid_Predictor_2.0
   ```

2. Pull example tracks:
   To run the quick start examples, you'll need the tract files. You can pull just these files without downloading the entire large file history:
   ```bash
   git lfs pull -I "example_tracks/**"
   ```

3. Pull large model and electrode files:
   This repository uses Git LFS for large files. Run the following command to download all remaining necessary files:
   ```bash
   git lfs pull
   ```

   **Selective Download (Suggested):**
   If you only need specific electrode files (all of the electrode files are about 35 GB), you can pull them individually using the `-I` (include) flag. For example:
   ```bash
   git lfs pull -I "electrodes/directed/monopolar/bsc_directional_anisotropic_monopolar_01(a,b,c)2(-a,b,c)3.txt"
   ```

4. Create a Python 3.10 virtual environment and install dependencies:

   > **Important:** Python 3.10 is required. TensorFlow 2.15 does not support Python 3.11 or newer. If you have multiple Python versions installed, use the `py -3.10` launcher (Windows) or `python3.10` (Linux/Mac) to target the correct version.

   **Windows:**
   ```bash
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Linux/Mac:**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Run the ANN prediction

```bash
python run/dti_ann_LUT.py \
    electrodes/medtronic_3387/monopolar/3387_anisotropic_monopolar_01-23.txt \
    example_tracks/whole_brain.txt \
    models/ann_19_reg \
    run/results.json \
    ssd dti 38 50 30 reg
```

See [run/README.md](run/README.md) for detailed argument descriptions.

### 2. Visualize the results

```bash
python graphing/plot_tracts_fast.py \
    --tract example_tracks/L_DRTT_voxel.txt \
    --results run/results.json \
    --output output_viz/
```

To overlay the electric potential field, add `--electrode` and `--electrode_center`:

```bash
python graphing/plot_tracts_fast.py \
    --tract example_tracks/L_DRTT_voxel.txt \
    --results run/results.json \
    --output output_viz/ \
    --electrode "electrodes/medtronic_3387/monopolar/3387_anisotropic_monopolar_01-23.txt" \
    --electrode_center 167 223 147 \
    --field_subsample 4
```

See [graphing/README.md](graphing/README.md) for all visualization options.

## Project Structure

```
ANN_Rapid_Predictor_2.0/
├── electrodes/          # Electrode voltage field data (FEM exports)
├── example_tracks/      # Example tractography files
├── graphing/            # Visualization scripts
│   ├── plot_tracts_fast.py      # Fast 3D visualization (100k+ fibers)
│   └── plot_tracts.py           # Per-fiber rendering with interactive pulse-width plots
├── models/              # Pre-trained ANN models
├── run/                 # Prediction scripts
│   ├── dti_ann_LUT.py           # Main ANN prediction script
│   └── process_DTI.py           # Tract processing & spline interpolation
├── requirements.txt     # Python dependencies
└── README.md
```

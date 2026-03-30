# ANN Rapid Predictor 2.0

Computational models are powerful tools that can enable the optimization of deep brain stimulation (DBS). To enhance the clinical practicality of these models, their computational expense and required technical expertise must be minimized. An important aspect of DBS models is the prediction of neural activation in response to electrical stimulation. Existing rapid predictors of activation simplify implementation and reduce prediction runtime, but at the expense of accuracy. We sought to address this issue by leveraging the speed and generalization abilities of artificial neural networks (ANNs) to create a novel predictor of neural fiber activation in response to DBS.

<img width="1918" height="1047" alt="image" src="https://github.com/user-attachments/assets/6003b882-db0a-4762-a454-f960ef602b0c" />

DRTT, ML, and PTR fibers activated by a bipolar electrode

## Installation

1. Download the repository:
   Go to [https://github.com/Erinpatrick/ANN_Rapid_Predictor_2.0](https://github.com/Erinpatrick/ANN_Rapid_Predictor_2.0), click the green **Code** button, and select **Download ZIP**. Extract the ZIP to a location of your choice, then open a terminal in the extracted folder (the one containing `run/`, `graphing/`, etc.).

2. Download large data files from the GitHub Release:
   Electrode files (~1.2 GB each) and example tract files are hosted on the [Releases page](https://github.com/Erinpatrick/ANN_Rapid_Predictor_2.0/releases/tag/v1.0-data). Download the files you need and place them in the correct directories:

   | Release asset | Place in |
   |---|---|
   | `L_DRTT_voxel.txt`, `L_ML_voxel.txt`, `L_PTR_voxel.txt`, `L_combined_DRTT_ML_PTR_voxel.txt` | `example_tracks/` |
   | `3387_anisotropic_monopolar_01-23.txt` | `electrodes/medtronic_3387/monopolar/` |
   | `3387_anisotropic_bipolar_0-12+3.txt` | `electrodes/medtronic_3387/asymmetric_bipolar/` |
   | `3387_anisotropic_tripolar_-0+1-23.txt`, `3387_anisotropic_tripolar_-01-2+3.txt` | `electrodes/medtronic_3387/tripolar/` |
   | `3387_anisotropic_quadrupolar_-0+1-2+3.txt` | `electrodes/medtronic_3387/quadrupolar/` |

   You only need to download the electrode configuration(s) you plan to use.

3. Create a Python 3.10 virtual environment and install dependencies:

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

Each visualization script produces a **single interactive HTML file** (`activation.html`)
with a pulse width dropdown and a voltage threshold input that you can change
directly in the browser.

```bash
python graphing/plot_tracts_fast.py \
    --tract example_tracks/L_DRTT_voxel.txt \
    --results run/results.json \
    --output output_viz/
```

Then open `output_viz/activation.html` in your browser. Use the dropdown to
switch pulse widths and the voltage input to adjust the activation threshold.

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

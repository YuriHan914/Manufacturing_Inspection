*[한국어](quickstart.ko.md)*

## 0. Installation

- Install the required packages.

```bash
pip install -r requirements.txt

cd third_party/OpenXAI
pip install -e .
```

- `OpenXAI` installation is required for XAI features.
- Download an LLM model.
  - Save it in the `model` folder.
  - The model folder name must match the expected name exactly.
  - [Gemma](https://huggingface.co/google/gemma-4-E4B-it)
  - Other LLM models can also be used.

## 1. Database connection

### Way1. Sample dataset
<p align="center"><img src="../assets/Query_Period.png" alt="Query period" width="500"/></p>

- If no DB is connected, the app runs with the default sample data.
- Run the app, and the screen below appears. Filter the data period you want to analyze, then click the `Load Data` button to get started.


### Way2. Database - Supabase
- Create the `.streamlit/secrets.toml` file.
- Add the DB connection settings as shown below. This app currently supports only Supabase.
- On the `Setting` page, click `DB Settings` to change the DB connection settings.

```toml
[connections.supabase]
SUPABASE_URL = "http://000.0.0.0:12345"
SUPABASE_KEY = "abcdefghij"
```



## 2. Dashboard page
<p align="center"><img src="../assets/Dashboard_Home.png" alt="Dashboard home" width="500"/></p>

- This page provides a summary view of the `Summary`, `Detail`, `Fine-tuning`, `Setting`, and `Log` pages.
- At the top of the page, set the data period and click `Load Data`. For sample data, keep the default date values.
- If you do not click `Load Data`, inference and other features will not run.

## 3. Summary page
<p align="center"><img src="../assets/Summary.png" alt="Summary" width="500"/></p>
- This page shows the data analysis summary.
<p align="center"><img src="../assets/Report.png" alt="Report" width="500"/></p>
- Click `Download Report` to save the current summary as a PDF file.

## 4. Analysis page
<p align="center"><img src="../assets/Analysis_classification.png" alt="Analysis - Classification" width="500"/></p>

- This page provides detailed AI inference results for images and related analysis features. Analysis starts only after you select images in `Select images`. Date and class filters are available.
- 2D Image Anomaly Detection (`Classification`,`Anomaly Detection`)
  - (Available features when a 3D Rubbish Bin is selected in the Dashboard table.)
  - Use the `Method` selector to switch between `Classification` and `Anomaly Detection`.
    - In `Anomaly Detection` mode, click `Run Anomaly Detection` in the sidebar to extract features for the selected images and score them against the memory bank with a PatchCore-based scorer.
      <p align="center"><img src="../assets/Analysis_AD.png" alt="Analysis - Anomaly Detection" width="500"/></p>
  - The `Result` tab shows the predicted class and image in `Classification` mode, or each image's anomaly score and Normal/Anomaly prediction (based on a calibrated threshold) in `Anomaly Detection` mode.
  - The `3D Visualization` tab compresses the selected images' features into lower dimensions for visualization (PCA, t-SNE, or UMAP in `Classification` mode; patch-averaged, PCA-compressed features in `Anomaly Detection` mode). It works only when at least 3 images are selected.
    <p align="center"><img src="../assets/Analysis_3d_visual.png" alt="3D Visualization" width="500"/></p>
  - The `XAI` tab uses OpenXAI to help interpret the model's predictions in `Classification` mode. In `Anomaly Detection` mode, this tab is replaced by an `Anomaly Heatmap`, which overlays the PatchCore anomaly map on each image instead.
    <p align="center"><img src="../assets/Analysis_XAI.png" alt="XAI / Anomaly Heatmap" width="500"/></p>

- 3D point cloud anomaly detection (`3D-KD`)
  <p align="center"><img src="../assets/3D-AD.png" alt="Analysis - 3D Anomaly Detection" width="500"/></p>

  - When the `3D Rubbish bin` table is selected on the Dashboard page, the Analysis page switches to 3D point cloud mode.
  - `3D-KD` (teacher–student knowledge distillation) is the only available `Method`.
  - Choose point clouds (`.ply`) in `Select point clouds`, then click `Run`. Results already saved in `outputs/3D-AD` are loaded instead of re-inferred; the rest are inferred and saved there.
  - The `Result` tab shows each point cloud's `Prediction` (Normal/Anomaly), `Max score`, `Image threshold`, and `Anomalous points` (%), along with an interactive 3D view colored by per-point anomaly score.
  - The `3D Visualization` and `XAI` tabs are not available for point cloud data.

## 5. Setting
<p align="center"><img src="../assets/Settings.png" alt="Settings" width="500"/></p>

- `DB Settings`: input values for the DB connection keys.
- `LLM Runtime`: options for the LLM model used in the left sidebar and in Summary analysis.

## 6. Log
<p align="center"><img src="../assets/Log.png" alt="Log" width="500"/></p>
- You can check the logs generated while using the app.
- Logs are automatically saved by date in the `log` folder.

## 7. Sidebar: LLM
<p align="center"><img src="../assets/LLM.png" alt="LLM sidebar" width="150"/></p>

- You can interact with the LLM through the command box in the sidebar.



#

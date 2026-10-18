# 📌 Changelog

All notable changes to this project will be documented in this file.

---

## [v0.8.0] - 2026-10-18

### Added

* Added 2D-3D multimodal anomaly detection with `2D-3D AD` (Utonia point features + ResNet-101 image features fused per point, scored against a PatchCore memory bank) — a new `2D-3D Rubbish bin` table (`data/2D-3D`, one record per 12-view object, dated 2026-01-01) can be selected on the Dashboard page
* Added 2D-3D AD to the Analysis and Summary pages: each view is judged abnormal when its view score exceeds that view's threshold in `model/2D-3D/object_level_thresholds.csv`, and an object is predicted Anomaly by hard voting (at least one abnormal view); shows prediction, abnormal views, max view score, and anomalous point ratio, plus a 3D view of the merged 12-view object colored by anomaly score, per-view scores, and per-view 2D heatmaps projected onto the camera images; results are saved under `outputs/2D-3D` and reused instead of being run again
* Inference runs `scripts/2D-3D/infer_app.py` in the `utonia` conda env (override with `FUSION_2D3D_PYTHON`); the Utonia checkpoint `model/2D-3D/utonia.pth` is downloaded from HuggingFace (`Pointcept/Utonia`) when missing, and the anomalous-point threshold / color range come from `model/2D-3D/calibration.json`
* Fine-tuning is disabled for the 2D-3D table
* The Analysis Result tab shows the 2D-3D object point cloud after a fixed display transform (`FUSION_DISPLAY_PLY_T3 @ FUSION_DISPLAY_PLY_T2 @ FUSION_DISPLAY_PLY_T1` in `scripts/fusion_inference.py`) and the heatmap images rotated 180°, with view score / threshold as the image caption; scoring still runs on the original ply and images
* The 2D heatmap on the Analysis Result tab is a Plotly image (`scrollZoom`, `dragmode="pan"`): mouse wheel zooms, drag pans, double-click resets; it opens on view 12
* The Summary page shows the selected object's 12 view heatmaps (rotated 180°) as small images in a 6 x 2 grid, with view score / threshold and abnormal views marked, instead of the 3D view

### Changed

* Renamed the sample selector on the Analysis page to `Select Sample` for every data source (previously `Select images`, `Select point clouds`, or `Select objects` depending on the selected table)

---

## [v0.7.0] - 2026-10-01

### Added

* Added 3D point cloud anomaly detection with `3D-KD` (teacher–student knowledge distillation) — a new `3D Rubbish bin` table can be selected on the Dashboard page, and results are saved under `outputs/3D-AD`; files that already have saved results are loaded instead of being run again
* Added 3D-KD inference to the Summary page: per-file prediction, max score, image threshold, and anomalous point ratio, plus a 3D view colored by anomaly score
* Added a 3D point cloud mode to the Analysis page: select `.ply` files and click `Run` to see each file's results and a 3D view colored by per-point anomaly score
* Added a 3D point cloud view to the Fine-tuning page that lists saved 3D-KD predictions (fine-tuning stays disabled until a 3D training dataset is available)

---

## [v0.6.0] - 2026-09-22

### Changed

* Converted the classification model (JAX-trained MobileViT-S) to ONNX and TensorRT engine formats for inference — the app now runs on a TensorRT engine (GPU) with an ONNX Runtime (CPU) fallback instead of the raw JAX/PyTorch model

---

## [v0.5.0] - 2026-08-04

### Changed

* Changed the default coreset projection method from JL (random projection) to PCA on the Anomaly Detection memory bank pipeline — projection is now fit via IncrementalPCA over disk-backed patch bank chunks, trading speed for a variance-preserving, deterministic projection
* Added `--proj-type` (`JL`/`PCA`) and `--pca-batch-size` options to `coreset_sampling.py` and `run_feature_pipeline.sh` to switch between projection methods

---

## [v0.4.0] - 2026-07-17

### Added

* Added Anomaly Detection feature (PatchCore-based memory bank scoring) on the Analysis page


---

## [v0.3.0] - 2026-04-14

### Added

* Added XAI (Explainable AI) feature integration
* Added event logging for dashboard interactions
* Added data period selector on the Dashboard Home page — users now select a date range from the database before running inference

### Changed

* Revised Dashboard Home workflow: data period selection from DB is now required prior to inference execution

---

## [v0.2.0] - 2026-04-13

### Added

* Added Active Learning-based sampling strategies
* Added data visualization features (PCA, t-SNE, UMAP)

### Fixed

* Fixed a dimension mismatch issue in PCA visualization
* Resolved several image loading errors

---

## [v0.1.0] - 2026-04-10

### Added

* Set up the initial project structure
* Implemented the image classification model
* Added training and inference scripts

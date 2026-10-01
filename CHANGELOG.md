# 📌 Changelog

All notable changes to this project will be documented in this file.

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

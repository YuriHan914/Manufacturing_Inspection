# 🏭 Manufacturing Inspection Application

*[English](README.en.md)*

반도체 검사를 위한 Streamlit 기반 애플리케이션으로, 이미지 분류, 추론, 액티브 러닝 기반 샘플 선택, 모델 파인튜닝 기능을 제공합니다.

---
## 📽 데모 영상

<p align="center">
  <a href="https://youtu.be/ZHjwQyarjOs">
    <img src="assets/Analysis_AD.png" width="700"/>
  </a>
</p>

<p align="center">
  <sub>Manufacturing Inspection Application (이미지를 클릭하세요)</sub>
</p>

---


## 🚀 주요 기능

* Streamlit 기반 검사 애플리케이션
* MobileViT 이미지 분류 학습 및 추론
* PatchCore 기반 이상 탐지
* 3D차원 포인트 클라우드 이상 탐지(3D Rubbish Bin)
* Active Learning 기반 샘플링
* 선택한 이미지를 이용한 인터랙티브 파인튜닝
* Gemma 기반 어시스턴트 응답

---

## 🛠️ 설치 방법

```bash
git clone https://github.com/GritGlass/Manufacturing_Inspection.git
cd Manufacturing_Inspection

git lfs install
git lfs pull


pip install -r requirements.txt
```

필수 설정:

* 학습/런타임 설정은 더 이상 레거시 로컬 `data` 폴더에서 로드되지 않습니다
* `.streamlit/secrets.toml`
* 선택 사항: `model/google__gemma-4-E2B-it/`에 로컬 모델 파일

---

## ▶️ 사용 방법

### 대시보드

```bash
streamlit run Dashboard.py
```

---

## 📚 문서

- [빠른 시작 가이드](docs/quickstart.md)


---

## 📂 프로젝트 구조

```text
.
├── .streamlit/
├── assets/
├── data/
├── log/
├── model/
├── output/
├── pages/
├── scripts/
│   ├── detail_finetune_mcp.py
│   └── local_gemma_model.py
├── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

---

## 📌 참고 사항

### 페이지 구성

* Dashboard : 데이터 분포, 최근 실행 내역, 최신 로그, 현재 모델 설정
* Summary : 데이터 분포, 정상/불량 현황, 월간/주간/일간 그래프, LLM 코멘트
* Detail : 분류 모델 추론 결과
* Fine tuning : 모델 선택, 파인튜닝, 액티브 러닝 샘플링, 데이터 라벨링
* Setting : 데이터베이스 설정, LLM 설정
* Log : 로그 이력

### 버전 업데이트

* [변경 이력](CHANGELOG.md)




#### 참고 자료
- 데이터 출처 : [Semiconductor](https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset)
- 모델 : [Mobilevit_small](https://huggingface.co/apple/mobilevit-small)

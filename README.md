
## 개요

Model_Benchmark는 CNN, 대형 언어 모델(LLM), 멀티모달 모델, Stable Diffusion 등 다양한 머신러닝 모델을 테스트하고 벤치마킹하기 위한 종합 도구입니다.

## 목차

- [기능](#기능)
- [필수 조건](#필수-조건)
- [설치 방법](#설치-방법)
- [설정 방법](#설정-방법)
- [테스트 실행](#테스트-실행)
- [결과 해석](#결과-해석)
- [추가 기능](#추가-기능)
- [문제 해결](#문제-해결)
- [라이선스](#라이선스)


## 기능

- **모델 구조 테스트**  
  모델 아키텍처, 파라미터 개수, 메모리 크기 등을 분석합니다.

- **알고리즘 구조 테스트**  
  학습 루프, 옵티마이저, 손실 함수의 동작을 검증합니다.

- **양자화**  
  FP16 및 INT8로 모델을 변환한 후 크기와 성능을 비교합니다.

- **동적 입력 테스트**  
  다양한 입력 크기에 대한 모델의 안정성을 평가합니다.

- **시각화**  
  모델 아키텍처를 그래프로 시각화하여 가독성을 높입니다.

- **벤치마킹**  
  추론 속도를 측정하고 성능 최적화를 위한 데이터를 제공합니다.

- **상세 로깅**  
  테스트 및 디버깅을 위한 세부 로그를 기록합니다.


## 필수 조건

- **운영체제**  
  Windows 10 이상

- **Python**  
  3.8 이상

- **필요 라이브러리**  
  - PyTorch
  - torchinfo
  - torchviz
  - transformers (LLM 및 멀티모달 모델용)
  - Graphviz

## 설치 방법

### 1. Graphviz 설치

Graphviz는 모델 아키텍처를 시각화하는 데 필수적입니다.

1. [Graphviz 다운로드](https://graphviz.org/download/) 페이지에서 설치 파일을 받으세요.
2. Windows의 경우, `windows_10_cmake_Release_graphviz-install-12.2.1-win64.exe` 파일을 다운로드하세요.
3. **중요**: 설치 중에 환경 변수(`PATH`) 추가 옵션을 반드시 체크하세요.

### 2. 가상환경 설정

의존성 관리를 위해 가상환경을 사용하는 것이 권장됩니다.

```bash
python -m venv venv
```

가상환경 활성화:

- **Windows**:  
  ```bash
  venv\Scripts\activate
  ```

- **MacOS 또는 Linux**:  
  ```bash
  source venv/bin/activate
  ```

### 3. 필수 Python 패키지 설치

필요한 패키지를 설치합니다:

```bash
pip install torch torchinfo torchviz transformers
```


## 설정 방법

### 1. 모델 클래스 정의 또는 가져오기

테스트를 실행하기 전에 `Model_Benchmark.py` 스크립트에 모델 클래스가 정의되어 있거나 가져와야 합니다. 예를 들어:

- **SimpleCNN**: 간단한 합성곱 신경망(CNN) 예제.
- **GPT2LMHeadModel**: Hugging Face에서 제공하는 GPT-2 사전 학습 모델.
- **CLIPModel**: 텍스트와 이미지를 처리하는 멀티모달 모델.
- **StableDiffusionModel**: 이미지 생성 모델(Stable Diffusion).

### 2. 모델 정의 설정

`Model_Benchmark.py` 파일에서 `model_definitions` 리스트를 찾아 테스트할 모델과 유형을 지정하세요.

```python
model_definitions = [
    {"model_class": SimpleCNN, "model_type": "cnn"},
    {"model_class": GPT2LMHeadModel, "model_type": "llm"},
    {"model_class": CLIPModel, "model_type": "multimodal"},
    {"model_class": StableDiffusionModel, "model_type": "stable_diffusion"}
]
```

### 3. 테스트 구성 설정

다양한 입력 채널, 클래스 수, 이미지 크기, 텍스트 입력 등을 정의합니다.

```python
test_configs = [
    {"input_channels": 3, "num_classes": 10, "image_size": (256, 256), "text": "테스트 입력"},
    {"input_channels": 4, "num_classes": 5, "image_size": (128, 128), "text": "또 다른 테스트 입력"}
]
```


## 테스트 실행

아래 명령어를 실행하여 테스트를 시작합니다:

```bash
python Model_Benchmark.py
```

스크립트는 각 모델과 구성에 대해 테스트를 수행하며 결과를 출력합니다.


## 결과 해석

스크립트 실행 후 결과는 콘솔과 `model_test.log` 파일에 기록됩니다. 주요 지표는 다음과 같습니다:

### 1. 전체 파라미터 (Total Parameters)  
모델 내 모든 가중치와 편향의 총 개수.  
**예**: `31,043,521`

### 2. 학습 가능한 파라미터 (Trainable Parameters)  
학습 중 업데이트되는 파라미터의 개수(`requires_grad=True`).  
**예**: `31,043,521`

### 3. 예상 모델 크기 (Estimated Model Size)  
FP32로 저장된 모델의 메모리 크기.  
**예**: `118.47 MB`

### 4. 실제 저장 모델 크기 (Actual Saved Model Size)  
디스크에 저장된 모델 파일의 크기.  
**예**: `118.50 MB`

### 5. 연산 복잡도 (Total Mult-Adds, G)  
순전파에 필요한 곱셈-덧셈 연산의 총 개수(GFLOPs).  
**예**: `54.65 G`

### 6. 입력 크기 (Input Size, MB)  
입력 데이터가 차지하는 메모리 크기.  
**예**: `0.79 MB`

### 7. 순전파/역전파 메모리 크기 (Forward/Backward Pass Size, MB)  
계산 중간 값이 차지하는 메모리 크기.  
**예**: `575.14 MB`

### 8. 파라미터 크기 (Params Size, MB)  
모델 파라미터의 메모리 크기.  
**예**: `124.17 MB`

### 9. 총 메모리 크기 (Estimated Total Size, MB)  
입력, 중간 계산, 파라미터를 포함한 총 메모리 요구량.  
**예**: `700.10 MB`

### 10. 모델 양자화  
- **FP16 (Half Precision)**:  
  FP16으로 변환 시 메모리 사용량 절반 감소.  
  **예**: `59.23 MB`

- **INT8 (Quantized)**:  
  INT8로 양자화 시 크기 감소.  
  **예**: `59.28 MB`

### 11. 모델 테스트 종류  
- **기본 모델 테스트**:  
  표준 구성(예: FP32, 3채널 입력)으로 모델을 테스트.  
  **예**: 입력 `(256, 256)` -> 출력 `[1, 1, 256, 256]`.

- **양자화 모델 테스트**:  
  FP16, INT8 모델에서 메모리 사용량과 성능 개선 확인.

- **동적 입력 크기 테스트**:  
  다양한 입력 크기에서 모델의 동작 확인.  
  **예**: `(128, 128)`, `(256, 256)`, `(512, 512)`.

### 12. 알고리즘 구조 테스트  
- **학습 루프 테스트**:  
  간단한 학습 루프를 실행해 학습 가능 여부를 확인.  
  **예**: 손실 값 로그 기록.

- **옵티마이저 및 손실 함수 테스트**:  
  옵티마이저와 손실 함수의 동작 검증.  
  **예**: 손실 감소 확인.

### 13. 모델 시각화  
모델 아키텍처를 시각화한 그래프 생성.  
**출력 파일**: `model_graph_<ModelClassName>.png`.

### 14. 벤치마킹  
추론 속도를 측정.  
**예**: `평균 추론 시간: 0.0042초 (cuda 사용)`

### 15. 메모리 사용량  
GPU 메모리 할당 및 예약 상태 확인.  
**예**:  
- `Allocated memory: 512.00 MB`
- `Reserved memory: 1024.00 MB`


## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

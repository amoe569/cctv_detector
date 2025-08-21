# CCTV AI Detection Server

Python 기반의 CCTV 영상 분석 및 AI 객체 탐지 서버입니다.

## 🚀 기술 스택

- **Python 3.9+**
- **OpenCV** for video processing
- **YOLOv8n** for object detection
- **Flask** for web framework
- **Ultralytics** for YOLO model
- **requests** for HTTP communication

## 📦 설치 및 실행

### 필수 요구사항
- Python 3.9+
- pip

### 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### YOLO 모델 다운로드
```bash
# YOLOv8n 모델 자동 다운로드 (첫 실행 시)
# 또는 수동으로 다운로드:
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 서버 실행
```bash
python app.py
```

## 🌐 접속

- **메인 페이지**: http://localhost:5001
- **MJPEG 스트림**: http://localhost:5001/stream/{camera_id}
- **테스트 이벤트**: http://localhost:5001 (테스트 패널)

## 📁 프로젝트 구조

```
detector/
├── app.py              # 메인 Flask 애플리케이션
├── requirements.txt    # Python 의존성
├── yolov8n.pt         # YOLO 모델 파일 (별도 다운로드)
├── Dockerfile         # Docker 컨테이너 설정
└── README.md          # 프로젝트 문서
```

## 🔧 환경 설정

`.env` 파일을 생성하여 다음 환경변수를 설정하세요:

```env
# Spring Boot 서버 설정
SPRING_BOOT_URL=http://localhost:8080

# Flask 서버 설정
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=True

# YOLO 모델 설정
YOLO_MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
```

## 📹 주요 기능

### 실시간 객체 탐지
- **RTSP 스트림** 수신 및 처리
- **YOLOv8n** 모델을 사용한 차량/사람 탐지
- **10대 이상 차량** 감지 시 "traffic_heavy" 이벤트 발생

### 이벤트 전송
- **Spring Boot 서버**로 이벤트 전송
- **HTTP POST** `/api/events/traffic` 엔드포인트
- **JSON 형식**으로 이벤트 데이터 전송

### 테스트 기능
- **웹 UI**에서 테스트 이벤트 발령
- **카메라 선택** 후 이벤트 전송
- **실시간 결과** 확인

## 🔍 객체 탐지 클래스

YOLOv8n 모델이 탐지하는 주요 객체:
- **car** (자동차)
- **truck** (트럭)
- **bus** (버스)
- **motorcycle** (오토바이)
- **person** (사람)

## 📊 성능 최적화

- **프레임 크기**: 640x640으로 리사이즈
- **탐지 주기**: 실시간 프레임별 처리
- **메모리 관리**: 불필요한 프레임 자동 해제
- **에러 처리**: 연결 실패 시 자동 재시도

## 🚨 이벤트 발생 조건

- **차량 수**: 10대 이상 감지 시
- **이벤트 타입**: "traffic_heavy"
- **메타데이터**: 차량 수, 메시지, 타임스탬프
- **카메라 상태**: WARNING 상태가 아닌 경우에만 전송

## 🔗 연동 시스템

- **Spring Boot Control Center**: 이벤트 수신 및 처리
- **SSE (Server-Sent Events)**: 실시간 이벤트 스트리밍
- **카메라 상태 관리**: 상태 변경 및 모니터링

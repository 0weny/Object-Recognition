## YOLOv8 기반 실시간 객체 탐지 및 영상 분석 시스템
본 프로젝트는 두 가지 주요 영상 입력 소스(실시간 웹캠과 mp4파일)를 대상으로 YOLOv8과 MediaPipe를 활용해 객체 탐지 및 얼굴 랜드마크 시각화를 수행합니다.
분석 결과는 실시간 시각화와 JSON 통계 데이터, 결과영상(mp4)을 도출하며, 다양한 인공지능/영상처리 프로젝트의 기반 시스템에 활용가능합니다.

<br>

## 1. 실시간 웹캠 영상에서 얼굴 및 사물 객체 인식 수행
 `webcam_face_detect_yolov8.py`

 
### 🔧 주요 기능
1. 실시간 웹캠 스트림 기반

2. YOLOv8으로 객체 탐지 (person, car, cell phone 등)

3. MediaPipe로 얼굴의 랜드마크 시각화 (눈, 입, 얼굴 윤곽, 홍채)
4. 실시간 시각화 (cv2.imshow) 출력


![Demo GIF](https://github.com/0weny/Object-Recognition/blob/main/static/facedetection.gif?raw=true)



<br>

## 2. mp4 영상에서의 객체 인식 및 분석
`mp4_video_object_detect.py`

### 🔧 주요 기능 
1. mp4 영상을 입력으로 받아 YOLOv8 모델로 객체 탐지 수행

2. MediaPipe로 얼굴 랜드마크(눈,입, 얼굴 윤곽, 홍채 등) 분석

3. 프레임마다 객체 및 얼굴을 시각화

4. 객체 등장 빈도를 JSON 통계 파일로 저장

5. 오디오/비디오 분리 후 ffmpeg로 병합하여 재합성

6. 임시 파일(audio.aac, temp_video.mp4) 자동 삭제로 저장 공간 절약

[출력 예시]
output_final_20250726_103021.mp4: 결과 영상 (객체+얼굴 시각화)

object_summary_20250726_103021.json: 객체별 등장 빈도 통계

<br>

## 환경 구축 및 가상환경 설정
환경 - `Python 3.11`

필수 라이브러리 설치

<pre><code>
  ultralytics
  opencv
  mediapipe 
  numpy 
  ffmpeg
</code></pre>



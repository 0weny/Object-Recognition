import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import subprocess
import json
from collections import Counter
import sys

# 입력 인자 받기
input_path = sys.argv[1]  # 예: uploads/input_20250726_203000.mp4
timestamp = sys.argv[2]   # 예: 20250726_203000

# 경로 설정
FINAL_VIDEO_PATH = f'static/outputs/output_final_{timestamp}.mp4'
JSON_PATH = f'static/outputs/object_summary_{timestamp}.json'
AUDIO_PATH = f'temp_audio_{timestamp}.aac'  # 임시 오디오 파일
TEMP_VIDEO_PATH = f'temp_video_{timestamp}.mp4'  # 임시 영상 파일

# YOLO 모델
model = YOLO('yolov8n.pt')

# MediaPipe 얼굴 탐지
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 객체 카운트
object_counter = Counter()

# 영상 열기
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ 영상 열기 실패")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, fps, (w, h))

print("🎞 영상 분석 시작...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"▶ 프레임 {frame_idx} 분석 중...")

    results = model.predict(frame, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = model.model.names[cls]

        if conf > 0.5:
            object_counter[label] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# JSON 저장
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(object_counter.most_common(), f, indent=2, ensure_ascii=False)
print(f"📊 객체 통계 저장: {JSON_PATH}")

# 오디오 추출
subprocess.run([
    "ffmpeg", "-y",
    "-i", input_path,
    "-vn", "-acodec", "copy", AUDIO_PATH
])

# 영상 + 오디오 병합
subprocess.run([
    "ffmpeg", "-y",
    "-i", TEMP_VIDEO_PATH,
    "-i", AUDIO_PATH,
    "-c:v", "copy", "-c:a", "aac", "-shortest",
    FINAL_VIDEO_PATH
])

# 임시 파일 삭제
os.remove(TEMP_VIDEO_PATH)
os.remove(AUDIO_PATH)

print(f"✅ 최종 영상 저장 완료: {FINAL_VIDEO_PATH}")

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO



# 1. YOLOv8 로드 (사람, 차량, 책상 등 COCO 객체 인식용)
model = YOLO('yolov8n.pt')  # 'yolov8s.pt'나 커스텀 모델도 가능

# 2. MediaPipe 얼굴 랜드마크 설정 (눈, 코, 입, 윤곽 탐지용)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. 랜드마크 연결선 (미디어파이프에서 제공)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 4. 웹캠 시작
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("실시간 YOLO + 얼굴 랜드마크 탐지 시작 (종료: q)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    ### (1) YOLOv8 객체 탐지
    yolo_results = model.predict(frame, verbose=False)[0]

    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = model.model.names[cls]

        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    ### (2) MediaPipe 얼굴 랜드마크 탐지
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 랜드마크 시각화 (눈, 코, 입, 윤곽 등 포함)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style()
            )

    # (3) 출력
    cv2.imshow("YOLOv8 + FaceMesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()

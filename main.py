import cv2
from ultralytics import YOLO
import webbrowser

human_boxes = []
balls_boxes = []
WIKI_URL1 = "https://www.wubearcats.com/sports/wbkb/2025-26/coach/Ashley_Corral"
WIKI_URL2 = "https://www.adidas.com.br/bolas-basquete"

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for (x1, y1, x2, y2) in human_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                print("Humano clicado! Abrindo a informações da ashley...")
                webbrowser.open(WIKI_URL1)
                break
        for (x1, y1, x2, y2) in balls_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                print("Bola clicado! Abrindo Propaganda")
                webbrowser.open(WIKI_URL2)
                break
                
VIDEO_PATH = "How to Shoot a Basketball-1920x1080-avc1-mp4a.mp4"
MODEL_PATH = "yolov10n.pt"
CONF_THRESHOLD = 0.3

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Erro ao abrir o vídeo")

cv2.namedWindow("YOLO")
cv2.setMouseCallback("YOLO", on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    human_boxes = []
    balls_boxes = []

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False
    )

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if model.names[cls_id] == "person":
                human_boxes.append((x1, y1, x2, y2))

            if model.names[cls_id] == "sports ball":
                balls_boxes.append((x1, y1, x2, y2))


            class_name = model.names[cls_id]
            label = f"{class_name} #{track_id} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

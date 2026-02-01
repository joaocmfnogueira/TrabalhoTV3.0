import cv2
from ultralytics import YOLO
from flask import Flask, Response, jsonify
import time
import traceback

VIDEO_PATH = "How to Shoot a Basketball-1920x1080-avc1-mp4a.mp4"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.9

WIKI_PERSON = "https://www.wubearcats.com/sports/wbkb/2025-26/coach/Ashley_Corral"
WIKI_BALL = "https://www.adidas.com.br/bolas-basquete"

last_detections = []
app = Flask(__name__)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

def make_tracker():
    # OpenCV >= 4.5 (mais comum)
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    # OpenCV legacy (algumas builds)
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError("Nenhum tracker compatível encontrado no OpenCV")


def gen_frames():
    global last_detections, cap

    FRAME_SKIP = 10
    SCALE_FACTOR = 0.2
    frame_count = 0

    trackers = []
    tracked_urls = []

    if not cap.isOpened():
        print("ERROR: VideoCapture não abriu o arquivo.", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # log e reinicia o vídeo (mantém stream vivo)
            print("cap.read() retornou False — reiniciando vídeo -> set POS 0")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # pequena espera para evitar loop tight se houver problema
            time.sleep(0.1)
            continue

        frame_count += 1

        try:
            # Resize proporcional
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)),
                interpolation=cv2.INTER_AREA
            )
        except Exception as e:
            print("Erro ao redimensionar frame:", e)
            traceback.print_exc()
            # ainda assim tente continuar
            pass

        detections = []

        # YOLO a cada FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            trackers.clear()
            tracked_urls.clear()

            try:
                results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            except Exception as e:
                print("Erro durante inferência do modelo:", e)
                traceback.print_exc()
                # não quebre o stream — espere e continue
                time.sleep(0.1)
                continue

            try:
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]

                        if class_name == "person":
                            url = WIKI_PERSON
                        elif class_name == "sports ball":
                            url = WIKI_BALL
                        else:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        try:
                            tracker = make_tracker()
                            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                            trackers.append(tracker)
                            tracked_urls.append(url)
                        except Exception as e:
                            print("Falha ao criar/init tracker:", e)
                            traceback.print_exc()
            except Exception as e:
                print("Erro ao processar results:", e)
                traceback.print_exc()

        # Atualiza trackers para frames intermediários
        for tracker, url in zip(trackers, tracked_urls):
            try:
                success, box = tracker.update(frame)
            except Exception as e:
                # se o tracker falhar, pule
                print("Tracker.update() falhou:", e)
                traceback.print_exc()
                continue

            if not success:
                # opcional: você pode tentar remover o tracker aqui
                continue

            x, y, w, h = map(int, box)
            detections.append({
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "url": url
            })

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        last_detections = detections

        try:
            _, buffer = cv2.imencode(".jpg", frame)
            jpg_bytes = buffer.tobytes()
        except Exception as e:
            print("Erro em imencode:", e)
            traceback.print_exc()
            # envie frame preto para não encerrar stream
            blank = 255 * np.ones((100, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode(".jpg", blank)
            jpg_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg_bytes +
            b"\r\n"
        )

@app.route("/video")
def video():
    return Response(gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    return jsonify(last_detections)

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Interactive Stream</title>
    <style>
        canvas { position:absolute; left:0; top:0; }
        img { position:absolute; left:0; top:0; }
    </style>
</head>
<body>

<img id="video" src="/video">
<canvas id="overlay"></canvas>

<script>
const img = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
let boxes = [];

function resize() {
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
}
window.onresize = resize;

async function fetchBoxes() {
    const res = await fetch("/detections");
    boxes = await res.json();
}

function drawBoxes() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;

    for (const b of boxes) {
        ctx.strokeRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
    }
}

canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    for (const b of boxes) {
        if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) {
            window.open(b.url, "_blank");
            break;
        }
    }
});

setInterval(async () => {
    await fetchBoxes();
    drawBoxes();
}, 200);

img.onload = resize;
</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

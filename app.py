import cv2
from ultralytics import YOLO
from flask import Flask, Response, jsonify

VIDEO_PATH = "How to Shoot a Basketball-1920x1080-avc1-mp4a.mp4"
MODEL_PATH = "yolov10n.pt"
CONF_THRESHOLD = 0.3

WIKI_PERSON = "https://www.wubearcats.com/sports/wbkb/2025-26/coach/Ashley_Corral"
WIKI_BALL = "https://www.adidas.com.br/bolas-basquete"

app = Flask(__name__)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

last_detections = []

def gen_frames():
    global last_detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []

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
                class_name = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name == "person":
                    url = WIKI_PERSON
                elif class_name == "sports ball":
                    url = WIKI_BALL
                else:
                    continue

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "url": url
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        last_detections = detections

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
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
    app.run(host="0.0.0.0", port=5000)

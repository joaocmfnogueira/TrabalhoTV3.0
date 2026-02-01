import cv2
import time
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

INTERACTIVE_OBJECTS = {
    "person_profile": {
        "yolo_classes": ["person"],
        "label": "Pessoa",
        "link": "https://www.wubearcats.com/sports/wbkb/2025-26/coach/Ashley_Corral"
    },
    "basketball_product": {
        "yolo_classes": ["sports ball"],
        "label": "Bola de Basquete",
        "link": "https://www.adidas.com.br/bolas-basquete"
    }
}

# ================= CONFIG =================
VIDEO_PATH = "How to Shoot a Basketball-1920x1080-avc1-mp4a.mp4"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.4
TARGET_FPS = 30

YOLO_INTERVAL = 5      # YOLO roda a cada N frames
BOX_TIMEOUT = 0.6      # segundos que a bbox permanece viva

WIKI_PERSON = "https://www.wubearcats.com/sports/wbkb/2025-26/coach/Ashley_Corral"
WIKI_BALL = "https://www.adidas.com.br/bolas-basquete"
# ==========================================

app = Flask(__name__)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Cache temporal
ACTIVE_BOXES = {}
last_detections = []

def limpar_boxes(boxes, t):
    return {
        k: v for k, v in boxes.items()
        if t - v["last_seen"] <= BOX_TIMEOUT
    }

def gen_frames(ai_enabled: bool):
    global ACTIVE_BOXES

    frame_id = 0
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    SCALE = 0.6

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_id += 1
        now = frame_id / fps_video

        # resize
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w*SCALE), int(h*SCALE)))

        # ========= IA =========
        if ai_enabled and frame_id % YOLO_INTERVAL == 0:
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = model.names[int(box.cls[0])]

                    # verifica se essa bbox é compatível com algum objeto interativo
                    matched_object = None
                    for obj in INTERACTIVE_OBJECTS.values():
                        if cls in obj["yolo_classes"]:
                            matched_object = obj
                            break

                    if not matched_object:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ACTIVE_BOXES[cls] = {
                        "box": (x1, y1, x2, y2),
                        "last_seen": now,
                        "meta": matched_object
                    }

        ACTIVE_BOXES = limpar_boxes(ACTIVE_BOXES, now)

        # ========= DESENHO =========
        if ai_enabled:
            for data in ACTIVE_BOXES.values():
                x1, y1, x2, y2 = data["box"]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # ========= STREAM =========
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(max(0, (1/TARGET_FPS) - (time.time()-start)))

@app.route("/video")
def video():
    ai = request.args.get("ai", "0") == "1"
    return Response(
        gen_frames(ai),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detections")
def detections():
    return jsonify(last_detections)

@app.route("/metadata")
def metadata():
    data = []

    for info in ACTIVE_BOXES.values():
        x1, y1, x2, y2 = info["box"]
        meta = info["meta"]

        data.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "label": meta["label"],
            "link": meta["link"]
        })

    return jsonify(data)




@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>YOLO Metadata Stream</title>

<style>
body {
    margin: 0;
    background: #111;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    overflow: hidden;
    font-family: Arial, sans-serif;
}

#video {
    max-width: 90vw;
    max-height: 90vh;
    cursor: pointer;
}

#btn {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 10px 18px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    border: none;
    border-radius: 6px;
    background: #222;
    color: #fff;
    z-index: 999;
}

#btn:hover {
    background: #333;
}
</style>
</head>

<body>

<img id="video" src="/video?ai=0">
<button id="btn">IA: OFF</button>

<script>
const video = document.getElementById("video");
const btn = document.getElementById("btn");

let aiEnabled = false;
let metadata = [];

btn.onclick = () => {
    aiEnabled = !aiEnabled;
    btn.innerText = aiEnabled ? "IA: ON" : "IA: OFF";

    // reinicia stream com flag de IA
    video.src = "/video?ai=" + (aiEnabled ? "1" : "0") + "&t=" + Date.now();
};

// busca metadados periodicamente
async function fetchMetadata() {
    if (!aiEnabled) return;
    const res = await fetch("/metadata");
    metadata = await res.json();
}

// hit-test no clique
video.addEventListener("click", (e) => {
    if (!aiEnabled || metadata.length === 0) return;

    const rect = video.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) * (video.naturalWidth / rect.width);
    const clickY = (e.clientY - rect.top) * (video.naturalHeight / rect.height);

    for (const m of metadata) {
        if (
            clickX >= m.x1 && clickX <= m.x2 &&
            clickY >= m.y1 && clickY <= m.y2
        ) {
            window.open(m.link, "_blank");
            break;
        }
    }
});

// polling leve só de metadados (não do vídeo)
setInterval(fetchMetadata, 300);
</script>

</body>
</html>

"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

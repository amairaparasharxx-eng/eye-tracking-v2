import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const status = document.getElementById("status");

let stream = null;
let landmarker = null;
let running = false;
let lastVideoTime = -1;

const setText = (id, value, suffix = "") => {
  document.getElementById(id).textContent = value == null ? "—" : `${value}${suffix}`;
};

function showMeasure(m) {
  setText("r-fissure", m.right?.fissure_mm?.toFixed?.(2), " mm");
  setText("l-fissure", m.left?.fissure_mm?.toFixed?.(2), " mm");
  setText("r-mrd1", m.right?.mrd1_mm?.toFixed?.(2), " mm");
  setText("l-mrd1", m.left?.mrd1_mm?.toFixed?.(2), " mm");
  setText("yaw", m.head_yaw_deg?.toFixed?.(1), "°");
  setText("pitch", m.head_pitch_deg?.toFixed?.(1), "°");
  setText("roll", m.head_roll_deg?.toFixed?.(1), "°");
  setText("quality", m.quality != null ? `${Math.round(m.quality * 100)}` : null, "%");
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );

  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false
  });
}

function drawLandmarks(result) {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const face = result.faceLandmarks?.[0];
  if (!face) return;

  ctx.fillStyle = "rgba(40, 180, 80, 0.8)";
  for (const p of face) {
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function loop() {
  if (!running || !landmarker) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const timestamp = performance.now();
    const result = landmarker.detectForVideo(video, timestamp);
    drawLandmarks(result);

    const face = result.faceLandmarks?.[0];
    if (face && face.length >= 478) {
      // MediaPipe's browser landmarks are normalized. Convert to the same
      // pixel-coordinate format used by the existing Python measurement code.
      const points = face.map(p => [p.x * video.videoWidth, p.y * video.videoHeight]);
      fetch("/api/measure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          points,
          width: video.videoWidth,
          height: video.videoHeight,
          timestamp: performance.now() / 1000
        })
      })
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data && !data.error) showMeasure(data);
        })
        .catch(() => {});
      status.textContent = "Face detected — tracking.";
    } else {
      status.textContent = "Camera on — move your face into view.";
    }
  }

  requestAnimationFrame(loop);
}

async function startCamera() {
  if (running) return;
  try {
    status.textContent = "Loading face tracker…";
    if (!landmarker) landmarker = await createLandmarker();

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    status.textContent = "Camera on — looking for your face…";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    status.textContent = "Could not start the camera. Allow camera access and try again.";
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  status.textContent = "Camera is off.";
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);

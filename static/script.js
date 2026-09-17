const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const status = document.getElementById("status");

const MEDIAPIPE_VERSION = "0.10.22";
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const WASM_URL = `${MEDIAPIPE_URL}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let stream = null;
let landmarker = null;
let running = false;
let lastVideoTime = -1;
let FaceLandmarkerClass = null;
let FilesetResolverClass = null;

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

async function loadMediaPipe() {
  if (FaceLandmarkerClass && FilesetResolverClass) return;

  status.textContent = "Loading face tracker…";
  try {
    const module = await import(MEDIAPIPE_URL);
    FaceLandmarkerClass = module.FaceLandmarker;
    FilesetResolverClass = module.FilesetResolver;
  } catch (err) {
    console.error("MediaPipe failed to load:", err);
    throw new Error("The face-tracking library could not be loaded. Check your internet connection and refresh the page.");
  }
}

async function createLandmarker() {
  await loadMediaPipe();

  const fileset = await FilesetResolverClass.forVisionTasks(WASM_URL);

  try {
    return await FaceLandmarkerClass.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    });
  } catch (gpuError) {
    console.warn("GPU face tracker failed; trying CPU:", gpuError);
    return await FaceLandmarkerClass.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    });
  }
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
        .catch(err => console.warn("Measurement request failed:", err));
      status.textContent = "Face detected — tracking.";
    } else {
      status.textContent = "Camera on — move your face into view.";
    }
  }

  requestAnimationFrame(loop);
}

async function startCamera() {
  if (running) return;

  startBtn.disabled = true;
  status.textContent = "Starting camera…";

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is not available in this browser or page context.");
    }

    // Ask for camera access before loading the tracker so the button gives
    // immediate feedback and the browser permission prompt can appear.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    status.textContent = "Camera on — loading face tracker…";
    if (!landmarker) landmarker = await createLandmarker();

    running = true;
    stopBtn.disabled = false;
    status.textContent = "Camera on — looking for your face…";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = `Could not start camera: ${err.message || "allow camera access and try again."}`;
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

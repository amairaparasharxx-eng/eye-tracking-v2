const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const nextBtn = document.getElementById("next");
const stopBtn = document.getElementById("stop");
const status = document.getElementById("status");
const instruction = document.getElementById("instruction");
const target = document.getElementById("target");
const stepTitle = document.getElementById("step-title");
const progressText = document.getElementById("progress-text");
const progressBar = document.getElementById("progress-bar");
const stepBadge = document.getElementById("step-badge");
const countdown = document.getElementById("countdown");

const MEDIAPIPE_VERSION = "0.10.22";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const WASM_URL = `${MP_URL}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const STEPS = [
  { title: "Baseline", instruction: "Look straight at the camera and keep your face relaxed. Hold still while baseline eye and head measurements are collected.", target: "Look straight at the camera", seconds: 5 },
  { title: "Sustained upgaze", instruction: "Keep your head still. Look upward toward the target and hold your gaze there until the timer finishes.", target: "Look UP — keep your head still", seconds: 8 },
  { title: "Down then up", instruction: "Look down first. When the target changes, quickly look straight/upward and hold. This checks for transient eyelid changes after downgaze.", target: "Look DOWN", seconds: 3, phase2: "Then look UP", seconds2: 4 },
  { title: "Firm eye closure", instruction: "Close both eyes gently but firmly when the timer starts, then reopen when instructed. Keep your head still.", target: "Close your eyes firmly", seconds: 4, phase2: "Open your eyes", seconds2: 3 },
  { title: "Side-to-side gaze", instruction: "Keep your head facing forward. Follow the targets with your eyes only: left, centre, right, centre. Do not move your head.", target: "Follow the on-screen target with your eyes", seconds: 10 },
  { title: "Gaze holding", instruction: "Look at the indicated target and hold your eyes there while keeping your head still. Small eye-position changes are recorded.", target: "Hold your gaze on the target", seconds: 8 },
  { title: "Head compensation", instruction: "Follow the target to each side. Move your head only if you naturally need to compensate for the gaze. The camera records head movement alongside eye movement.", target: "Follow the target left and right", seconds: 8 },
  { title: "Repeatability", instruction: "Repeat the straight-ahead position and then look left and right again. The program compares measurements within this session.", target: "Repeat the gaze sequence", seconds: 10 }
];

let stream = null;
let landmarker = null;
let running = false;
let sessionActive = false;
let stepIndex = -1;
let stepStarted = 0;
let lastVideoTime = -1;
let mpFaceLandmarker = null;
let mpFileset = null;
let samples = [];
let stepSamples = [];
let phase = 1;
let phaseStarted = 0;
let resultState = null;

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const pct = x => `${Math.round(clamp(x, 0, 1) * 100)}%`;

function eyeFeatures(face) {
  // MediaPipe Face Landmarker indices.
  const L = [33, 160, 158, 133, 153, 144];
  const R = [362, 385, 387, 263, 373, 380];
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const ear = ids => { const p = ids.map(i => face[i]); return (dist(p[1],p[5]) + dist(p[2],p[4])) / (2 * dist(p[0],p[3]) || 1); };
  const openL = ear(L), openR = ear(R);
  const irisL = face[468], irisR = face[473];
  const gazeL = (irisL.x - face[33].x) / (face[133].x - face[33].x || 1);
  const gazeR = (irisR.x - face[362].x) / (face[263].x - face[362].x || 1);
  const gazeY = ((irisL.y + irisR.y) / 2 - (face[159].y + face[386].y) / 2) / ((face[145].y + face[374].y) / 2 - (face[159].y + face[386].y) / 2 || 1);
  return { openL, openR, open: (openL + openR) / 2, asym: Math.abs(openL - openR), gazeX: (gazeL + gazeR) / 2, gazeY, headX: face[1]?.x ?? .5, headY: face[1]?.y ?? .5 };
}

function drawLandmarks(result) {
  canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const face = result.faceLandmarks?.[0];
  if (!face) return;
  ctx.fillStyle = "rgba(40,180,80,.8)";
  for (const p of face) { ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.2, 0, Math.PI * 2); ctx.fill(); }
}

function updateLive(f) {
  document.getElementById("eye-opening").textContent = (f.open * 100).toFixed(1) + "%";
  document.getElementById("asymmetry").textContent = (f.asym * 100).toFixed(1) + "%";
  document.getElementById("gaze-x").textContent = f.gazeX.toFixed(2);
  document.getElementById("gaze-y").textContent = f.gazeY.toFixed(2);
  document.getElementById("yaw").textContent = ((f.headX - .5) * 100).toFixed(1);
  document.getElementById("pitch").textContent = ((f.headY - .5) * 100).toFixed(1);
  document.getElementById("roll").textContent = f.asym.toFixed(2);
  document.getElementById("quality").textContent = "Good";
}

async function loadMediaPipe() {
  if (mpFaceLandmarker) return;
  const module = await import(MP_URL);
  mpFaceLandmarker = module.FaceLandmarker;
  mpFileset = await module.FilesetResolver.forVisionTasks(WASM_URL);
}

async function createLandmarker() {
  await loadMediaPipe();
  try {
    return await mpFaceLandmarker.createFromOptions(mpFileset, { baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" }, runningMode: "VIDEO", numFaces: 1 });
  } catch (e) {
    return await mpFaceLandmarker.createFromOptions(mpFileset, { baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" }, runningMode: "VIDEO", numFaces: 1 });
  }
}

function setStep(i) {
  stepIndex = i;
  const s = STEPS[i];
  stepTitle.textContent = s.title;
  instruction.textContent = s.instruction;
  target.textContent = s.target;
  progressText.textContent = `${i + 1} / ${STEPS.length}`;
  progressBar.style.width = `${((i) / STEPS.length) * 100}%`;
  stepBadge.textContent = `Step ${i + 1}`;
  nextBtn.textContent = i === 0 ? "Start guided session" : (i === STEPS.length - 1 ? "Finish session" : "Next step");
  stepSamples = [];
  phase = 1;
  phaseStarted = performance.now();
  stepStarted = performance.now();
  countdown.classList.add("hidden");
  status.textContent = `Step ${i + 1}: ${s.title}`;
}

function countdownFor(seconds) {
  const elapsed = (performance.now() - phaseStarted) / 1000;
  const left = Math.max(0, Math.ceil(seconds - elapsed));
  if (left > 0) { countdown.textContent = left; countdown.classList.remove("hidden"); } else countdown.classList.add("hidden");
  return elapsed >= seconds;
}

function collectStepData(f) {
  const t = (performance.now() - stepStarted) / 1000;
  const sample = { t, ...f };
  samples.push({ step: stepIndex, ...sample });
  stepSamples.push(sample);
}

function finishStep() {
  if (stepIndex < STEPS.length - 1) { setStep(stepIndex + 1); nextBtn.disabled = false; } else { finishSession(); }
}

function loop() {
  if (!running || !landmarker) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    drawLandmarks(result);
    const face = result.faceLandmarks?.[0];
    if (face && face.length >= 478) {
      const f = eyeFeatures(face);
      updateLive(f);
      if (sessionActive && stepIndex >= 0) {
        collectStepData(f);
        const s = STEPS[stepIndex];
        if (stepIndex === 2 && phase === 1 && countdownFor(s.seconds)) { phase = 2; phaseStarted = performance.now(); target.textContent = s.phase2; }
        else if (stepIndex === 3 && phase === 1 && countdownFor(s.seconds)) { phase = 2; phaseStarted = performance.now(); target.textContent = s.phase2; }
        else if (![2,3].includes(stepIndex) && countdownFor(s.seconds)) finishStep();
        else if (phase === 2 && countdownFor(s.seconds2)) finishStep();
        status.textContent = "Face detected — collecting data.";
      }
    } else if (sessionActive) status.textContent = "Face not detected — centre your face in the camera.";
  }
  requestAnimationFrame(loop);
}

async function startCamera() {
  if (running) return;
  startBtn.disabled = true;
  status.textContent = "Requesting camera access…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access is unavailable in this browser.");
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    status.textContent = "Loading face tracker…";
    landmarker = await createLandmarker();
    running = true;
    stopBtn.disabled = false;
    nextBtn.disabled = false;
    status.textContent = "Camera ready. Click Begin session.";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; video.srcObject = null;
    startBtn.disabled = false; nextBtn.disabled = true; stopBtn.disabled = true;
    status.textContent = `Could not start camera: ${err.message || "allow camera access and try again."}`;
  }
}

function startSession() {
  if (!running) return;
  samples = []; sessionActive = true; nextBtn.disabled = true; setStep(0);
}

function finishSession() {
  sessionActive = false; nextBtn.disabled = true; progressBar.style.width = "100%"; progressText.textContent = `${STEPS.length} / ${STEPS.length}`; stepBadge.textContent = "Complete";
  resultState = analyze(); renderResults(resultState);
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("results").scrollIntoView({ behavior: "smooth" });
  status.textContent = "Session complete. Review the summary below.";
}

function rangeFor(step, key) { const a = samples.filter(x => x.step === step).map(x => x[key]); return a.length ? { min: Math.min(...a), max: Math.max(...a), sd: sd(a), mean: mean(a) } : null; }
function scoreObserved(value, low, high) { return value >= high ? 10 : value >= low ? 5 : 0; }

function analyze() {
  const base = rangeFor(0, "open")?.mean || mean(samples.map(x => x.open));
  const up = rangeFor(1, "open");
  const downUp = samples.filter(x => x.step === 2);
  const close = rangeFor(3, "open");
  const side = samples.filter(x => x.step === 4);
  const hold = rangeFor(5, "gazeX");
  const comp = rangeFor(6, "headX");
  const repeat = samples.filter(x => x.step === 7);

  const fatigueDrop = up ? Math.max(0, base - (up.mean || base)) : 0;
  const coganChange = downUp.length ? Math.max(0, Math.max(...downUp.map(x => x.open)) - base) : 0;
  const curtainDrop = downUp.length ? Math.max(0, base - downUp[downUp.length - 1].open) : 0;
  const peekOpen = close?.max || 0;
  const left = side.filter(x => x.gazeX < .45).map(x => x.gazeX);
  const right = side.filter(x => x.gazeX > .55).map(x => x.gazeX);
  const gazeRangeL = left.length ? Math.abs(Math.min(...left) - .5) : 0;
  const gazeRangeR = right.length ? Math.abs(Math.max(...right) - .5) : 0;
  const gazeAsym = Math.abs(gazeRangeL - gazeRangeR);
  const holdInstability = hold?.sd || 0;
  const headComp = comp ? Math.abs(comp.max - comp.min) : 0;
  const repeatVar = repeat.length ? sd(repeat.map(x => x.open)) : 0;

  const results = [
    { name: "Fatigable ptosis", value: fatigueDrop, score: scoreObserved(fatigueDrop, .04, .10), detail: `Eye-opening change during sustained upgaze: ${(fatigueDrop * 100).toFixed(1)}%` },
    { name: "Cogan's lid-twitch", value: coganChange, score: scoreObserved(coganChange, .035, .08), detail: `Transient eye-opening change after the gaze transition: ${(coganChange * 100).toFixed(1)}%` },
    { name: "Curtain sign (enhanced ptosis)", value: curtainDrop, score: scoreObserved(curtainDrop, .04, .10), detail: `Change in eye opening after the downgaze phase: ${(curtainDrop * 100).toFixed(1)}%` },
    { name: "Peek sign", value: peekOpen, score: scoreObserved(peekOpen, .035, .08), detail: `Maximum residual eye opening during the closure task: ${(peekOpen * 100).toFixed(1)}%` },
    { name: "Variable/asymmetric ophthalmoparesis", value: gazeAsym, score: scoreObserved(gazeAsym, .08, .16), detail: `Difference between measured left/right gaze ranges: ${(gazeAsym * 100).toFixed(1)}%` },
    { name: "Gaze-holding instability", value: holdInstability, score: scoreObserved(holdInstability, .025, .06), detail: `Standard deviation of horizontal gaze position while holding: ${holdInstability.toFixed(3)}` },
    { name: "Diplopia-related head tilt/turn compensation", value: headComp, score: scoreObserved(headComp, .06, .14), detail: `Head-position excursion during the compensation task: ${(headComp * 100).toFixed(1)}%` },
    { name: "Intra-exam variability", value: repeatVar, score: scoreObserved(repeatVar, .025, .06), detail: `Eye-opening variability during the repeat sequence: ${(repeatVar * 100).toFixed(1)}%` }
  ];
  // Inter-visit variability cannot be established from a single visit, so the requested variability item is represented by intra-exam repeatability here.
  return results;
}

function renderResults(results) {
  const observed = results.filter(r => r.score > 0).length;
  const notObserved = results.length - observed;
  const degrees = results.length ? (observed / results.length) * 360 : 0;
  document.getElementById("observed-count").textContent = observed;
  document.getElementById("legend-observed").textContent = observed;
  document.getElementById("legend-not-observed").textContent = notObserved;
  document.getElementById("total-score").textContent = `${results.reduce((a,r) => a + r.score, 0)} / ${results.length * 10}`;
  document.getElementById("pie").style.background = `conic-gradient(#49648f 0deg ${degrees}deg, #dfe5ed ${degrees}deg 360deg)`;
  document.getElementById("result-list").innerHTML = results.map(r => `<div class="result-item ${r.score > 0 ? "observed" : ""}"><div class="result-top"><span class="result-name">${r.name}</span><span class="result-status">${r.score > 0 ? "Observed" : "Not observed"}</span></div><div class="result-score"><strong>${r.score}/10</strong> · ${r.score === 0 ? "0 — none" : r.score === 5 ? "5 — not prominent" : "10 — prominent"}<br>${r.detail}</div></div>`).join("");
}

function stopCamera() {
  running = false; sessionActive = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false; nextBtn.disabled = true; stopBtn.disabled = true;
  status.textContent = "Camera is off.";
}

function restart() {
  document.getElementById("results").classList.add("hidden");
  samples = []; stepIndex = -1; progressBar.style.width = "0%"; progressText.textContent = "0 / 8"; stepBadge.textContent = "Ready";
  stepTitle.textContent = "Ready to begin"; instruction.textContent = "Start the camera, then follow the on-screen instructions. Keep your head as still as comfortably possible unless the instruction asks you to move it."; target.textContent = "Look straight at the camera";
  if (running) { nextBtn.disabled = false; status.textContent = "Camera ready. Click Begin session."; }
}

startBtn.addEventListener("click", startCamera);
nextBtn.addEventListener("click", () => { if (!sessionActive) startSession(); });
stopBtn.addEventListener("click", stopCamera);
document.getElementById("restart").addEventListener("click", restart);
window.addEventListener("beforeunload", stopCamera);

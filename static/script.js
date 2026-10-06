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
const gazeTarget = document.getElementById("gaze-target");
const instructionOverlay = document.getElementById("instruction-overlay");
const instructionOverlayText = document.getElementById("instruction-overlay-text");
const instructionContinueBtn = document.getElementById("instruction-continue");
const eyeConsent = document.getElementById("eye-consent");
const eyeSummaryConsent = document.getElementById("eye-summary-consent");
const eyeDiagnosticConsent = document.getElementById("eye-diagnostic-consent");
const mgSummaryInput = document.getElementById("mg-summary-input");
const combineMgButton = document.getElementById("combine-mg");
const mgImportStatus = document.getElementById("mg-import-status");
const combinedSummary = document.getElementById("combined-summary");

const MEDIAPIPE_VERSION = "0.10.35";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const STEPS = [
  { title:"Baseline", instruction:"Keep your head still and look at the blue dot in the centre.", phases:[{target:"Blue dot: CENTER",x:50,y:50,seconds:30,showDot:true}] },
  { title:"Sustained upgaze — ptosis observation", instruction:"Keep your head still and look only at the upward arrow. Keep your gaze there for the full 60 seconds.", phases:[{target:"Arrow: UP — keep your head still",x:50,y:18,seconds:60,direction:"up"}] },
  { title:"Horizontal saccadic movement", instruction:"Follow the arrows with your eyes only. Use the blue dot only when the target is centred. Keep your head still.", phases:[
    {target:"Arrow: LEFT",x:15,y:50,seconds:30,direction:"left"},
    {target:"Blue dot: CENTRE",x:50,y:50,seconds:30,showDot:true},
    {target:"Arrow: RIGHT",x:85,y:50,seconds:30,direction:"right"},
    {target:"Blue dot: CENTRE",x:50,y:50,seconds:30,showDot:true}
  ] },
  { title:"Gaze holding", instruction:"Keep your head still and hold your gaze on the blue dot in the centre.", phases:[{target:"Blue dot: CENTER — hold",x:50,y:50,seconds:30,showDot:true}] },
  { title:"Head compensation and repeatability", instruction:"Repeat the gaze sequence: centre, left, right, then centre. Follow the arrows for left and right, and the blue dot when centred.", phases:[
    {target:"Blue dot: CENTRE",x:50,y:50,seconds:30,showDot:true},
    {target:"Arrow: LEFT",x:15,y:50,seconds:30,direction:"left"},
    {target:"Arrow: RIGHT",x:85,y:50,seconds:30,direction:"right"},
    {target:"Blue dot: CENTRE",x:50,y:50,seconds:30,showDot:true}
  ] }
];

let stream = null, landmarker = null, running = false, sessionActive = false;
let stepIndex = -1, stepStarted = 0, lastVideoTime = -1;
let mpFaceLandmarker = null, mpFileset = null, samples = [], stepSamples = [];
let phase = 0, phaseStarted = 0, lastGazeX = null;

const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };

function eyeFeatures(face) {
  const L = [33,160,158,133,153,144], R = [362,385,387,263,373,380];
  const dist = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);
  const ear = ids => { const p=ids.map(i=>face[i]); return (dist(p[1],p[5])+dist(p[2],p[4]))/(2*dist(p[0],p[3])||1); };
  const openL=ear(L), openR=ear(R), irisL=face[468], irisR=face[473];
  const gazeL=(irisL.x-face[33].x)/(face[133].x-face[33].x||1);
  const gazeR=(irisR.x-face[362].x)/(face[263].x-face[362].x||1);
  const gazeY=((irisL.y+irisR.y)/2-(face[159].y+face[386].y)/2)/((face[145].y+face[374].y)/2-(face[159].y+face[386].y)/2||1);
  return {openL,openR,open:(openL+openR)/2,asym:Math.abs(openL-openR),gazeX:(gazeL+gazeR)/2,gazeY,headX:face[1]?.x??.5,headY:face[1]?.y??.5};
}

function drawLandmarks(result) {
  canvas.width=video.videoWidth||640; canvas.height=video.videoHeight||480; ctx.clearRect(0,0,canvas.width,canvas.height);
  const face=result.faceLandmarks?.[0]; if(!face)return;
  ctx.fillStyle="rgba(40,180,80,.8)";
  for(const p of face){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,1.2,0,Math.PI*2);ctx.fill();}
}

function updateLive(f){
  document.getElementById("eye-opening").textContent=(f.open*100).toFixed(1)+"%";
  document.getElementById("gaze-x").textContent=f.gazeX.toFixed(2);
  document.getElementById("saccades").textContent=lastGazeX===null?"—":Math.abs(f.gazeX-lastGazeX).toFixed(3);
  document.getElementById("yaw").textContent=((f.headX-.5)*100).toFixed(1)+"%";
  document.getElementById("quality").textContent="Tracking OK";
  lastGazeX=f.gazeX;
}

async function loadMediaPipe(){
  if(mpFaceLandmarker)return;
  const module=await import(MP_URL);
  mpFaceLandmarker=module.FaceLandmarker;
  mpFileset=await module.FilesetResolver.forVisionTasks(WASM_URL);
}

async function createLandmarker(){
  await loadMediaPipe();
  try{return await mpFaceLandmarker.createFromOptions(mpFileset,{baseOptions:{modelAssetPath:MODEL_URL,delegate:"GPU"},runningMode:"VIDEO",numFaces:1});}
  catch(e){console.warn("GPU initialization failed; using CPU",e);return await mpFaceLandmarker.createFromOptions(mpFileset,{baseOptions:{modelAssetPath:MODEL_URL,delegate:"CPU"},runningMode:"VIDEO",numFaces:1});}
}

function positionGazeTarget(x, y) {
  const width = video.clientWidth || video.offsetWidth;
  const height = video.clientHeight || video.offsetHeight;
  if (!width || !height) return;
  gazeTarget.style.left = (width * x / 100) + "px";
  gazeTarget.style.top = (height * y / 100) + "px";
}

function applyPhase(step, phaseIndex) {
  const s = STEPS[step];
  const p = s.phases[phaseIndex];
  if (!p) return;
  target.textContent = p.target;
  instruction.textContent = s.instruction;
  positionGazeTarget(p.x, p.y);
  gazeTarget.classList.remove("arrow-left","arrow-right","arrow-up","center-dot");
  if (p.direction) {
    gazeTarget.textContent = p.direction === "left" ? "←" : p.direction === "right" ? "→" : "↑";
    gazeTarget.classList.add("active", "arrow-" + p.direction);
  } else if (p.showDot) {
    gazeTarget.textContent = "";
    gazeTarget.classList.add("active", "center-dot");
  } else {
    gazeTarget.classList.remove("active");
  }
}

function setStep(i){
  stepIndex=i;
  const s=STEPS[i];
  stepTitle.textContent=s.title;
  instruction.textContent=s.instruction;
  progressText.textContent=(i+1) + " / " + STEPS.length;
  progressBar.style.width=(i/STEPS.length*100) + "%";
  stepBadge.textContent="Step " + (i+1);
  stepSamples=[];
  phase=0;
  phaseStarted=0;
  stepStarted=0;
  countdown.classList.add("hidden");
  applyPhase(i,0);
  instructionOverlayText.textContent=s.instruction;
  instructionOverlay.classList.add("active");
  status.textContent="Step " + (i+1) + ": " + s.title + " — read the instructions, then click Continue.";
}

let audioContext = null;

function playTimerBeep(){
  try{
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    const now = audioContext.currentTime;
    [0, 0.18, 0.36].forEach((offset) => {
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.22, now + offset + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.12);
      osc.connect(gain);
      gain.connect(audioContext.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.13);
    });
  }catch(e){
    console.warn("Timer sound could not play:", e);
  }
}

function playPhaseBeep(){
  try{
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    if(audioContext.state === "suspended") audioContext.resume();
    const now = audioContext.currentTime;
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = "sine";
    osc.frequency.value = 660;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.11);
    osc.connect(gain);
    gain.connect(audioContext.destination);
    osc.start(now);
    osc.stop(now + 0.12);
  }catch(e){
    console.warn("Phase sound could not play:", e);
  }
}

function countdownFor(seconds){
  const elapsed=(performance.now()-phaseStarted)/1000, left=Math.max(0,Math.ceil(seconds-elapsed));
  if(left>0){countdown.textContent=left;countdown.classList.remove("hidden");}else countdown.classList.add("hidden"); return elapsed>=seconds;
}
function collectStepData(f){const sample={t:(performance.now()-stepStarted)/1000,...f};samples.push({step:stepIndex,...sample});stepSamples.push(sample);}
function finishStep(){
  playTimerBeep();
  if(stepIndex<STEPS.length-1){setStep(stepIndex+1);nextBtn.disabled=false;}else finishSession();
}

function loop(){
  if(!running||!landmarker)return;
  if(video.readyState>=2&&video.currentTime!==lastVideoTime){
    lastVideoTime=video.currentTime;
    const result=landmarker.detectForVideo(video,performance.now());
    drawLandmarks(result);
    const face=result.faceLandmarks?.[0];
    if(face&&face.length>=478){
      const f=eyeFeatures(face);
      updateLive(f);
      if(sessionActive&&stepIndex>=0){
        if (instructionOverlay.classList.contains("active")) {
          status.textContent="Step " + (stepIndex+1) + ": " + STEPS[stepIndex].title + " — click Continue when you are ready.";
          requestAnimationFrame(loop);
          return;
        }
        collectStepData(f);
        const s=STEPS[stepIndex];
        const currentPhase=s.phases[phase];
        if(currentPhase && countdownFor(currentPhase.seconds)){
          if(phase < s.phases.length-1){
            phase += 1;
            phaseStarted=performance.now();
            playPhaseBeep();
            applyPhase(stepIndex,phase);
          } else {
            finishStep();
          }
        }
        status.textContent="Face detected — collecting data.";
      }
    } else if(sessionActive){
      status.textContent="Face not detected — centre your face in the camera.";
    }
  }
  requestAnimationFrame(loop);
}

async function startCamera(){
  if(running)return;
  if(!eyeConsent?.checked || !eyeDiagnosticConsent?.checked){
    status.textContent="Please confirm all required consent statements before starting the prototype session.";
    return;
  }
  startBtn.disabled=true; status.textContent="Requesting camera access…";
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("Camera access is unavailable in this browser.");
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},audio:false});
    video.srcObject=stream; await video.play(); status.textContent="Loading face tracker…"; landmarker=await createLandmarker(); running=true;
    stopBtn.disabled=false; nextBtn.disabled=false; status.textContent="Camera ready. Click Begin session."; requestAnimationFrame(loop);
  }catch(err){
    console.error(err); if(stream)stream.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; startBtn.disabled=false; nextBtn.disabled=true; stopBtn.disabled=true;
    status.textContent=`Could not start camera: ${err.message||"allow camera access and try again."}`;
  }
}
function startSession(){
  if(!running)return;
  samples=[];
  lastGazeX=null;
  sessionActive=true;
  nextBtn.disabled=true;
  setStep(0);
}

function finishSession(){
  sessionActive=false;nextBtn.disabled=true;progressBar.style.width="100%";progressText.textContent=`${STEPS.length} / ${STEPS.length}`;stepBadge.textContent="Complete";
  renderResults(analyze()); document.getElementById("results").classList.remove("hidden"); document.getElementById("results").scrollIntoView({behavior:"smooth"}); status.textContent="Session complete. Review the summary below.";
}
function rangeFor(step,key){const a=samples.filter(x=>x.step===step).map(x=>x[key]);return a.length?{min:Math.min(...a),max:Math.max(...a),sd:sd(a),mean:mean(a)}:null;}
function scoreObserved(value){
  const percent=Math.max(0,Math.min(100,value*100));
  if(percent<=5)return 0;
  if(percent<=50)return 5;
  return 10;
}
function observationPercent(value){
  return Math.max(0,Math.min(100,value*100));
}
function scoreLabel(value){
  const percent=observationPercent(value);
  if(percent<=5)return "Not observed";
  if(percent<=25)return "Not prominently observed";
  if(percent<=50)return "Observed";
  if(percent<=75)return "Prominently observed";
  return "Extreme prominence observed";
}

function analyze(){
  const baseline=rangeFor(0,"open");
  const up=rangeFor(1,"open");
  const horizontal=samples.filter(x=>x.step===2);
  const hold=rangeFor(3,"gazeX");
  const repeat=samples.filter(x=>x.step===4);

  const ptosisChange=(baseline&&up) ? Math.max(0,(baseline.mean-up.min)/(baseline.mean||1)) : 0;

  const saccades=[];
  for(let i=1;i<horizontal.length;i++){
    const dt=Math.max(.016,horizontal[i].t-horizontal[i-1].t);
    const delta=Math.abs(horizontal[i].gazeX-horizontal[i-1].gazeX);
    if(delta>.025) saccades.push({delta,velocity:delta/dt,t:horizontal[i].t});
  }
  const midpoint=horizontal.length ? horizontal[Math.floor(horizontal.length/2)].t : 0;
  const first=saccades.filter(x=>x.t<=midpoint);
  const second=saccades.filter(x=>x.t>midpoint);
  const firstAmp=first.length ? mean(first.map(x=>x.delta)) : 0;
  const secondAmp=second.length ? mean(second.map(x=>x.delta)) : 0;
  const saccadeChange=firstAmp>0 ? Math.abs(firstAmp-secondAmp)/firstAmp : 0;

  const holdInstability=hold?.sd||0;
  const repeatGaze=repeat.map(x=>x.gazeX);
  const repeatVar=repeatGaze.length ? sd(repeatGaze) : 0;

  return [
    {name:"Ptosis (eyelid-opening change during sustained upgaze)",score:scoreObserved(ptosisChange),percent:observationPercent(ptosisChange),detail:`Change in estimated eyelid opening during the 60-second sustained-upgaze task: ${(ptosisChange*100).toFixed(1)}%`},
    {name:"Fatigable saccadic movement",score:scoreObserved(saccadeChange),percent:observationPercent(saccadeChange),detail:`Relative change in measured gaze-jump amplitude between the first and second portions of the horizontal task: ${(saccadeChange*100).toFixed(1)}%`},
    {name:"Gaze-holding instability",score:scoreObserved(holdInstability),percent:observationPercent(holdInstability),detail:`Standard deviation of measured horizontal gaze position while holding: ${holdInstability.toFixed(3)}`},
    {name:"Intra-exam variability / repeatability",score:scoreObserved(repeatVar),percent:observationPercent(repeatVar),detail:`Variation in measured horizontal gaze position during the repeated gaze sequence: ${repeatVar.toFixed(3)}`},
  ];
}

function renderResults(results){
  const list=document.getElementById("result-list"), observed=results.filter(r=>r.score>0).length, total=results.reduce((sum,r)=>sum+r.score,0);
  document.getElementById("total-score").textContent=`${total} / ${results.length*10}`; document.getElementById("observed-count").textContent=observed; document.getElementById("legend-observed").textContent=observed; document.getElementById("legend-not-observed").textContent=results.length-observed;
  const angle=observed/results.length*360; document.getElementById("pie").style.background=`conic-gradient(#246bce 0deg ${angle}deg,#dfe5ed ${angle}deg 360deg)`;
  list.innerHTML=results.map(r=>`<article class="result-item"><div><h3>${r.name}</h3><p>${r.detail} · Observation level: ${r.percent.toFixed(0)}%</p></div><div class="result-score"><strong>${r.score}</strong><span>${scoreLabel(r.percent/100)}</span></div></article>`).join("");
}

function escapeHtml(value){
  return String(value ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;")
    .replace(/'/g,"&#039;");
}

function parseMgTransfer(text){
  const parsed=JSON.parse(text);
  if(parsed?.format!=="mg-screening-transfer-v1" || !parsed.questionnaire){
    throw new Error("This does not look like a valid MG Screening v2 summary package.");
  }
  return parsed.questionnaire;
}

function renderCombinedSummary(mg, eyeResults){
  const labels={
    q3:"Diagnosis of MG",q4:"Diagnosis of Thymoma",q5:"Thymectomy",
    q6:"Family history of MG",q7:"Other autoimmune disease",
    q8:"Slurring of speech",q9:"Trouble eating, chewing, or swallowing",
    q10:"Shortness of breath",q11:"Trouble standing from a chair",
    q12:"Diplopia (double vision)",q13:"Ptosis (eyelid droop)"
  };
  const symptomRows=Object.entries(mg.answers||{}).map(([id,a])=>{
    const answer=a.answer?"Yes":"No";
    const severity=a.answer && a.severity!=null ? " (severity "+a.severity+"/10)" : "";
    return "<tr><td>"+escapeHtml(labels[id]||id)+"</td><td>"+answer+severity+"</td></tr>";
  }).join("");
  const eyeRows=eyeResults.map(r=>
    "<tr><td>"+escapeHtml(r.name)+"</td><td>"+r.percent.toFixed(0)+"% — "+escapeHtml(scoreLabel(r.percent/100))+"</td></tr>"
  ).join("");
  const meds=(mg.medications||"").trim() ? escapeHtml(mg.medications) : "None entered";
  combinedSummary.innerHTML=
    "<h3>Combined prototype summary</h3>"+
    "<p><strong>This is an informational combination of two prototype outputs. It does not diagnose MG or any other condition.</strong></p>"+
    "<table class='summary-table'>"+
    "<tr><th>Questionnaire participant</th><td>"+escapeHtml(mg.name||"Not provided")+"</td></tr>"+
    "<tr><th>Age</th><td>"+escapeHtml(mg.age||"Not provided")+"</td></tr>"+
    "<tr><th>MG questionnaire score</th><td>"+escapeHtml(mg.scores?.totalScore)+" / "+escapeHtml(mg.scores?.maxTotalScore)+"</td></tr>"+
    "<tr><th>MG questionnaire band</th><td>"+escapeHtml(mg.grading?.band||"Not provided")+"</td></tr>"+
    "<tr><th>Medications entered</th><td>"+meds+"</td></tr></table>"+
    "<h4>Questionnaire answers</h4>"+
    "<table class='summary-table'><tr><th>Question</th><th>Response</th></tr>"+symptomRows+"</table>"+
    "<h4>Eye observation results</h4>"+
    "<table class='summary-table'><tr><th>Observation</th><th>Prototype observation level</th></tr>"+eyeRows+"</table>"+
    "<p class='result-note'>The questionnaire and camera observations measure different things and should not be interpreted as a combined clinical score. Clinical interpretation requires a qualified healthcare professional.</p>";
  combinedSummary.classList.remove("hidden");
}

function combineMgSummary(){
  if(!eyeSummaryConsent?.checked){
    mgImportStatus.textContent="Please confirm that you consent to combining the questionnaire results with this prototype summary.";
    return;
  }
  if(!mgSummaryInput.value.trim()){
    mgImportStatus.textContent="Paste the copied MG questionnaire summary data first.";
    return;
  }
  try{
    const mg=parseMgTransfer(mgSummaryInput.value.trim());
    renderCombinedSummary(mg,analyze());
    mgImportStatus.textContent="MG questionnaire results were added to the summary. Nothing was uploaded by this action.";
  }catch(err){
    combinedSummary.classList.add("hidden");
    mgImportStatus.textContent=err.message||"Could not read the pasted MG summary.";
  }
}

function stopCamera(){running=false;gazeTarget?.classList.remove("active");instructionOverlay?.classList.remove("active");sessionActive=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;startBtn.disabled=false;nextBtn.disabled=true;stopBtn.disabled=true;status.textContent="Camera is off.";ctx.clearRect(0,0,canvas.width,canvas.height);}
startBtn.addEventListener("click",startCamera);
nextBtn.addEventListener("click",startSession);
instructionContinueBtn?.addEventListener("click",()=>{
  if(!sessionActive || stepIndex<0)return;
  instructionOverlay.classList.remove("active");
  phaseStarted=performance.now();
  stepStarted=performance.now();
  countdown.classList.add("hidden");
  status.textContent="Step " + (stepIndex+1) + ": " + STEPS[stepIndex].title + " — collecting data.";
});
stopBtn.addEventListener("click",stopCamera);
document.getElementById("restart").addEventListener("click",()=>window.location.reload());
combineMgButton?.addEventListener("click",combineMgSummary);
eyeConsent?.addEventListener("change",()=>{ if(!eyeConsent.checked && running) stopCamera(); });
window.addEventListener("beforeunload",stopCamera);

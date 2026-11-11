// SnapFoley test frontend — talks to local backend only
const API = location.origin.includes("localhost") || location.origin.includes("127.0.0.1")
  ? location.origin
  : "http://127.0.0.1:7860";

let taskId = null;

const $ = id => document.getElementById(id);
const els = { dz: $("dropzone"), input: $("videoInput"), preview: $("preview"),
  fileMeta: $("fileMeta"), processBtn: $("processBtn"), progressWrap: $("progressWrap"),
  bar: $("progressBar"), ptext: $("progressText"), taskIdEl: $("taskId"),
  health: $("healthBadge"), results: $("results"), empty: $("empty"),
  statEvents: $("statEvents"), statDuration: $("statDuration"), statPrompt: $("statPrompt"),
  timeline: $("timeline"), eventsBody: $("eventsBody"), apiBase: $("apiBase") };

els.apiBase.textContent = API;

async function checkHealth(){
  try{
    const r = await fetch(API + "/api/health");
    const j = await r.json();
    els.health.className = "health ok";
    els.health.textContent = `● offline • ${j.profile || "local"} • ffmpeg OK`;
  }catch(e){
    els.health.className = "health err";
    els.health.textContent = `✕ backend not reachable (${API}) — run uvicorn app.main:app --port 7860`;
  }
}
checkHealth();

const dz = $("dropzone"), input = $("videoInput");
dz.addEventListener("click", () => input.click());
dz.addEventListener("keydown", e => { if(e.key==="Enter") input.click(); });
["dragenter","dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if(f) setFile(f); });
input.addEventListener("change", () => { if(input.files[0]) setFile(input.files[0]); });

function setFile(file){
  if(!file.type.startsWith("video/")) { alert("Please pick a video file"); return; }
  if(file.size > 200*1024*1024){ alert("File too large (max 200MB)"); return; }
  els.preview.src = URL.createObjectURL(file);
  els.preview.classList.remove("hidden");
  els.fileMeta.textContent = `${file.name} • ${(file.size/1024/1024).toFixed(1)}MB`;
  els.inputEl = file;
  els.processBtn.disabled = false;
  els.progressWrap.classList.add("hidden");
}

$("yoloConf").addEventListener("input", e => $("yoloVal").textContent = Number(e.target.value).toFixed(2));

function setProgress(pct, text){
  els.progressWrap.classList.remove("hidden");
  els.bar.style.width = Math.round(pct*100) + "%";
  els.ptext.textContent = text || `${Math.round(pct*100)}%`;
}

els.processBtn.addEventListener("click", async () => {
  const file = els.inputEl;
  if(!file) return;
  setProgress(0.02, "Uploading video to local backend…");
  const fd = new FormData();
  fd.append("file", file);
  const config = { sensitivity: $("sensitivity").value, yolo_conf: Number($("yoloConf").value) };
  fd.append("config", JSON.stringify(config));

  try{
    const up = await fetch(API + "/api/upload-video", { method: "POST", body: fd });
    if(!up.ok) throw new Error(await up.text());
    const j = await up.json();
    taskId = j.task_id;
    els.taskIdEl.textContent = "task: " + taskId;
    setProgress(0.1, "Uploaded — processing…");

    const pr = await fetch(API + `/api/process/${taskId}`, { method: "POST" });
    if(!pr.ok) throw new Error(await pr.text());
    poll(taskId);
  }catch(e){
    setProgress(0, "Error: " + e.message);
  }
});

async function poll(id){
  try{
    const r = await fetch(API + `/api/status/${id}`);
    const j = await r.json();
    setProgress(j.progress ?? 0.5, `Status: ${j.status} — ${Math.round((j.progress??0)*100)}%`);
    if(j.status === "processing"){ setTimeout(() => poll(id), 1500); return; }
    if(j.status === "completed"){ render(j); setProgress(1, "Done"); return; }
    throw new Error(j.error || "failed");
  }catch(e){ setProgress(0, "Error: " + e.message); }
}

function render(j){
  els.empty.classList.add("hidden");
  els.results.classList.remove("hidden");
  const evs = j.events || [];
  els.statEvents.textContent = evs.length;
  els.statDuration.textContent = (j.timeline?.duration_sec || 0).toFixed(1) + "s";
  els.statPrompt.textContent = (j.prompt || "—").slice(0, 40);
  // events table
  els.eventsBody.innerHTML = evs.map((e,i) => `<tr><td>${i+1}</td><td>${e.timecode||e.start_sec.toFixed(2)+"s"}</td><td>${e.foot||e.type}</td><td>${e.type}</td><td>${Number(e.confidence||0).toFixed(2)}</td></tr>`).join("") || `<tr><td colspan="5">No events (try high sensitivity or better video)</td></tr>`;
  // timeline
  const tl = j.timeline || { tracks: [], duration_sec: 10 };
  const dur = Math.max(tl.duration_sec || 10, 1);
  let html = "";
  (tl.tracks || []).forEach(tr => {
    (tr.clips || []).forEach(c => {
      const ev = evs.find(e => e.event_id === c.event_id) || {};
      const left = ((ev.start_sec || 0) / dur * 100);
      const w = Math.max(((ev.duration_hint || 0.5) / dur * 100), 2);
      html += `<div class="clip ${tr.name==='ambience'?'amb':''}" style="left:${left}%;width:${w}%" title="${c.event_id}"></div>`;
    });
  });
  els.timeline.innerHTML = html || `<div style="padding:12px;color:var(--muted)">No clips yet</div>`;
}

$("genFinalBtn").addEventListener("click", async () => {
  if(!taskId) return;
  $("finalWrap").classList.remove("hidden");
  $("finalStatus").textContent = "Requested final video — rendering (annotated + audio + ffmpeg)…";
  try{
    const r = await fetch(API + `/api/generate-audio-video/${taskId}`, { method: "POST" });
    if(!r.ok) throw new Error(await r.text());
    pollFinal();
  }catch(e){ $("finalStatus").textContent = "Error: " + e.message; }
});
async function pollFinal(){
  const r = await fetch(API + `/api/audio-video-status/${taskId}`);
  const j = await r.json();
  $("finalStatus").textContent = j.audio_video_generating ? "Generating video with audio… keep this tab open" :
    j.audio_video_ready ? "Ready — press play" : (j.audio_video_error || "Not ready");
  if(j.audio_video_generating){ setTimeout(pollFinal, 1500); return; }
  if(j.audio_video_ready){
    $("finalVideo").src = `${API}/api/download-audio-video/${taskId}`;
  }
}
$("dlFinalBtn").addEventListener("click", () => taskId && window.open(`${API}/api/download-audio-video/${taskId}`, "_blank"));
$("csvBtn").addEventListener("click", () => taskId && window.open(`${API}/api/export-csv/${taskId}`, "_blank"));
$("jsonBtn").addEventListener("click", () => taskId && window.open(`${API}/api/export-json/${taskId}`, "_blank"));
$("resetBtn").addEventListener("click", () => location.reload());

$("scriptBtn").addEventListener("click", async () => {
  const f = $("scriptInput").files[0];
  if(!f){ alert("Pick a .txt script"); return; }
  const fd = new FormData(); fd.append("file", f);
  try{
    const r = await fetch(API + "/api/upload-script", { method: "POST", body: fd });
    const j = await r.json();
    $("scriptOut").textContent = JSON.stringify(j, null, 2) + `\n\nTimeline API: ${API}/api/timeline/${j.task_id}`;
    if(j.task_id) taskId = j.task_id;
  }catch(e){ $("scriptOut").textContent = "Error: " + e.message; }
});

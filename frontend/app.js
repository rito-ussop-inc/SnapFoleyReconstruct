// SnapFoley frontend — upload, auto-foley, results, and a manual sound editor.
const API = location.protocol.startsWith("http") ? location.origin : "http://127.0.0.1:7860";

let taskId = null;
let videoDuration = 10;
let editorEvents = [];   // [{event_id?, type, start_sec, gain}] — the user's edit state

const $ = id => document.getElementById(id);
const els = { dz: $("dropzone"), input: $("videoInput"), preview: $("preview"),
  fileMeta: $("fileMeta"), processBtn: $("processBtn"), progressWrap: $("progressWrap"),
  bar: $("progressBar"), ptext: $("progressText"), taskIdEl: $("taskId"),
  health: $("healthBadge"), results: $("results"), empty: $("empty"),
  statEvents: $("statEvents"), statDuration: $("statDuration"), statPrompt: $("statPrompt"),
  timeline: $("timeline"), eventsBody: $("eventsBody"), apiBase: $("apiBase"),
  editBody: $("editBody"), editStatus: $("editStatus"),
  addEventBtn: $("addEventBtn"), applyMixBtn: $("applyMixBtn"),
  finalVideo: $("finalVideo"), finalWrap: $("finalWrap"), finalStatus: $("finalStatus") };

els.apiBase.textContent = API;

async function checkHealth(){
  try{
    const r = await fetch(API + "/api/health");
    const j = await r.json();
    els.health.className = "health ok";
    els.health.textContent = `● offline • ${j.profile || "local"} • ffmpeg OK`;
  }catch(e){
    els.health.className = "health err";
    els.health.textContent = `✕ backend not reachable (${API}) — start the server first`;
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
  els.preview.onloadedmetadata = () => { videoDuration = els.preview.duration || 10; };
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
  const config = { sensitivity: $("sensitivity").value, yolo_conf: Number($("yoloConf").value) };
  try{
    // chunked upload: tunnel proxies (codespaces/cloudflare) 413-reject big single bodies
    setProgress(0.02, "Uploading video…");
    const j = await uploadChunked(file, config,
      p => setProgress(0.02 + 0.08 * p, `Uploading… ${Math.round(p * 100)}% (${(file.size/1024/1024).toFixed(1)}MB)`));
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

const CHUNK_SIZE = 4 * 1024 * 1024; // 4MB per part — safely under tunnel body limits

async function uploadChunked(file, config, onProgress){
  const uploadId = (crypto.randomUUID ? crypto.randomUUID() : "u" + Date.now() + Math.random().toString(36).slice(2))
    .replace(/[^A-Za-z0-9_-]/g, "").slice(0, 64);
  const total = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
  for(let i = 0; i < total; i++){
    const blob = file.slice(i * CHUNK_SIZE, Math.min((i + 1) * CHUNK_SIZE, file.size));
    const fd = new FormData();
    fd.append("upload_id", uploadId);
    fd.append("index", String(i));
    fd.append("chunk", blob, "chunk.bin");
    const r = await fetch(API + "/api/upload-chunk", { method: "POST", body: fd });
    if(!r.ok) throw new Error("upload failed at part " + (i + 1) + "/" + total + ": " + (await r.text()).slice(0, 200));
    onProgress((i + 1) / total);
  }
  const done = await fetch(API + "/api/upload-complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upload_id: uploadId, filename: file.name, config })
  });
  if(!done.ok) throw new Error("finalize failed: " + (await done.text()).slice(0, 200));
  return done.json();
}

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

function detailOf(e){
  if(e.foot) return e.foot;
  if((e.type === "footstep") && e.event_id){
    if(/LEFT/i.test(e.event_id)) return "left step";
    if(/RIGHT/i.test(e.event_id)) return "right step";
  }
  if(e.type === "typing") return "key strike";
  return "—";
}

function render(j){
  els.empty.classList.add("hidden");
  els.results.classList.remove("hidden");
  const evs = j.events || [];
  videoDuration = (j.timeline?.duration_sec || videoDuration) || 10;
  els.statEvents.textContent = evs.length;
  els.statDuration.textContent = videoDuration.toFixed(1) + "s";
  els.statPrompt.textContent = (j.prompt || "—").slice(0, 40);
  els.statPrompt.title = j.prompt || "";
  // events table
  els.eventsBody.innerHTML = evs.map((e,i) =>
    `<tr><td>${i+1}</td><td>${e.timecode || e.start_sec.toFixed(2)+"s"}</td><td>${detailOf(e)}</td><td>${e.type}</td><td>${Number(e.confidence||0).toFixed(2)}</td></tr>`
  ).join("") || `<tr><td colspan="5">No events detected — the audio will be silent (try higher sensitivity)</td></tr>`;
  // timeline
  const tl = j.timeline || { tracks: [], duration_sec: 10 };
  const dur = Math.max(tl.duration_sec || 10, 1);
  let html = "";
  (tl.tracks || []).forEach(tr => {
    (tr.clips || []).forEach(c => {
      const ev = evs.find(e => e.event_id === c.event_id) || {};
      const left = ((ev.start_sec || 0) / dur * 100);
      const w = Math.max(((ev.duration_hint || 0.5) / dur * 100), 2);
      html += `<div class="clip ${tr.name==='ambience'?'amb':''} ${ev.type||''}" style="left:${left}%;width:${w}%" title="${ev.type||'clip'} @ ${(ev.start_sec||0).toFixed(2)}s"></div>`;
    });
  });
  els.timeline.innerHTML = html || `<div style="padding:12px;color:var(--muted)">No clips yet</div>`;
  // editor state = what the server detected
  editorEvents = evs.map(e => ({ event_id: e.event_id, type: e.type,
    start_sec: Number(e.start_sec||0), gain: Number(e.gain ?? 1) }));
  renderEditor();
  els.addEventBtn.disabled = false;
  els.applyMixBtn.disabled = false;
  els.editStatus.textContent = "";
}

// ------------------------------ manual editor --------------------------------

function playheadSec(){
  const v = (!els.finalWrap.classList.contains("hidden") && els.finalVideo.src) ? els.finalVideo : els.preview;
  return Math.max(0, Math.min(v.currentTime || 0, videoDuration - 0.05));
}

function renderEditor(){
  if(!editorEvents.length){
    els.editBody.innerHTML = `<tr><td colspan="4" style="color:var(--muted)">No sounds yet — add one at the playhead.</td></tr>`;
    return;
  }
  els.editBody.innerHTML = editorEvents.map((ev, i) => `
    <tr>
      <td><input type="number" class="ed-time" data-i="${i}" min="0" max="${videoDuration.toFixed(2)}" step="0.05" value="${ev.start_sec.toFixed(2)}" style="width:100%"></td>
      <td><select class="ed-type" data-i="${i}">
        ${["footstep","typing","activity","ambience"].map(t => `<option value="${t}" ${t===ev.type?"selected":""}>${t}</option>`).join("")}
      </select></td>
      <td><input type="range" class="ed-gain" data-i="${i}" min="0" max="2" step="0.05" value="${ev.gain}" style="width:90%;vertical-align:middle">
          <span class="ed-gval" data-i="${i}">${ev.gain.toFixed(2)}</span></td>
      <td><button class="btn ghost ed-del" data-i="${i}" title="delete">✕</button></td>
    </tr>`).join("");

  els.editBody.querySelectorAll(".ed-time").forEach(inp => inp.addEventListener("change", e => {
    const i = +e.target.dataset.i;
    editorEvents[i].start_sec = Math.max(0, Math.min(Number(e.target.value) || 0, videoDuration));
    e.target.value = editorEvents[i].start_sec.toFixed(2);
  }));
  els.editBody.querySelectorAll(".ed-type").forEach(sel => sel.addEventListener("change", e => {
    editorEvents[+e.target.dataset.i].type = e.target.value;
  }));
  els.editBody.querySelectorAll(".ed-gain").forEach(sl => sl.addEventListener("input", e => {
    const i = +e.target.dataset.i;
    editorEvents[i].gain = Number(e.target.value);
    els.editBody.querySelector(`.ed-gval[data-i="${i}"]`).textContent = editorEvents[i].gain.toFixed(2);
  }));
  els.editBody.querySelectorAll(".ed-del").forEach(btn => btn.addEventListener("click", e => {
    editorEvents.splice(+e.target.dataset.i, 1);
    renderEditor();
  }));
}

els.addEventBtn.addEventListener("click", () => {
  editorEvents.push({ type: "activity", start_sec: playheadSec(), gain: 1.0 });
  renderEditor();
  els.editStatus.textContent = `Added one sound at ${playheadSec().toFixed(2)}s — press “Re-render audio” to hear it.`;
});

els.applyMixBtn.addEventListener("click", async () => {
  if(!taskId) return;
  els.applyMixBtn.disabled = true;
  els.editStatus.textContent = "Re-rendering audio with your edits…";
  try{
    const r = await fetch(API + `/api/remix/${taskId}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: editorEvents.map(e => ({
        event_id: e.event_id || null, type: e.type, start_sec: e.start_sec, gain: e.gain })) })
    });
    if(!r.ok) throw new Error((await r.text()).slice(0, 200));
    els.finalWrap.classList.remove("hidden");
    els.finalStatus.textContent = "Re-rendering (audio + ffmpeg mux)…";
    pollFinal(true);
  }catch(e){
    els.editStatus.textContent = "Error: " + e.message;
    els.applyMixBtn.disabled = false;
  }
});

async function pollFinal(bust){
  const r = await fetch(API + `/api/audio-video-status/${taskId}`);
  const j = await r.json();
  els.finalStatus.textContent = j.audio_video_generating ? "Generating video with audio… keep this tab open" :
    j.audio_video_ready ? "Ready — press play" : (j.audio_video_error || "Not ready");
  if(j.audio_video_generating){ setTimeout(() => pollFinal(bust), 1500); return; }
  if(j.audio_video_ready){
    els.finalVideo.src = `${API}/api/download-audio-video/${taskId}` + (bust ? `?v=${Date.now()}` : "");
    if(bust) refreshStats();
  }
  els.applyMixBtn.disabled = false;
}

async function refreshStats(){
  try{
    const j = await (await fetch(API + `/api/status/${taskId}`)).json();
    if(j.status !== "completed") return;
    const evs = j.events || [];
    els.statEvents.textContent = evs.length;
    els.statPrompt.textContent = (j.prompt || "—").slice(0, 40);
    els.eventsBody.innerHTML = evs.map((e,i) =>
      `<tr><td>${i+1}</td><td>${e.timecode || e.start_sec.toFixed(2)+"s"}</td><td>${detailOf(e)}</td><td>${e.type}</td><td>${Number(e.confidence||0).toFixed(2)}</td></tr>`).join("");
  }catch(e){}
}

$("genFinalBtn").addEventListener("click", async () => {
  if(!taskId) return;
  els.finalWrap.classList.remove("hidden");
  els.finalStatus.textContent = "Requested final video — rendering (annotated + audio + ffmpeg)…";
  try{
    const r = await fetch(API + `/api/generate-audio-video/${taskId}`, { method: "POST" });
    if(!r.ok) throw new Error(await r.text());
    pollFinal(false);
  }catch(e){ els.finalStatus.textContent = "Error: " + e.message; }
});

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

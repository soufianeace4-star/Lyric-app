import os, re, json, uuid, subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import anthropic

DATA = Path("jobs"); DATA.mkdir(exist_ok=True)
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = anthropic.Anthropic()  # يقرأ ANTHROPIC_API_KEY من البيئة
JOBS, _whisper = {}, None
app = FastAPI()

def whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(os.getenv("WHISPER_MODEL", "small"), compute_type="int8")
    return _whisper

def translate(lines, lang):
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        system=f"You translate subtitles into {lang}. Natural, context-aware, concise. "
               "Return ONLY a JSON array of strings with exactly the same length and order as the input.",
        messages=[{"role": "user", "content": json.dumps(lines, ensure_ascii=False)}])
    txt = re.sub(r"^```(?:json)?|```$", "", msg.content[0].text.strip()).strip()
    out = json.loads(txt)
    return (out + [""] * len(lines))[:len(lines)]

def process(job_id, path, lang):
    job = JOBS[job_id]
    try:
        wav = path.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, capture_output=True)
        job.update(status="transcribing", progress=10)
        segs, info = whisper().transcribe(str(wav), vad_filter=True)
        dur = info.duration or 1
        cues = job["cues"]
        for s in segs:
            cues.append({"s": s.start, "e": s.end, "t": s.text.strip(), "tr": ""})
            job["progress"] = 10 + int(50 * min(s.end / dur, 1))
        job.update(status="translating", lang_detected=info.language)
        for i in range(0, len(cues), 40):
            batch = cues[i:i + 40]
            for c, t in zip(batch, translate([c["t"] for c in batch], lang)):
                c["tr"] = t
            job["progress"] = 60 + int(40 * min((i + 40) / len(cues), 1))
        job.update(status="done", progress=100)
    except Exception as e:
        job.update(status="error", error=str(e))
    finally:
        path.unlink(missing_ok=True); path.with_suffix(".wav").unlink(missing_ok=True)

@app.post("/api/jobs")
async def create(bg: BackgroundTasks, file: UploadFile = File(...), lang: str = Form("Arabic")):
    job_id = uuid.uuid4().hex[:12]
    path = DATA / f"{job_id}{Path(file.filename).suffix}"
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    JOBS[job_id] = {"status": "queued", "progress": 0, "cues": []}
    bg.add_task(process, job_id, path, lang)
    return {"id": job_id}

@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]

app.mount("/", StaticFiles(directory="static", html=True), name="static")

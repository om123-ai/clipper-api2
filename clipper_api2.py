#!/usr/bin/env python3
"""
clipper_service_fixed.py

Fixed and hardened version of the clipper_service Flask app you provided.

Main improvements / fixes applied:
- More robust OpenAI API response handling (tries multiple client call shapes).
- Better JSON extraction from the AI reply (more tolerant regex and fallbacks).
- Improved ffmpeg clipping (explicit re-encoding to avoid copy/seek issues).
- Safer thread / Flask reloader note (run with use_reloader=False to avoid double threads).
- Clearer job status updates and error handling.
- Minor utility improvements and logging.

Requirements:
- yt-dlp and ffmpeg on PATH
- Python packages: flask, openai (or the OpenAI Python SDK compatible with the import used),
  or adapt the OpenAI client initialization / calls to match your installed SDK.
- Set OPENAI_API_KEY in the environment.

Run (development):
  python clipper_service_fixed.py

In production, run under gunicorn/uvicorn and *do not* use Flask's debug reloader.
"""

import os
import re
import json
import uuid
import shlex
import traceback
import subprocess
from threading import Thread, Lock
from flask import Flask, request, jsonify, send_from_directory

# If your OpenAI client differs, adapt this import.
from openai import OpenAI

# --- Configuration ---
app = Flask(__name__, static_folder="static")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable")

# initialize client (adjust if your SDK requires a different form)
client = OpenAI(api_key=OPENAI_API_KEY)

JOBS = {}
JOBS_LOCK = Lock()

# Output directories
BASE_OUTPUT = os.path.abspath("static")
CLIPS_DIR = os.path.join(BASE_OUTPUT, "clips")
VIDEOS_DIR = os.path.join(BASE_OUTPUT, "videos")
SUBS_DIR = os.path.join(BASE_OUTPUT, "subs")
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(SUBS_DIR, exist_ok=True)


# --- Helpers ---

def sanitize_filename(name: str) -> str:
    """Removes characters that are invalid for file names."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def time_to_seconds(t):
    """
    Accepts time strings like "HH:MM:SS", "MM:SS", "SS", or numbers (int/float) and returns seconds (float).
    Handles fractional seconds as well.
    """
    if isinstance(t, (int, float)):
        return float(t)
    t = str(t).strip()
    # already a number (like "12.5")
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    parts = t.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"Unrecognized time format: {t}")


def seconds_to_hhmmss(s: float):
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}"  # includes milliseconds


def run_cmd(cmd, cwd=None, check=True, capture_output=True, text=True):
    """Run subprocess command and return CompletedProcess. Raises on non-zero if check=True."""
    if isinstance(cmd, (list, tuple)):
        pass
    else:
        # allow string command
        cmd = shlex.split(cmd)
    # Print command for debugging
    print("RUN:", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=capture_output, text=text)


def extract_video_id(url):
    """
    Try a few common ways to extract YouTube id; fallback to sanitized uuid if not found.
    """
    m = re.search(r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    # fallback: use uuid portion
    return str(uuid.uuid4())[:12]


def safe_update_job(job_id, **kwargs):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)


# Helper to robustly get text from OpenAI responses across SDK versions
def _extract_ai_text(response_obj):
    # Try typical shapes
    try:
        # If response is a dict-like
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices")
            if choices:
                c = choices[0]
                # depending on shape
                if isinstance(c, dict):
                    m = c.get("message") or {}
                    return m.get("content") or c.get("text")
        else:
            # object with attributes
            # Chat completion style: response.choices[0].message.content
            try:
                return response_obj.choices[0].message.content
            except Exception:
                pass
            # Responses API style: response.output_text or response.output[0].content[0].text
            try:
                return getattr(response_obj, "output_text")
            except Exception:
                pass
            try:
                out = response_obj.output
                if out and len(out) > 0:
                    # try to find text content structures
                    first = out[0]
                    if isinstance(first, dict):
                        # maybe {'content': [{'type':'output_text','text':'...'}]}
                        cont = first.get("content") or []
                        for c in cont:
                            if isinstance(c, dict) and (c.get("type") == "output_text" or c.get("type") == "text"):
                                return c.get("text")
            except Exception:
                pass
    except Exception:
        pass
    # final fallback to str()
    return str(response_obj)


# --- Main job ---
def run_clipping_job(job_id, video_url):
    safe_update_job(job_id, status="processing", message="Starting job")
    try:
        video_id = extract_video_id(video_url)
        base_name = sanitize_filename(video_id)

        # 1) Download subtitles (auto English) using yt-dlp
        safe_update_job(job_id, message="Downloading subtitles (if available)...")
        sub_output_template = os.path.join(SUBS_DIR, "%(id)s.%(ext)s")
        try:
            run_cmd([
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt/srv3/srt",
                "-o", sub_output_template,
                video_url
            ])
        except subprocess.CalledProcessError as e:
            # subtitles may not exist; continue but warn in job
            print("yt-dlp subtitle call failed:", e)
            safe_update_job(job_id, message="No auto-subtitles found or yt-dlp error while fetching subtitles (proceeding).")

        # locate subtitle file
        found_sub = None
        # try variants yt-dlp might produce
        candidates = [
            os.path.join(SUBS_DIR, f"{video_id}.en.vtt"),
            os.path.join(SUBS_DIR, f"{video_id}.en.srt"),
            os.path.join(SUBS_DIR, f"{video_id}.vtt"),
            os.path.join(SUBS_DIR, f"{video_id}.srt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                found_sub = candidate
                break

        transcript_text = ""
        if found_sub:
            safe_update_job(job_id, message=f"Reading subtitles: {os.path.basename(found_sub)}")
            with open(found_sub, "r", encoding="utf-8", errors="ignore") as f:
                transcript_text = f.read()
        else:
            safe_update_job(job_id, message="No subtitles found; attempting to extract description as fallback.")
            try:
                cp = run_cmd(["yt-dlp", "--dump-json", video_url])
                # last line of dump-json is the JSON for the video
                info = json.loads(cp.stdout.splitlines()[-1]) if cp.stdout else {}
                transcript_text = info.get("description", "") or ""
            except Exception as e:
                print("Failed to get description via yt-dlp:", e)
                transcript_text = ""

        if not transcript_text.strip():
            safe_update_job(job_id, message="No transcript/subtitles found; AI will have limited input.")

        # 2) Ask OpenAI for highlight segments
        safe_update_job(job_id, message="Asking AI to identify highlight segments...")

        ai_prompt = (
            "You are a video editor assistant. Based on the transcript/text below, identify up to 3 highlight moments.\n"
            "Return a JSON ARRAY of objects inside a ```json code block. Each object must contain:\n"
            ' - "start_time": start (format HH:MM:SS or seconds),\n'
            ' - "end_time": end (format HH:MM:SS or seconds),\n'
            ' - "reason": short explanation (max 140 chars).\n'
            "Clips should be between 20 and 75 seconds. If transcript is empty, you may return an empty array.\n\n"
            "Transcript/Text:\n\n"
            + transcript_text
        )

        raw_content = None
        # Try a couple of client shapes to be resilient to SDK differences
        try:
            try:
                # chat completion style (some SDKs provide this)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful video editing assistant that responds in JSON format."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    max_tokens=800,
                )
                raw_content = _extract_ai_text(response)
            except Exception as e_chat:
                print("chat.completions failed:", e_chat)
                # Try Responses API shape
                try:
                    response = client.responses.create(
                        model="gpt-4o",
                        input=ai_prompt,
                        max_tokens=800
                    )
                    raw_content = _extract_ai_text(response)
                except Exception as e_resp:
                    print("responses.create failed:", e_resp)
                    # final fallback: raise
                    raise

            if not raw_content:
                raise ValueError("Empty AI response")

        except Exception as e:
            safe_update_job(job_id, status="error", message=f"OpenAI request failed: {e}")
            return

        # Extract JSON array from raw_content robustly
        ai_segments = []
        try:
            # first try to find ```json block (allow possible leading spaces)
            match = re.search(r"```json\s*(\[.*?\])\s*```", raw_content, re.DOTALL | re.IGNORECASE)
            if match:
                json_text = match.group(1)
                ai_segments = json.loads(json_text)
            else:
                # fallback: find first outermost JSON array in the text
                first = raw_content.find("[")
                last = raw_content.rfind("]")
                if first != -1 and last != -1 and last > first:
                    json_text = raw_content[first:last+1]
                    ai_segments = json.loads(json_text)

            if not isinstance(ai_segments, list):
                raise ValueError("AI did not return a JSON array")

        except Exception as e:
            print("Failed to parse JSON from AI response:", e)
            print("AI raw content:\n", raw_content)
            ai_segments = []

        if not ai_segments:
            safe_update_job(job_id, message="AI returned no segments. Marking job complete with no clips.", status="complete", clips=[])
            return

        # 3) Download the full video robustly (merged mp4)
        safe_update_job(job_id, message="Downloading video (full) for reliable clipping...")
        video_output_template = os.path.join(VIDEOS_DIR, f"{base_name}.%(ext)s")
        try:
            run_cmd([
                "yt-dlp",
                "-f", "bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", video_output_template,
                video_url
            ])
        except subprocess.CalledProcessError as e:
            safe_update_job(job_id, status="error", message=f"yt-dlp failed to download video: {e}")
            return

        # find downloaded file
        downloaded_file = None
        for ext in ("mp4", "mkv", "webm", "mpv"):
            candidate = os.path.join(VIDEOS_DIR, f"{base_name}.{ext}")
            if os.path.exists(candidate):
                downloaded_file = candidate
                break
        if not downloaded_file:
            # fallback: use most recent file in VIDEOS_DIR
            files = [os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR)]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                downloaded_file = max(files, key=os.path.getmtime)
        if not downloaded_file:
            safe_update_job(job_id, status="error", message="Could not find downloaded video file after yt-dlp run.")
            return

        # probe duration
        total_dur = None
        try:
            probe = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", downloaded_file])
            total_dur = float(probe.stdout.strip())
        except Exception:
            total_dur = None

        # 4) Create clips
        final_clips = []
        for idx, seg in enumerate(ai_segments):
            safe_update_job(job_id, message=f"Processing AI segment {idx+1}/{len(ai_segments)}...")
            try:
                start_raw = seg.get("start_time") or seg.get("start") or seg.get("from")
                end_raw = seg.get("end_time") or seg.get("end") or seg.get("to")
                reason = (seg.get("reason") or seg.get("label") or "AI Highlight").strip()
                if start_raw is None or end_raw is None:
                    print("Segment missing start/end; skipping:", seg)
                    continue
                start_sec = time_to_seconds(start_raw)
                end_sec = time_to_seconds(end_raw)
                if end_sec <= start_sec:
                    print("Segment end <= start; skipping:", seg)
                    continue
                duration = end_sec - start_sec

                # Enforce clip length bounds: between 20 and 75 seconds
                MIN_CLIP = 20.0
                MAX_CLIP = 75.0
                if duration < MIN_CLIP:
                    end_sec = start_sec + MIN_CLIP
                    duration = end_sec - start_sec
                if duration > MAX_CLIP:
                    end_sec = start_sec + MAX_CLIP
                    duration = MAX_CLIP

                # clamp to file duration if known
                if total_dur is not None:
                    if start_sec >= total_dur:
                        print("Start beyond file duration; skipping segment")
                        continue
                    if end_sec > total_dur:
                        end_sec = total_dur
                        duration = end_sec - start_sec
                        if duration < MIN_CLIP:
                            print("Adjusted duration too short after clamping; skipping")
                            continue

                clip_filename = f"{job_id}_clip_{idx+1}.mp4"
                clip_path = os.path.join(CLIPS_DIR, clip_filename)

                # Use accurate seeking: -ss before -i can be fast but less accurate for re-encoding; we will
                # use -ss before -i for speed and re-encode to ensure compatibility. Provide encoding settings.
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    # seek (precise because we'll re-encode)
                    "-ss", f"{start_sec:.3f}",
                    "-i", downloaded_file,
                    "-t", f"{duration:.3f}",
                    # re-encode to ensure playable result and accurate trimming
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    # metadata
                    "-movflags", "+faststart",
                    clip_path
                ]

                run_cmd(ffmpeg_cmd)

                public_url = f"/clips/{clip_filename}"
                final_clips.append({
                    "reason": reason,
                    "url": public_url,
                    "start_time": round(start_sec, 3),
                    "end_time": round(start_sec + duration, 3)
                })
            except Exception as e:
                print("Error processing segment", idx, e)
                safe_update_job(job_id, message=f"Error processing segment {idx+1}: {e}")
                continue

        # finalize
        safe_update_job(job_id, status="complete", message="Clipping complete", clips=final_clips)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in job {job_id}: {e}\n{tb}")
        safe_update_job(job_id, status="error", message=str(e))


# Health check
@app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy"}), 200


# API Endpoints
@app.route('/api/create-clips', methods=['POST'])
def create_clips_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    video_url = payload.get("url") or request.args.get("url")
    if not video_url:
        return jsonify({"error": "URL is required"}), 400
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'pending', 'message': 'Job queued', 'clips': []}
    thread = Thread(target=run_clipping_job, args=(job_id, video_url), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id}), 202


@app.route('/api/check-status/<job_id>', methods=['GET'])
def check_status_endpoint(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# Serve static clips (simple)
@app.route('/clips/<path:filename>', methods=['GET'])
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename, as_attachment=False)




# Root route for Render (so it doesn’t 404 on /)
@app.route("/")
def home():
    return jsonify({
        "message": "Clipper API is running!",
        "endpoints": {
            "health": "/api/health",
            "create_clips": "/api/create-clips",
            "check_status": "/api/check-status/<job_id>",
            "clips": "/clips/<filename>"
        }
    })


if __name__ == '__main__':
    # For development / production
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)


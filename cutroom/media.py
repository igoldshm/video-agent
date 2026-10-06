"""Thin ffmpeg/ffprobe wrappers: the media-processing layer under the tools."""
from __future__ import annotations

import json
import os
import subprocess

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
THUMB = 16  # frames are analysed at THUMB x THUMB pixels


def ffmpeg(args: list[str]) -> bytes:
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True)
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode(errors="replace")[-400:])
    return p.stdout


def probe(path: str) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                        "-show_streams", path], capture_output=True)
    if p.returncode:
        raise RuntimeError("ffprobe failed: " + p.stderr.decode(errors="replace")[-400:])
    info = json.loads(p.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    return {
        "duration": float(info["format"].get("duration", 0)),
        "width": video["width"] if video else 0,
        "height": video["height"] if video else 0,
        "has_video": video is not None,
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def thumb_frames(path: str) -> list[bytes]:
    """Every frame downscaled to THUMB x THUMB RGB. Cheap enough to run on each clip."""
    raw = ffmpeg(["-i", path, "-vf", f"scale={THUMB}:{THUMB}:flags=area", "-f", "rawvideo",
                  "-pix_fmt", "rgb24", "-"])
    n = THUMB * THUMB * 3
    return [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)]


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def concat(clips: list[str], audio: str | None, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    listing = out_path + ".txt"
    with open(listing, "w") as f:
        f.writelines(f"file '{os.path.abspath(c)}'\n" for c in clips)
    args = ["-f", "concat", "-safe", "0", "-i", listing]
    if audio:
        args += ["-i", audio, "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    ffmpeg(args)
    os.remove(listing)


def tone_bed(path: str, freq: int, seconds: int = 60) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-af", "volume=0.2",
            "-c:a", "aac", path])


def normalize(src: str, dst: str, width: int, height: int, duration: float) -> None:
    """Bring a clip from any model to one size, frame rate and codec so clips can be joined."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1")
    ffmpeg(["-i", src, "-t", f"{duration}", "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])


def jpeg_frames(path: str, n: int = 3, width: int = 512) -> list[bytes]:
    """n evenly spaced frames as JPEG bytes, for a vision model to look at."""
    duration = probe(path)["duration"]
    return [ffmpeg(["-ss", f"{duration * (i + 0.5) / n:.2f}", "-i", path, "-frames:v", "1",
                    "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-"]) for i in range(n)]


def to_jpeg(src: str, dst: str, width: int, height: int | None = None) -> None:
    """Re-encode any still as a JPEG small enough to send inline; with `height`, fit and pad to that frame."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
          if height else f"scale={width}:-2")
    ffmpeg(["-i", src, "-vf", vf, "-frames:v", "1", "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", dst])


def still_jpeg(path: str, width: int = 512) -> bytes:
    """One JPEG for a vision model, from either a still image or the middle of a clip."""
    if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return ffmpeg(["-i", path, "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-"])
    return jpeg_frames(path, 1, width)[0]

"""Clean a talking-head video and export a synced MP4 and SRT."""
import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import tempfile

SPEED = 1.2
PAUSE = 0.15
GUARD = 0.045
FILLERS = re.compile(r"^(?:u+h+|u+m+|e+r+m*|h+m+|m{2,}|[эе]{2,}|м{2,}|эм+)$")


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def duration(path, stream=None):
    args = ("-select_streams", stream) if stream else ()
    result = run("ffprobe", "-v", "error", *args, "-show_entries",
                 "stream=duration" if stream else "format=duration",
                 "-of", "json", str(path))
    data = json.loads(result.stdout)
    return float(data["streams"][0]["duration"] if stream else data["format"]["duration"])


def recognize(model, audio, language):
    segments, _ = model.transcribe(str(audio), language=language,
                                   word_timestamps=True, vad_filter=True)
    return [(w.start, w.end, w.word.strip(), w.probability)
            for segment in segments for w in (segment.words or []) if w.word.strip()]


def normalize(word):
    return re.sub(r"[^\w]", "", word.casefold())


def speech_cuts(words):
    tokens = [normalize(w[2]) for w in words]
    cuts = []
    for i, word in enumerate(words):
        t = tokens[i]
        if FILLERS.fullmatch(t) and word[3] >= 0.40:
            cuts.append((word[0], word[1]))
            continue
        # Adjacent false starts and repeated runs. Require the second
        # version to begin immediately or after a short hesitation.
        for count in range(min(12, i, len(words) - i), 0, -1):
            first, second = tokens[i-count:i], tokens[i:i+count]
            exact = first == second and all(first)
            near = count >= 5 and SequenceMatcher(None, first, second).ratio() >= 0.84
            if (exact or near) and word[0] - words[i-1][1] <= 0.7:
                cuts.append((words[i-count][0], words[i-1][1]))
                break
        if (i + 1 < len(words) and word[2].endswith("-") and t
                and tokens[i+1].startswith(t) and words[i+1][0] - word[1] < 0.5):
            cuts.append((word[0], word[1]))
    return cuts


def detect(model, audio, language, speed):
    words = recognize(model, audio, language)
    if not words:
        raise ValueError("Speech recognition produced no words")
    log = run("ffmpeg", "-hide_banner", "-i", str(audio), "-af",
              "silencedetect=n=-38dB:d=0.22", "-f", "null", "-").stderr
    starts = re.findall(r"silence_start: ([0-9.]+)", log)
    ends = re.findall(r"silence_end: ([0-9.]+)", log)
    total = duration(audio)
    half = PAUSE * speed / 2
    cuts = speech_cuts(words)
    for a, b in zip(starts, ends, strict=True):
        a, b = float(a), float(b)
        # Protect recognized word boundaries, including quiet consonants.
        overlaps = [(x, y) for x, y, *_ in words if x < b and a < y]
        if overlaps:
            continue
        before = max((y for _, y, *_ in words if y <= a), default=0)
        after = min((x for x, *_ in words if x >= b), default=total)
        left = max(a + half, before + GUARD) if a > .02 else 0
        right = min(b - half, after - GUARD) if b < total - .02 else total
        if right - left > .035:
            cuts.append((left, right))
    return words, cuts


def keep_ranges(cuts, length):
    kept = []
    cursor = 0.0
    for a, b in sorted(cuts):
        a, b = max(0, a), min(length, b)
        if b <= a:
            continue
        if a > cursor:
            kept.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < length:
        kept.append((cursor, length))
    if not kept:
        raise ValueError("Edits would remove the whole file")
    return kept


def compose(source, edits, speed):
    """Project cuts detected in the preview back onto original media time."""
    result = []
    cursor = 0.0
    for a, b in source:
        end = cursor + (b-a) / speed
        for start, stop in edits:
            left, right = max(start, cursor), min(stop, end)
            if right > left:
                result.append((a + (left-cursor)*speed, a + (right-cursor)*speed))
        cursor = end
    return result


def graph(ranges, video):
    parts, labels = [], []
    for i, (a, b) in enumerate(ranges):
        if video:
            parts.append(f"[0:v]trim=start={a:.6f}:end={b:.6f},setpts=PTS-STARTPTS[v{i}]")
            labels.append(f"[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")
    parts.append("".join(labels) + f"concat=n={len(ranges)}:v={int(video)}:a=1"
                 + ("[picture][audio]" if video else "[audio]"))
    if video:
        parts.append(f"[picture]setpts=PTS/{SPEED}[v]")
    parts.append(f"[audio]atempo={SPEED}[a]")
    return ";".join(parts)


def stamp(t):
    ms = round(t * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    from faster_whisper import WhisperModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=("ru", "en"))
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    subtitles = args.output.with_suffix(".srt")
    if args.output.suffix.lower() != ".mp4" or args.output.exists() or subtitles.exists():
        parser.error("Choose a new .mp4 output path")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    with tempfile.TemporaryDirectory(dir=args.output.resolve().parent) as scratch:
        tmp = Path(scratch)
        source_audio, preview, encoded = (tmp / name for name in ("source.wav", "check.wav", "checked.mp4"))
        run("ffmpeg", "-v", "error", "-i", str(args.source), "-vn", "-ar", "16000",
            "-ac", "1", str(source_audio))
        length = duration(source_audio)
        _, cuts = detect(model, source_audio, args.language, SPEED)
        ranges = keep_ranges(cuts, length)
        for _ in range(4):
            run("ffmpeg", "-v", "error", "-y", "-i", str(source_audio),
                "-filter_complex", graph(ranges, False), "-map", "[a]", str(preview))
            words, issues = detect(model, preview, args.language, 1)
            if not issues:
                break
            ranges = compose(ranges, keep_ranges(issues, duration(preview)), SPEED)
        else:
            raise RuntimeError("Speech cleanup did not converge. Review the input before export.")
        run("ffmpeg", "-v", "error", "-i", str(args.source),
            "-filter_complex", graph(ranges, True), "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-fps_mode", "vfr", "-c:a", "aac",
            "-b:a", "192k", "-movflags", "+faststart", str(encoded))
        audio_end, video_end = duration(encoded, "a:0"), duration(encoded, "v:0")
        if max(abs(audio_end-video_end), abs(audio_end-duration(preview))) > 0.10:
            raise RuntimeError("Audio/video duration mismatch")
        entries = []
        end = 0.0
        for a, b, word, _ in words:
            a, b = max(0, a, end), min(audio_end, video_end, b)
            if b - a < .035:
                continue
            entries.append(f"{len(entries)+1}\n{stamp(a)} --> {stamp(b)}\n{word}")
            end = b
        if not entries:
            raise RuntimeError("No subtitle cues could be synchronized")
        text = "\n\n".join(entries) + "\n"
        (tmp / "checked.srt").write_text(text, encoding="utf-8")
        encoded.rename(args.output)
        (tmp / "checked.srt").rename(subtitles)


if __name__ == "__main__":
    main()

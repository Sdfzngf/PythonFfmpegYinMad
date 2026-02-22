import argparse
import heapq
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import mido
except ImportError:
    mido = None


@dataclass
class NoteEvent:
    start: float
    end: float
    note: int
    lane: int = 0


def die(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def has_stream(path: str, stream_type: str) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream_type,
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        path,
    ]
    if not os.path.isfile(path):
        return False
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def build_atempo_chain(tempo: float) -> str:
    factors = []
    t = tempo
    while t < 0.5:
        factors.append(0.5)
        t /= 0.5
    while t > 2.0:
        factors.append(2.0)
        t /= 2.0
    factors.append(t)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def get_stream_duration(path: str, stream_type: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream_type,
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if value:
        return float(value)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_notes(mid_path: str) -> List[NoteEvent]:
    if mido is None:
        die("缺少依赖：mido。请先执行：pip install mido")

    mid = mido.MidiFile(mid_path)
    tempo = 500000
    time_sec = 0.0
    active: Dict[Tuple[int, int], List[float]] = {}
    notes: List[NoteEvent] = []

    for msg in mido.merge_tracks(mid.tracks):
        time_sec += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            channel = getattr(msg, "channel", 0)
            key = (msg.note, channel)
            active.setdefault(key, []).append(time_sec)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            channel = getattr(msg, "channel", 0)
            key = (msg.note, channel)
            if key in active and active[key]:
                start = active[key].pop()
                if not active[key]:
                    del active[key]
                end = time_sec
                if end > start:
                    notes.append(NoteEvent(start=start, end=end, note=msg.note))

    notes.sort(key=lambda n: n.start)
    return notes


def assign_lanes(notes: List[NoteEvent]) -> int:
    active: List[tuple[float, int]] = []
    available: List[int] = []
    next_lane = 0
    max_poly = 0

    for note in notes:
        while active and active[0][0] <= note.start + 1e-9:
            _, lane = heapq.heappop(active)
            heapq.heappush(available, lane)

        if available:
            note.lane = heapq.heappop(available)
        else:
            note.lane = next_lane
            next_lane += 1

        heapq.heappush(active, (note.end, note.lane))
        max_poly = max(max_poly, len(active))

    return max_poly


def build_segments(
    notes: List[NoteEvent],
    src_video: str,
    out_video: str,
    temp_dir: str,
    mix_chunk: int,
    resume: bool,
    resume_from: int,
    crf: float,
    preset: str,
) -> None:
    notes = [note for note in notes if (note.end - note.start) > 1e-6]
    if not notes:
        die("MIDI 中没有可用音符。")

    if os.path.isdir(temp_dir) and not resume:
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    src_duration = get_stream_duration(src_video, "v")
    if src_duration <= 0:
        die("源视频时长无效。")

    width = 1280
    height = 720
    fps = 30
    min_aac_dur = 1024.0 / 48000.0

    max_poly = assign_lanes(notes)
    total_duration = max(n.end for n in notes)

    grid_cols = max(1, math.ceil(math.sqrt(max_poly)))
    grid_rows = max(1, math.ceil(max_poly / grid_cols))
    cell_w = max(1, width // grid_cols)
    cell_h = max(1, height // grid_rows)

    segments = []
    seg_infos = []

    for idx, note in enumerate(notes):
        note_duration = note.end - note.start
        if note_duration <= 1e-6:
            continue

        semitones = note.note - 60
        tempo = src_duration / note_duration
        pitch_ratio = 2 ** (semitones / 12.0)
        seg_fps = min(1000, max(fps, math.ceil(1.0 / note_duration)))
        fade_dur = min(0.01, max(0.002, note_duration / 2.0))
        fade_out_start = max(0.0, note_duration - fade_dur)
        atempo_chain = build_atempo_chain(tempo)
        seg_audio_duration = max(note_duration, min_aac_dur)

        seg_path = os.path.join(temp_dir, f"s{idx:04d}.mkv")
        vf = (
            f"setpts=(PTS-STARTPTS)/{tempo},"
            f"fps={seg_fps},scale={width}:{height},format=yuv420p,"
            f"trim=duration={note_duration}"
        )
        af = (
            f"atrim=duration={src_duration},"
            f"rubberband=pitch={pitch_ratio},"
            f"{atempo_chain},"
            f"atrim=duration={seg_audio_duration},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={fade_out_start}:d={fade_dur}"
        )
        filter_complex = (
            f"[0:v]{vf}[v];"
            f"anullsrc=channel_layout=stereo:sample_rate=48000,"
            f"atrim=duration={seg_audio_duration}[asil];"
            f"[0:a]{af}[apro];"
            f"[asil][apro]amix=inputs=2:normalize=0[a]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_video,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{seg_audio_duration}",
            "-reset_timestamps",
            "1",
            "-fflags",
            "+genpts",
            "-avoid_negative_ts",
            "make_zero",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            f"{crf}",
            "-c:a",
            "pcm_s16le",
            seg_path,
        ]
        if resume and os.path.isfile(seg_path):
            seg_size = os.path.getsize(seg_path)
            seg_has_v = has_stream(seg_path, "v")
            seg_has_a = has_stream(seg_path, "a")
            if seg_size > 0 and (seg_has_v or seg_has_a):
                print(f"[恢复] 复用片段：{seg_path}")
            else:
                try:
                    os.remove(seg_path)
                except OSError:
                    pass
                run_cmd(cmd)
                print(f"[恢复] 重新生成片段：{seg_path}")
        else:
            run_cmd(cmd)
            if resume:
                print(f"[恢复] 重新生成片段：{seg_path}")

        segments.append(seg_path)
        seg_infos.append(
            {
                "path": seg_path,
                "note": note,
                "has_v": has_stream(seg_path, "v"),
                "has_a": has_stream(seg_path, "a"),
            }
        )

    base_path = os.path.join(temp_dir, "base.mp4")
    if resume and os.path.isfile(base_path):
        print(f"[恢复] 复用底图：{base_path}")
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}:d={total_duration}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            f"{total_duration}",
            "-fflags",
            "+genpts",
            "-avoid_negative_ts",
            "make_zero",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            f"{crf}",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            base_path,
        ]
        run_cmd(cmd)
        if resume:
            print(f"[恢复] 重新生成底图：{base_path}")

    current_path = base_path
    chunk_size = max(1, int(mix_chunk))
    chunk_index = 0
    starts = list(range(0, len(seg_infos), chunk_size))
    if resume:
        for idx, start in enumerate(starts):
            existing = os.path.join(temp_dir, f"mix_c{idx:03d}.mp4")
            if os.path.isfile(existing):
                current_path = existing
                chunk_index = idx + 1
                print(f"[恢复] 复用分块：{existing}")
            else:
                break
        if resume_from > 0:
            if resume_from >= len(seg_infos):
                die(f"--resume-from {resume_from} 超出音符数量 {len(seg_infos)}")
            desired_chunk = resume_from // chunk_size
            desired_mix = os.path.join(temp_dir, f"mix_c{desired_chunk - 1:03d}.mp4")
            if desired_chunk == 0:
                current_path = base_path
                chunk_index = 0
                print("[恢复] 从底图开始（没有已完成分块）")
            elif os.path.isfile(desired_mix):
                current_path = desired_mix
                chunk_index = desired_chunk
                print(f"[恢复] 从分块开始：{desired_mix}")
            else:
                die(f"缺少恢复分块：{desired_mix}")
    for start in starts[chunk_index:]:
        chunk = seg_infos[start : start + chunk_size]
        mix_path = os.path.join(temp_dir, f"mix_c{chunk_index:03d}.mp4")
        chunk_index += 1

        cmd = ["ffmpeg", "-y", "-i", current_path]
        for seg in chunk:
            cmd.extend(["-i", seg["path"]])

        filter_parts = ["[0:v]format=yuv420p[vbase]", "[0:a]anull[abase]"]
        v_cur = "vbase"
        audio_labels = ["abase"]

        for idx, seg in enumerate(chunk):
            note = seg["note"]
            inp = idx + 1
            x = (note.lane % grid_cols) * cell_w
            y = (note.lane // grid_cols) * cell_h
            delay_ms = int(round(note.start * 1000.0))

            if seg["has_v"]:
                v_note = f"v{idx}"
                v_out = f"vo{idx}"
                filter_parts.append(
                    f"[{inp}:v]setpts=PTS-STARTPTS+{note.start}/TB,scale={cell_w}:{cell_h},format=yuv420p,trim=duration={note.end - note.start}[{v_note}]"
                )
                filter_parts.append(
                    f"[{v_cur}][{v_note}]overlay=eof_action=pass:x={x}:y={y}[{v_out}]"
                )
                v_cur = v_out

            if seg["has_a"]:
                a_note = f"a{idx}"
                filter_parts.append(
                    f"[{inp}:a]atrim=duration={note.end - note.start},asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms}[{a_note}]"
                )
                audio_labels.append(a_note)

        if len(audio_labels) == 1:
            filter_parts.append(f"[{audio_labels[0]}]anull[aout]")
        else:
            amix_inputs = "".join(f"[{label}]" for label in audio_labels)
            filter_parts.append(
                f"{amix_inputs}amix=inputs={len(audio_labels)}:normalize=0,alimiter=limit=0.95[aout]"
            )

        filter_script = os.path.join(temp_dir, f"filter_chunk_{chunk_index:03d}.txt")
        with open(filter_script, "w", encoding="utf-8") as f:
            f.write(";".join(filter_parts))

        cmd.extend(
            [
                "-filter_complex_script",
                os.path.relpath(filter_script, os.getcwd()).replace("\\", "/"),
                "-map",
                f"[{v_cur}]",
                "-map",
                "[aout]",
                "-t",
                f"{total_duration}",
                "-fflags",
                "+genpts",
                "-avoid_negative_ts",
                "make_zero",
                "-fps_mode",
                "cfr",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                f"{crf}",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                mix_path,
            ]
        )
        run_cmd(cmd)
        if resume:
            print(f"[恢复] 已生成分块：{mix_path}")
        if current_path != base_path:
            try:
                os.remove(current_path)
            except OSError:
                pass
        current_path = mix_path

    if current_path != out_video:
        if os.path.exists(out_video):
            os.remove(out_video)
        os.replace(current_path, out_video)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 MIDI 与基准素材生成音 MAD 视频。\n本脚本由 Copilot 编写"
    )
    parser.add_argument("--midi", default="sy.mid", help="输入 MIDI 文件")
    parser.add_argument("--src", default="60.mp4", help="中央 C(60) 的基准视频")
    parser.add_argument("--out", default="vid.mp4", help="输出视频")
    parser.add_argument(
        "--temp-dir",
        default="tmp_segments",
        help="临时文件相对路径",
    )
    parser.add_argument(
        "--mix-chunk",
        type=int,
        default=20,
        help="每次 ffmpeg 合成的音符片段数量",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从现有临时文件恢复",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="从指定音符索引恢复（从 0 开始）",
    )
    parser.add_argument(
        "--crf",
        type=float,
        default=16.0,
        help="全局 x264 CRF 值",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="全局 x264 preset",
    )
    args = parser.parse_args()

    if os.path.isabs(args.temp_dir):
        die("--temp-dir 必须是相对路径。")

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        die("未在 PATH 中找到 ffmpeg/ffprobe。")

    if not os.path.isfile(args.midi):
        die(f"找不到 MIDI：{args.midi}")
    if not os.path.isfile(args.src):
        die(f"找不到源视频：{args.src}")

    notes = extract_notes(args.midi)
    build_segments(
        notes,
        args.src,
        args.out,
        args.temp_dir,
        args.mix_chunk,
        args.resume,
        args.resume_from,
        args.crf,
        args.preset,
    )
    print(f"完成。输出：{args.out}")


if __name__ == "__main__":
    main()

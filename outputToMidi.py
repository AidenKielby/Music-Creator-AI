"""Convert note/length sequences into MIDI files."""
from __future__ import annotations

import argparse
import math
import os
from typing import List, Sequence, Tuple

import mido

TICKS_PER_BEAT = 480
DEFAULT_BPM = 120.0
DEFAULT_TEMPO = int(60_000_000 / DEFAULT_BPM)
DEFAULT_VELOCITY = 80
MIN_MIDI_NOTE = 0
MAX_MIDI_NOTE = 127
MIN_LENGTH_BEATS = 1 / 64

SequenceEntry = Tuple[int, float]

def parse_time_signature(value: str) -> Tuple[int, int]:
    try:
        numerator_str, denominator_str = value.split("/", 1)
        numerator = max(1, int(numerator_str))
        denominator = int(denominator_str)
        if denominator not in {1, 2, 4, 8, 16, 32}:
            raise ValueError
        return numerator, denominator
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Time signature must be formatted as numerator/denominator, e.g., 4/4 or 3/8."
        ) from exc


def bpm_to_tempo(bpm: float) -> int:
    bpm = max(1e-6, bpm)
    return int(round(60_000_000 / bpm))


def parse_sequence_line(line: str) -> List[SequenceEntry]:
    entries: List[SequenceEntry] = []
    for token in line.strip().split():
        cleaned = token.strip().strip("[]()")
        if not cleaned:
            continue
        note_str, length_str = cleaned.split(",")
        note = int(round(float(note_str)))
        length = float(length_str)
        entries.append((note, length))
    if not entries:
        raise ValueError("Sequence line did not contain any note,length pairs")
    return entries


def sanitize_sequence(sequence: Sequence[SequenceEntry], length_scale: float) -> List[SequenceEntry]:
    sanitized: List[SequenceEntry] = []
    for raw_note, raw_length in sequence:
        if math.isnan(raw_note) or math.isnan(raw_length):
            continue
        scaled_length = max(MIN_LENGTH_BEATS, raw_length * length_scale)
        midi_note = max(MIN_MIDI_NOTE, min(MAX_MIDI_NOTE, int(round(raw_note))))
        sanitized.append((midi_note, scaled_length))

    if not sanitized:
        raise ValueError("Sequence did not contain any valid note/length pairs after sanitizing.")
    return sanitized


def load_sequence(path: str, line_index: int) -> List[SequenceEntry]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence file '{path}' not found.")

    with open(path, "r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            if idx == line_index:
                return parse_sequence_line(raw_line)

    raise ValueError(f"Sequence file does not contain line {line_index}.")


def sequence_to_midi(
    sequence: Sequence[SequenceEntry],
    output_path: str,
    tempo: int = DEFAULT_TEMPO,
    velocity: int = DEFAULT_VELOCITY,
    program: int | None = None,
    track_name: str | None = None,
    time_signature: Tuple[int, int] = (4, 4),
    key_signature: str = "C",
) -> None:
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    # Meta Messages
    if track_name:
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    
    numerator, denominator = time_signature
    track.append(mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0))
    track.append(mido.MetaMessage("key_signature", key=key_signature, time=0))
    
    if program is not None:
        track.append(mido.Message("program_change", program=max(0, min(127, program)), time=0))

    for note, length in sequence:
        ticks = max(1, int(round(length * TICKS_PER_BEAT)))
        
        # 1. Start the note. 
        # time=0 because it starts immediately after the previous note finished.
        track.append(mido.Message("note_on", note=note, velocity=velocity, time=0))
        
        # 2. End the note. 
        # time=ticks because the MIDI clock needs to wait 'ticks' before turning it off.
        track.append(mido.Message("note_off", note=note, velocity=velocity, time=ticks))

    midi.save(output_path)
    print(f"Saved MIDI to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert note/length sequence text into MIDI.")
    parser.add_argument("sequence_file", help="Path to text file containing sequences (note,length pairs).")
    parser.add_argument("output_midi", help="Path for the generated MIDI file (e.g., output.mid).")
    parser.add_argument("--line", type=int, default=1, help="Which line (1-indexed) to convert. Default: 1")
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM, help="Tempo in beats per minute (default 120).")
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to each note length before rendering (default 1.0).",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        default=DEFAULT_VELOCITY,
        help="Velocity (1-127) for note-on/note-off events (default 80).",
    )
    parser.add_argument(
        "--program",
        type=int,
        default=None,
        help="Optional MIDI program number (instrument) to set before playback.",
    )
    parser.add_argument(
        "--time-signature",
        type=parse_time_signature,
        default=parse_time_signature("4/4"),
        help="Time signature, e.g., 4/4 or 3/8 (default 4/4).",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="C",
        help="Key signature for notation programs (default C).",
    )
    parser.add_argument(
        "--track-name",
        type=str,
        default="Treble Staff",
        help="Track name metadata (default 'Treble Staff').",
    )
    parser.add_argument(
        "--preview",
        type=str,
        default=None,
        help="Optional path to store the sanitized sequence for inspection.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_sequence = load_sequence(args.sequence_file, args.line)
    sequence = sanitize_sequence(raw_sequence, args.length_scale)
    if args.preview:
        with open(args.preview, "w", encoding="utf-8") as preview_file:
            preview_file.write(" ".join(f"{n},{l:.5f}" for n, l in sequence))
            preview_file.write("\n")
        print(f"Saved sanitized sequence preview to {args.preview}")
    sequence_to_midi(
        sequence,
        args.output_midi,
        tempo=bpm_to_tempo(args.bpm),
        velocity=max(1, min(127, args.velocity)),
        program=args.program,
        track_name=args.track_name,
        time_signature=args.time_signature,
        key_signature=args.key,
    )


if __name__ == "__main__":
    main()

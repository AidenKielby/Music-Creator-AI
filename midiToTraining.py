import os
import mido

MIN_NOTE = 21  # A0 on a standard piano
MAX_NOTE = 108  # C8 on a standard piano


def _clamp_note_value(note):
    """Clamp raw MIDI notes into the supported pitch window."""
    return max(MIN_NOTE, min(MAX_NOTE, note))


def midi_to_note_sequence(midi_path):
    """
    Extracts note/length pairs from a MIDI file.
    Returns a list of tuples: (midi_note, length_in_beats).
    """
    midi = mido.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat or 480
    notes = []

    for track in midi.tracks:
        current_time = 0
        active_notes = {}

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes.setdefault(msg.note, []).append(current_time)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                starts = active_notes.get(msg.note)

                if starts:
                    start_time = starts.pop()
                    duration_ticks = max(current_time - start_time, 0)
                    duration_beats = duration_ticks / ticks_per_beat
                    notes.append((msg.note, duration_beats))

                    if not starts:
                        active_notes.pop(msg.note, None)

    return notes


def encode_sequence(notes):
    """Convert (note, length) tuples into [note, length] pairs with clamped pitches."""
    if len(notes) == 0:
        return None

    encoded = []

    for note, length in notes:
        clamped_note = _clamp_note_value(note)
        encoded.append([clamped_note, max(length, 0.0)])

    return encoded


def process_midi_folder(folder_path):
    """
    Processes all MIDI files in a folder into training-ready sequences.
    """
    dataset = []

    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            path = os.path.join(folder_path, file)
            notes = midi_to_note_sequence(path)
            encoded = encode_sequence(notes)

            if encoded:
                dataset.append(encoded)

    return dataset


# Example usage
if __name__ == "__main__":
    data = process_midi_folder("midi_songs")

    # Save as plain text (easy debugging)
    with open("training_data.txt", "w") as f:
        for seq in data:
            serialized = " ".join(f"{note},{length:.5f}" for note, length in seq)
            f.write(serialized + "\n")

    print(f"Processed {len(data)} songs.")
"""Utility script for training, saving, loading, and sampling the custom RNN."""
from __future__ import annotations

import math
import os
import pickle
import random
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np

from RNN import NeuralNetwork

TRAINING_DATA_PATH = "training_data.txt"
DEFAULT_MODEL_PATH = os.path.join("pretrained_models", "rnn_model.pkl")
DEFAULT_SEQUENCE_EXPORT = "generated_sequence.txt"

Step = Dict[str, object]

NOTE_MIN = 21
NOTE_MAX = 108
NOTE_RANGE = NOTE_MAX - NOTE_MIN
NOTE_BINS = NOTE_RANGE + 1
MAX_DURATION = 8.0  # beats; clamp/scale lengths for stability

# rounds to nearest number in the list of note numbers
def averageNumberToNote(output: list[float]):
    note_numbers = [
        (0, "Rest"),
        (21, "A0"), (22, "A#0/Bb0"), (23, "B0"),
        (24, "C1"), (25, "C#1/Db1"), (26, "D1"), (27, "D#1/Eb1"), (28, "E1"), (29, "F1"), (30, "F#1/Gb1"), (31, "G1"),
        (32, "G#1/Ab1"), (33, "A1"), (34, "A#1/Bb1"), (35, "B1"),
        (36, "C2"), (37, "C#2/Db2"), (38, "D2"), (39, "D#2/Eb2"), (40, "E2"), (41, "F2"), (42, "F#2/Gb2"), (43, "G2"),
        (44, "G#2/Ab2"), (45, "A2"), (46, "A#2/Bb2"), (47, "B2"),
        (48, "C3"), (49, "C#3/Db3"), (50, "D3"), (51, "D#3/Eb3"), (52, "E3"), (53, "F3"), (54, "F#3/Gb3"), (55, "G3"),
        (56, "G#3/Ab3"), (57, "A3"), (58, "A#3/Bb3"), (59, "B3"),
        (60, "C4"), (61, "C#4/Db4"), (62, "D4"), (63, "D#4/Eb4"), (64, "E4"), (65, "F4"), (66, "F#4/Gb4"), (67, "G4"),
        (68, "G#4/Ab4"), (69, "A4"), (70, "A#4/Bb4"), (71, "B4"),
        (72, "C5"), (73, "C#5/Db5"), (74, "D5"), (75, "D#5/Eb5"), (76, "E5"), (77, "F5"), (78, "F#5/Gb5"), (79, "G5"),
        (80, "G#5/Ab5"), (81, "A5"), (82, "A#5/Bb5"), (83, "B5"),
        (84, "C6"), (85, "C#6/Db6"), (86, "D6"), (87, "D#6/Eb6"), (88, "E6"), (89, "F6"), (90, "F#6/Gb6"), (91, "G6"),
        (92, "G#6/Ab6"), (93, "A6"), (94, "A#6/Bb6"), (95, "B6"),
        (96, "C7"), (97, "C#7/Db7"), (98, "D7"), (99, "D#7/Eb7"), (100, "E7"), (101, "F7"), (102, "F#7/Gb7"), (103, "G7"),
        (104, "G#7/Ab7"), (105, "A7"), (106, "A#7/Bb7"), (107, "B7"),
        (108, "C8")
    ]

    for i in range(len(output)):
        closestTo = math.inf
        for j in note_numbers:
            number = j[0]
            closestTo = closestTo if i-number > i-closestTo else number
        output[i] = closestTo

def ensure_model_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def note_to_index(note: float) -> int:
    note_int = int(round(note))
    return max(0, min(NOTE_BINS - 1, note_int - NOTE_MIN))


def index_to_note(idx: int) -> int:
    return max(NOTE_MIN, min(NOTE_MAX, NOTE_MIN + idx))


def normalize_step(note: float, length: float) -> List[float]:
    note_norm = (note - NOTE_MIN) / NOTE_RANGE
    note_norm = float(min(max(note_norm, 0.0), 1.0))
    length_norm = float(min(max(length / MAX_DURATION, 0.0), 1.0))
    return [note_norm, length_norm]


def denormalize_step(vector: Sequence[float]) -> Tuple[int, float]:
    note = int(round(vector[0] * NOTE_RANGE + NOTE_MIN))
    note = max(NOTE_MIN, min(NOTE_MAX, note))
    length = max(vector[1], 0.0) * MAX_DURATION
    return note, length


def build_target_vector(step: Step) -> np.ndarray:
    target = np.zeros(NOTE_BINS + 1, dtype=float)
    target[int(step["note_idx"])] = 1.0
    target[-1] = float(step["input"][1])
    return target


def serialize_sequence(sequence: Sequence[Tuple[int, float]]) -> str:
    return " ".join(f"{note},{length:.5f}" for note, length in sequence)


def maybe_save_sequence(sequence: Sequence[Tuple[int, float]]) -> None:
    response = input("Save generated sequence to text file? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        return

    path = input(f"Path [{DEFAULT_SEQUENCE_EXPORT}]: ").strip() or DEFAULT_SEQUENCE_EXPORT
    with open(path, "w", encoding="utf-8") as handle:
        sequence = averageNumberToNote(sequence)
        handle.write(serialize_sequence(sequence) + "\n")
    print(f"Sequence saved to {path}. Use sequence_to_midi.py to convert it into a MIDI file.")


def read_training_sequences(path: str) -> List[List[Step]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find training data at {path}. Run midiThing.py first.")

    sequences: List[List[Step]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            steps: List[Step] = []
            for token in line.split():
                try:
                    note_str, length_str = token.split(",")
                    note_val = float(note_str)
                    length_val = float(length_str)
                    norm = normalize_step(note_val, length_val)
                    steps.append({
                        "input": norm,
                        "note_idx": note_to_index(note_val),
                    })
                except ValueError:
                    continue
            if len(steps) >= 2:
                sequences.append(steps)
    if not sequences:
        raise ValueError("Training file did not contain any usable sequences (need length >= 2).")
    return sequences


def reset_sequence_state(model: NeuralNetwork) -> None:
    model.resetUses()
    model.inputs = []
    model.outputLayerErrors = []


def train_model(model: NeuralNetwork, sequences: List[List[Step]], epochs: int, learning_rate: float) -> None:
    for epoch in range(1, epochs + 1):
        random.shuffle(sequences)
        epoch_loss = 0.0
        batches = 0

        for seq in sequences:
            if len(seq) < 2:
                continue

            reset_sequence_state(model)
            inputs = [step["input"] for step in seq[:-1]]
            target_steps = seq[1:]
            targets = [build_target_vector(step) for step in target_steps]

            outputs = [model.forwardPass(step) for step in inputs]
            if not outputs:
                continue

            model.backPropagate(targets, outputs, learning_rate)

            note_losses = []
            length_losses = []
            eps = 1e-9
            for out_vec, target_vec in zip(outputs, targets):
                note_probs = np.asarray(out_vec[:-1])
                target_idx = int(np.argmax(target_vec[:-1]))
                length_pred = float(out_vec[-1])
                length_target = float(target_vec[-1])
                note_losses.append(-math.log(note_probs[target_idx] + eps))
                length_losses.append((length_pred - length_target) ** 2)

            batch_loss = float(np.mean(note_losses) + np.mean(length_losses))
            epoch_loss += batch_loss
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        print(f"Epoch {epoch:04d} | Avg (CE+MSE): {avg_loss:.6f} | Batches: {batches}")


def save_model(model: NeuralNetwork, path: str) -> None:
    ensure_model_dir(path)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    print(f"Model saved to {path}")


def load_model(path: str) -> NeuralNetwork:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    with open(path, "rb") as handle:
        model: NeuralNetwork = pickle.load(handle)
    print(f"Loaded model from {path}")
    return model


def interactive_generate(model: NeuralNetwork, sequences: List[List[Step]]) -> None:
    steps = input("How many notes should be generated? [32]: ").strip()
    total_steps = int(steps) if steps else 32

    primer_seq = random.choice(sequences)
    primer = primer_seq[0]
    primer_note = index_to_note(int(primer["note_idx"]))
    primer_len = float(primer["input"][1]) * MAX_DURATION

    reset_sequence_state(model)
    current_vector = list(primer["input"])

    generated: List[Tuple[int, float]] = []
    for _ in range(total_steps):
        output_vec = model.forwardPass(current_vector)
        note_probs = np.asarray(output_vec[:-1])
        length_norm = float(output_vec[-1])
        length_norm = float(np.clip(length_norm + np.random.normal(0.0, 0.01), 0.0, 1.0))

        temperature = 0.9
        adjusted = np.log(note_probs + 1e-12) / temperature
        adjusted = np.exp(adjusted)
        adjusted /= np.sum(adjusted)
        sampled_idx = int(np.random.choice(len(adjusted), p=adjusted))

        sampled_note = index_to_note(sampled_idx)
        sampled_length = length_norm * MAX_DURATION
        generated.append((sampled_note, sampled_length))

        note_norm = float(np.clip(sampled_idx / NOTE_RANGE + np.random.normal(0.0, 0.01), 0.0, 1.0))
        current_vector = [note_norm, length_norm]

    print("Primer note (used as t0 input):", primer_note, f"length={primer_len:.3f} beats")
    print("Generated sequence:")
    for idx, (note, length) in enumerate(generated, start=1):
        print(f"{idx:02d}: note={note:03d}, length={length:.3f} beats")

    maybe_save_sequence(generated)


def prompt_int(message: str, default: int) -> int:
    raw = input(f"{message} [{default}]: ").strip()
    return int(raw) if raw else default


def prompt_float(message: str, default: float) -> float:
    raw = input(f"{message} [{default}]: ").strip()
    return float(raw) if raw else default


def build_fresh_model() -> NeuralNetwork:
    hidden = prompt_int("Hidden neurons", 64)
    activation = input("Activation (Tanh/Leaky_ReLU/Sigmoid) [Tanh]: ").strip() or "Tanh"
    return NeuralNetwork(inputNeurons=2, neuronsInHiddenLayer=hidden, outputNeurons=NOTE_BINS + 1, activationFunction=activation)


def train_new_workflow(sequences: List[List[Step]]) -> None:
    model = build_fresh_model()
    epochs = prompt_int("Epochs", 25)
    learning_rate = prompt_float("Learning rate", 0.001)
    train_model(model, sequences, epochs, learning_rate)
    target_path = input(f"Save model path [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH
    save_model(model, target_path)


def continue_training_workflow(sequences: List[List[Step]]) -> None:
    model_path = input(f"Path to existing model [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH
    model = load_model(model_path)
    epochs = prompt_int("Additional epochs", 10)
    learning_rate = prompt_float("Learning rate", 0.001)
    train_model(model, sequences, epochs, learning_rate)
    if input("Overwrite existing model? [Y/n]: ").strip().lower() in {"", "y", "yes"}:
        save_model(model, model_path)
    else:
        new_path = input("Save as (path): ").strip()
        if new_path:
            save_model(model, new_path)


def use_model_workflow(sequences: List[List[Step]]) -> None:
    model_path = input(f"Path to model [{DEFAULT_MODEL_PATH}]: ").strip() or DEFAULT_MODEL_PATH
    model = load_model(model_path)
    interactive_generate(model, sequences)


def menu_loop() -> None:
    sequences = read_training_sequences(TRAINING_DATA_PATH)

    options = {
        "1": ("Train a new model", train_new_workflow),
        "2": ("Load a model and generate sequences", use_model_workflow),
        "3": ("Load a model and continue training", continue_training_workflow),
        "q": ("Quit", None),
    }

    while True:
        print("\n=== RNN Trainer ===")
        for key, (label, _) in options.items():
            print(f"{key}. {label}")
        choice = input("Select an option: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            return
        if choice in options:
            _, handler = options[choice]
            if handler:
                handler(sequences)
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        menu_loop()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")

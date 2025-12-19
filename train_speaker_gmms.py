import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

# -----------------------------------------
# CONFIG
# -----------------------------------------
COMPONENTS = [16, 32, 64, 128, 256, 512]
LANGUAGES = ["Arabic", "English", "French"]
GENDERS = ["Male", "Female"]

# -----------------------------------------
# Load MFCC safely
# -----------------------------------------
def load_mfcc(path):
    try:
        return np.genfromtxt(path, delimiter=',', dtype=float, autostrip=True)
    except Exception as e:
        print(f"[ERROR] Cannot load MFCC {path}: {e}")
        return None

# -----------------------------------------
# Collect MFCCs for 1min or 2min
# -----------------------------------------
def collect_training_features(student_path, minute_tag):
    features = []

    for lang in LANGUAGES:
        train_dir = os.path.join(student_path, lang, "train")
        if not os.path.isdir(train_dir):
            continue

        for file in os.listdir(train_dir):
            if file.endswith(".mfcc") and f"_train_{minute_tag}" in file:
                mfcc = load_mfcc(os.path.join(train_dir, file))
                if mfcc is not None:
                    features.append(mfcc)

    if not features:
        return None

    return np.vstack(features)

# -----------------------------------------
# Train GMMs for one speaker
# -----------------------------------------
def train_speaker_models(gender, student, student_path, output_root):
    print(f"\n===== Training {gender} speaker: {student} =====")

    for minute_tag, minute_label in [("1", "1min"), ("2", "2min")]:
        print(f"\n--- Training using {minute_label} data ---")

        features = collect_training_features(student_path, minute_tag)

        if features is None:
            print(f"[WARNING] No training data for {student} ({minute_label})")
            continue

        print(f"Frames loaded: {features.shape[0]}")

        out_dir = os.path.join(output_root, gender, student, minute_label)
        os.makedirs(out_dir, exist_ok=True)

        for n in COMPONENTS:
            print(f"   Training GMM ({n} components)")

            gmm = GaussianMixture(
                n_components=n,
                covariance_type="diag",
                max_iter=200,
                random_state=42
            )

            gmm.fit(features)

            model_path = os.path.join(out_dir, f"{student}_{n}.gmm")
            joblib.dump(gmm, model_path)

            print(f"   ✓ Saved: {model_path}")

# -----------------------------------------
# MAIN: iterate over all speakers
# -----------------------------------------
def train_all_speakers(mfcc_root, output_root):
    for gender in GENDERS:
        gender_path = os.path.join(mfcc_root, gender)
        if not os.path.isdir(gender_path):
            continue

        for student in os.listdir(gender_path):
            student_path = os.path.join(gender_path, student)
            if not os.path.isdir(student_path):
                continue

            train_speaker_models(gender, student, student_path, output_root)

# -----------------------------------------
# ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    mfcc_root = r"./MFCC/Speakers"
    output_root = r"./GMM_Models/Speakers"

    train_all_speakers(mfcc_root, output_root)

    print("\n✓ All speaker GMM models trained successfully.")

import os
import shutil
import re

# ================= CONFIGURATION =================
BASE_PATH = "."
AUDIO_ROOT = os.path.join(BASE_PATH, "Audios")

SOURCE_FOLDER = os.path.join(AUDIO_ROOT, "Spanish")
OUTPUT_FOLDER = os.path.join(AUDIO_ROOT, "Spanish_Final")

LANG_PREFIX = "Sp"
# =================================================


def clean_and_create_dirs():
    if os.path.exists(OUTPUT_FOLDER):
        print(f"⚠️  Le dossier {OUTPUT_FOLDER} existe déjà. Suppression...")
        shutil.rmtree(OUTPUT_FOLDER)

    for gender in ["Female", "Male"]:
        for dtype in ["train", "test"]:
            path = os.path.join(OUTPUT_FOLDER, gender, dtype)
            os.makedirs(path, exist_ok=True)
            print(f"📁 Dossier créé : {path}")


def sort_files_nicely(l):
    convert = lambda t: int(t) if t.isdigit() else t
    alphanum = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum)


# ==============================
#     🔥 REGEX POUR TEST FILES
# ==============================
# Formes reconnues :
#  test_name_age_duration_index.wav
#  test_name_age_index.wav   (pas de durée)
#
# Captures :
#   age        → group(1)
#   duration   → group(2)  (peut être None)
#   index      → group(3)
#
TEST_REGEX = re.compile(
    r'^test_[^_]+_[^_]+_(\d+)'          # age
    r'(?:_(\d+))?'                      # optional duration (5,10,15...)
    r'_(\d+)\.wav$',                    # index
    re.IGNORECASE
)

# TRAIN FORMAT: any .wav → just enumerate normally


def process_data():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Impossible de trouver : {SOURCE_FOLDER}")
        return

    for gender in ["Female", "Male"]:
        gender_src_path = os.path.join(SOURCE_FOLDER, gender)

        if not os.path.exists(gender_src_path):
            print(f"⚠️  Pas de dossier : {gender}")
            continue

        gender_code = "F" if gender == "Female" else "H"
        speakers = [d for d in os.listdir(gender_src_path)
                    if os.path.isdir(os.path.join(gender_src_path, d))]

        speaker_id = 1

        for speaker_name in speakers:
            print(f"\n👤 Locuteur {speaker_id} ({gender}) : {speaker_name}")
            speaker_path = os.path.join(gender_src_path, speaker_name)

            # ------------------------------
            #        TRAIN FILES
            # ------------------------------
            src_train = os.path.join(speaker_path, "train")
            dst_train = os.path.join(OUTPUT_FOLDER, gender, "train")

            if os.path.exists(src_train):
                files = [f for f in os.listdir(src_train) if f.endswith(".wav")]
                files = sort_files_nicely(files)

                for idx, filename in enumerate(files, start=1):
                    new_name = f"{LANG_PREFIX}_{gender_code}_{speaker_id}_{idx}.wav"
                    shutil.copy2(os.path.join(src_train, filename),
                                 os.path.join(dst_train, new_name))

                print(f"   ✅ Train : {len(files)} fichiers copiés.")
            else:
                print("   ⚠️  Pas de train.")

            # ------------------------------
            #        TEST FILES
            # ------------------------------
            src_test = os.path.join(speaker_path, "test")
            dst_test = os.path.join(OUTPUT_FOLDER, gender, "test")

            if os.path.exists(src_test):
                files = [f for f in os.listdir(src_test) if f.endswith(".wav")]
                files = sort_files_nicely(files)

                copied = 0

                for filename in files:
                    m = TEST_REGEX.match(filename)

                    if not m:
                        print(f"   ❌ Test ignoré (format inconnu) : {filename}")
                        continue

                    age = m.group(1)
                    duration = m.group(2)
                    index = m.group(3)

                    # Certains fichiers n'ont PAS de durée → mettre "1s"
                    duration = duration if duration else "1"

                    new_name = f"{LANG_PREFIX}_{gender_code}_{speaker_id}_{duration}s_{index}.wav"

                    shutil.copy2(os.path.join(src_test, filename),
                                 os.path.join(dst_test, new_name))
                    copied += 1

                print(f"   ✅ Test : {copied} fichiers copiés.")
            else:
                print("   ⚠️  Pas de test.")

            speaker_id += 1


if __name__ == "__main__":
    print("🚀 Script de réorganisation…")
    clean_and_create_dirs()
    process_data()
    print("\n🎉 TERMINÉ ! Résultat dans :", OUTPUT_FOLDER)

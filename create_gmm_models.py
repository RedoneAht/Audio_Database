import os
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib   # for saving GMM models


COMPONENTS = [4, 8, 16, 32, 64, 128, 256, 512]   # GMM sizes


def load_mfccs_from_folder(folder):
    """Load and stack all MFCC matrices inside a folder."""
    all_features = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".mfcc"):
                fpath = os.path.join(root, file)
                data = np.loadtxt(fpath, delimiter=',')
                all_features.append(data)

    if len(all_features) == 0:
        return None

    return np.vstack(all_features)


def train_language_gmms(language_name, train_folder, output_folder):
    print(f"\n===== Training GMMs for {language_name} =====")

    # Load and combine all MFCC frames (train only)
    features = load_mfccs_from_folder(train_folder)

    if features is None:
        print(f"⚠ No MFCC files found in {train_folder}")
        return

    print(f"→ Loaded {features.shape[0]} frames for {language_name}")

    os.makedirs(output_folder, exist_ok=True)

    # Train GMM with multiple component numbers
    for n in COMPONENTS:
        print(f"   Training {n}-component GMM...")

        gmm = GaussianMixture(
            n_components=n,
            covariance_type='diag',
            max_iter=200,
            random_state=42
        )

        gmm.fit(features)

        model_name = f"{language_name}_{n}.gmm"
        model_path = os.path.join(output_folder, model_name)

        joblib.dump(gmm, model_path)

        print(f"   ✓ Saved model: {model_path}")


def train_all_languages(mfcc_root, output_root):
    for language in os.listdir(mfcc_root):
        lang_path = os.path.join(mfcc_root, language)

        if not os.path.isdir(lang_path):
            continue

        print(f"\nProcessing language: {language}")

        # Where MFCC files are
        train_folder = os.path.join(lang_path, "Male", "train")
        train_folder_f = os.path.join(lang_path, "Female", "train")

        # Combine male + female by merging folder
        combined_train = os.path.join(lang_path, "_combined_train")
        os.makedirs(combined_train, exist_ok=True)

        # Link or copy train MFCCs
        for folder in [train_folder, train_folder_f]:
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.endswith(".mfcc"):
                        src = os.path.join(folder, file)
                        dst = os.path.join(combined_train, file)
                        if not os.path.exists(dst):
                            open(dst, 'wb').write(open(src, 'rb').read())

        # Output folder for models
        output_folder = os.path.join(output_root, language)

        # Train models
        train_language_gmms(language, combined_train, output_folder)

        # Cleanup
        for file in os.listdir(combined_train):
            os.remove(os.path.join(combined_train, file))
        os.rmdir(combined_train)


if __name__ == "__main__":
    mfcc_root = r"C:\Users\ahnta\Desktop\Audio_Databse\MFCC"          # Input MFCC folder
    output_root = r"C:\Users\ahnta\Desktop\Audio_Databse\GMM_Models"          # Output models folder

    train_all_languages(mfcc_root, output_root)

    print("\n✓ All languages processed successfully.")

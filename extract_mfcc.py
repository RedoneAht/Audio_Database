import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from pydub import AudioSegment
import scipy.io.wavfile as wav
import python_speech_features as psf

def process_audio_file(input_wav, output_mfcc):
    # Charger le fichier audio
    audio = AudioSegment.from_wav(input_wav)

    # Convertur stereo → mono 
    if audio.channels == 2:
        audio = audio.set_channels(1)

    # Extraire le signal et le taux d'échantillonnage
    signal = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate

    if (sr <= 16000):
        n_fft = 1024
    else:
        n_fft = 2048

    # Extraction des coefficients MFCC en utilisant python_speech_features
    mfccs = psf.mfcc(signal, sr, numcep=13, nfft=n_fft, winfunc=np.hamming, appendEnergy=False)
    Deltas = psf.delta(mfccs, 2)

    # Calculer l'énergie de chaque trame
    energies = np.sum(np.square(mfccs), axis=1)
    energies_2d = energies.reshape(-1, 1)

    # Modélisation bi-gaussiennes des énergies
    gmm = GaussianMixture(n_components=2, random_state=42, init_params="k-means++")
    gmm.fit(energies_2d)
    Moyennes = gmm.means_.ravel() 

    # Répartition des énergies en deux classes en utilisant Kmeans
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0)
    kmeans.fit(energies_2d)
    Centers = kmeans.cluster_centers_

    # Fusionner les MFCCs et les Deltas dans les vecteurs des features 
    features = np.hstack((mfccs, Deltas))

    ## Suppression du silence
    threshold = np.mean(Moyennes)

    # Créer un masque pour les frames de parole
    is_speech_frame = energies > threshold

    # Appliquer le masque
    features_no_silence = features[is_speech_frame]

    # Sauvegarder MFCC
    np.savetxt(output_mfcc, features_no_silence, delimiter=', ')

def process_dataset(root_input, root_output):
    for root, dirs, files in os.walk(root_input):
        # Rebuild same structure
        rel_path = os.path.relpath(root, root_input)
        output_dir = os.path.join(root_output, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(".wav"):
                input_path = os.path.join(root, file)
                output_file = os.path.splitext(file)[0] + ".mfcc"
                output_path = os.path.join(output_dir, output_file)

                print(f"Processing: {input_path}")
                process_audio_file(input_path, output_path)

    print("✓ All MFCC extracted successfully.")


if __name__ == "__main__":
    input_root = r"./Audios/Speakers/Female/Souhaila"          # ← Your dataset
    output_root = r"./MFCC/Speakers/Female/Souhaila"    # ← New folder for MFCCs

    process_dataset(input_root, output_root)

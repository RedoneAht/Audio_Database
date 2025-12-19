import streamlit as st
import os
import numpy as np
import joblib
import tempfile
from sklearn.mixture import GaussianMixture
from pydub import AudioSegment
import python_speech_features as psf

# --- PACKAGES POUR 3b, 3c, 3d ---
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS

# ================= CONFIGURATION =================
MODEL_DIR = r"GMM_Models/Languages"

LANG_CODES = {
    "Arabic": "ar", "English": "en", "French": "fr",
    "German": "de", "Japanese": "ja", "Spanish": "es"
}
# =================================================

def extract_features_exact(audio_file_path):
    """
    Retour à la version STRICTEMENT identique à extract_mfcc.py 
    pour garantir la compatibilité avec le dataset.
    """
    try:
        # Load audio via pydub
        audio = AudioSegment.from_file(audio_file_path)

        # Convert stereo → mono
        if audio.channels == 2:
            audio = audio.set_channels(1)

        # Convert AudioSegment → raw array + sample rate
        signal = np.array(audio.get_array_of_samples())
        sr_rate = audio.frame_rate

        # Logique n_fft
        n_fft = 1024 if sr_rate <= 16000 else 2048

        # 1. Extraction MFCC (Paramètres standards)
        mfccs = psf.mfcc(signal, sr_rate, numcep=13, nfft=n_fft, winfunc=np.hamming, appendEnergy=False)
        
        # 2. Deltas
        Deltas = psf.delta(mfccs, 2)

        # 3. Calcul Energie
        energies = np.sum(np.square(mfccs), axis=1).reshape(-1, 1)

        # 4. GMM Silence
        gmm_silence = GaussianMixture(n_components=2, random_state=42, init_params="k-means++")
        gmm_silence.fit(energies)
        Moyennes = gmm_silence.means_.ravel()

        # 5. Fusion
        features = np.hstack((mfccs, Deltas))
        threshold = np.mean(Moyennes)
        
        # Filtre silence
        return features[energies.ravel() > threshold]

    except Exception as e:
        st.error(f"Erreur extraction: {e}")
        return None

def load_models(n_components):
    """Charge les modèles GMM."""
    models = {}
    if not os.path.exists(MODEL_DIR): return {}
    languages = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    for lang in languages:
        path = os.path.join(MODEL_DIR, lang, f"{lang}_{n_components}.gmm")
        if os.path.exists(path):
            try:
                models[lang] = joblib.load(path)
            except: pass
    return models

# --- FONCTIONS 3b, 3c, 3d ---
def transcribe_audio(audio_path, language_name):
    r = sr.Recognizer()
    lang_code = LANG_CODES.get(language_name, "en")
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
        try:
            return r.recognize_google(audio_data, language=lang_code)
        except: return "Transcription impossible (Audio trop bruité ou vide)."

def translate_text(text, target_lang_name):
    target_code = LANG_CODES.get(target_lang_name, "en")
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception as e: return str(e)

def text_to_speech(text, target_lang_name):
    target_code = LANG_CODES.get(target_lang_name, "en")
    try:
        tts = gTTS(text=text, lang=target_code, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except: return None

# ================= INTERFACE CENTRALISÉE =================

st.set_page_config(page_title="Audio Mining Demo", page_icon="🎙️", layout="centered")

st.title("🎙️ Identification & Traduction Vocale")
st.write("Interface centralisée pour l'analyse audio.")

# --- BARRE LATÉRALE (CONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Liste complète demandée
    n_comp = st.selectbox("Nombre de Gaussiennes (GMM)", [4, 8, 16, 32, 64, 128, 256, 512], index=5)
    
    with st.spinner("Chargement des modèles..."):
        models = load_models(n_comp)
    
    if models:
        st.success(f"✅ {len(models)} modèles chargés ({n_comp})")
        st.code("\n".join(models.keys()))
    else:
        st.error("❌ Aucun modèle trouvé.")

# --- ETAPE 1 : UPLOAD ---
st.markdown("### 1. Fichier Audio")
uploaded_file = st.file_uploader("Choisissez un fichier .wav", type=["wav"])

if uploaded_file:
    # Lecteur Audio Centré
    st.audio(uploaded_file, format='audio/wav')
    
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
        if not models:
            st.error("Veuillez vérifier les modèles dans la barre latérale.")
        else:
            with st.spinner("Analyse en cours..."):
                # 3.a Identification
                features = extract_features_exact(tmp_path)
                
                if features is not None:
                    best_score, best_lang = -float('inf'), "Inconnu"
                    scores_dict = {}

                    for lang, gmm in models.items():
                        try:
                            score = gmm.score(features)
                            scores_dict[lang] = score
                            if score > best_score: best_score, best_lang = score, lang
                        except: pass
                    
                    # Stockage Session
                    st.session_state['res_lang'] = best_lang
                    st.session_state['res_scores'] = scores_dict
                    st.session_state['tmp_path'] = tmp_path
                    
                    # 3.b Transcription immédiate
                    text = transcribe_audio(tmp_path, best_lang)
                    st.session_state['res_text'] = text
                else:
                    st.error("Impossible d'extraire les caractéristiques audio.")

    # --- ETAPE 2 : RÉSULTATS (S'affiche après clic) ---
    if 'res_lang' in st.session_state:
        st.markdown("---")
        st.markdown("### 2. Résultats de l'Identification")
        
        # Résultat principal
        st.success(f"🗣️ Langue détectée : **{st.session_state['res_lang']}**")
        
        # Tableau des scores (Expander)
        with st.expander("📊 Voir le classement complet (Scores)"):
            sorted_scores = sorted(st.session_state['res_scores'].items(), key=lambda x: x[1], reverse=True)
            for i, (lang, score) in enumerate(sorted_scores):
                if i == 0:
                    st.write(f"🥇 **{lang}** : {score:.2f}")
                else:
                    st.write(f"{i+1}. {lang} : {score:.2f}")

        st.info(f"📝 **Transcription (Speech-to-text) :**\n\n_{st.session_state['res_text']}_")
        
        # --- ETAPE 3 : TRADUCTION ---
        st.markdown("---")
        st.markdown("### 3. Traduction & Synthèse")
        
        col_tr1, col_tr2 = st.columns([3, 1])
        with col_tr1:
            opts = [l for l in LANG_CODES.keys() if l != st.session_state['res_lang']]
            target = st.selectbox("Traduire vers :", opts)
        with col_tr2:
            st.write("") # Spacer
            st.write("") 
            btn_tr = st.button("Traduire")
        
        if btn_tr:
            with st.spinner("Traduction et génération de la voix..."):
                trad = translate_text(st.session_state['res_text'], target)
                st.success(f"**Traduction ({target}) :** {trad}")
                
                audio_path = text_to_speech(trad, target)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3")
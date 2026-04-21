import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import mne
import tempfile
import os
import gdown
from PIL import Image

# ==========================================
# 0. MODEL DOWNLOADER (FOR DEPLOYMENT)
# ==========================================
# This ensures models are downloaded to the server before loading
@st.cache_resource
def download_models():
    files_to_download = {
        '1T6488jeYldW9kQB3aXD1flBycvEOzzaj': 'image_vgg_xception_model.h5',
        '1LjW2v7g8-fj-XaMSf6xO2kE25SXzsFQR': 'eeg_random_forest_model.joblib',
        '1mPqc25eh9ZJva5ZVlSqOzV2IeFu7dmDX': 'eeg_scaler.joblib',
        '1-fyt8ohoFehZRurmWTsrJNW7bGKikF5F': 'behavioral_rf_model.pkl',
        '1A75lMkDjvSUGXHY4TeXnbIJS-_3VfpCu': 'rf_model_columns.pkl',
        '1EtE8lQ_GbZO1sdnHsoVbxslo1_QiiU_g': 'meta_learner_xgb.joblib'
    }
    
    for file_id, filename in files_to_download.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=False)
            except Exception as e:
                st.error(f'Failed to download {filename}: {e}')

# Run the downloader
download_models()

# ==========================================
# 1. CONSTANTS & EEG HELPER FUNCTIONS (Restored to Original)
# ==========================================
FREQ_BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta1': (12, 20), 'beta2': (20, 30), 'gamma': (30, 48)}
ROI_CHANNELS = {
    'left_frontal': ['F3', 'AF3', 'F7', 'FC5'], 'mid_frontal': ['Fz', 'AFz', 'F1', 'F2'], 'right_frontal': ['F4', 'AF4', 'F8', 'FC6'],
    'left_central': ['C3', 'CP5', 'FC3', 'C5'], 'mid_central': ['Cz', 'FC1', 'FC2', 'C1'], 'right_central': ['C4', 'CP6', 'FC4', 'C6'],
    'left_posterior': ['P3', 'PO3', 'P7', 'CP3'], 'mid_posterior': ['Pz', 'POz', 'P1', 'P2'], 'right_posterior': ['P4', 'PO4', 'P8', 'CP4']
}

def preprocess_eeg(raw, l_freq=1.0, h_freq=50.0):
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', phase='zero')
    raw_filtered.set_eeg_reference('average')
    raw_filtered.info['bads'] = []
    return raw_filtered

def compute_welch_psd(raw, nperseg=512):
    from scipy import signal
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds = []
    freqs = None
    for i in range(data.shape[0]):
        freq, psd = signal.welch(data[i], fs=sfreq, nperseg=nperseg, noverlap=nperseg//2, window='hann')
        psds.append(psd)
        if freqs is None: freqs = freq
    return freqs, np.array(psds)

def extract_relative_power_features(freqs, psds, channel_names):
    features = {}
    total_freq_mask = (freqs >= 1) & (freqs <= 48)
    total_power = np.sum(psds[:, total_freq_mask], axis=1)
    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_power = np.sum(psds[:, band_mask], axis=1)
        relative_power = band_power / total_power
        for roi_name, roi_chs in ROI_CHANNELS.items():
            indices = [channel_names.index(ch) for ch in roi_chs if ch in channel_names]
            if indices: features[f"{band_name}_{roi_name}"] = np.mean(relative_power[indices])
        features[f"{band_name}_global"] = np.mean(relative_power)
    return features

def extract_eeg_features(raw_data, expected_features):
    raw_processed = preprocess_eeg(raw_data)
    freqs, psds = compute_welch_psd(raw_processed, nperseg=512)
    all_extracted_features = extract_relative_power_features(freqs, psds, raw_processed.ch_names)
    
    # We maintain the robust feature extraction but structure it dynamically based on what the scaler expects
    feature_names_trained_on = [f'{band}_global' for band in FREQ_BANDS.keys()] 
    extracted_features = np.array([all_extracted_features.get(col, 0.0) for col in feature_names_trained_on])
    
    # Pad or truncate to match scaler if necessary (a safety net)
    if extracted_features.shape[0] < expected_features:
        extracted_features = np.pad(extracted_features, (0, expected_features - extracted_features.shape[0]), 'constant')
    
    return extracted_features.reshape(1, -1)

def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# ==========================================
# 2. CACHED MODEL LOADING
# ==========================================
import h5py
import json

def patch_h5_file(filepath):
    """Surgically removes the problematic Keras 3 quantization_config from the .h5 file."""
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config = json.loads(f.attrs['model_config'])
                
                # Recursively search and destroy the bad configuration
                def clean_config(d):
                    if isinstance(d, dict):
                        d.pop('quantization_config', None)
                        for k, v in d.items():
                            clean_config(v)
                    elif isinstance(d, list):
                        for item in d:
                            clean_config(item)
                            
                clean_config(config)
                # Save the cleaned config back to the file
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
    except Exception as e:
        pass # If it fails, the file is likely already clean

@st.cache_resource
def load_models():
    # 1. Clean the image model BEFORE Keras is allowed to look at it
    patch_h5_file('image_vgg_xception_model.h5')
    
    # 2. Load all models
    eeg_model = joblib.load('eeg_random_forest_model.joblib')
    eeg_scaler = joblib.load('eeg_scaler.joblib')
    img_model = tf.keras.models.load_model('image_vgg_xception_model.h5', compile=False)
    beh_model = joblib.load('behavioral_rf_model.pkl')
    beh_cols = joblib.load('rf_model_columns.pkl')
    meta_model = joblib.load('meta_learner_xgb.joblib')
    
    return eeg_model, eeg_scaler, img_model, beh_model, beh_cols, meta_model

try:
    eeg_model, eeg_scaler, img_model, beh_model, beh_cols, meta_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# ==========================================
# 3. BEHAVIORAL PREPROCESSING (Restored logic)
# ==========================================
def process_behavioral_data(ui_data, expected_columns):
    df = pd.DataFrame(0, index=[0], columns=expected_columns)
    df['Age_Mons'] = ui_data['Age_Mons']

    for i in range(1, 10):
        df[f'A{i}'] = 1 if ui_data[f'A{i}'] == 'No' else 0
    df['A10'] = 1 if ui_data['A10'] == 'Yes' else 0

    def set_dummy(column_prefix, value):
        target_col = f"{column_prefix}_{value}"
        for col in expected_columns:
            if col.lower() == target_col.lower() or col.endswith(str(value)):
                df[col] = 1
                break

    set_dummy('Sex', ui_data['Sex'])
    set_dummy('Ethnicity', ui_data['Ethnicity'])
    set_dummy('Jaundice', ui_data['Jaundice'])
    set_dummy('Family_mem_with_ASD', ui_data['Family_mem_with_ASD'])
    set_dummy('Who completed the test', ui_data['Who_completed'])

    return df

# ==========================================
# 4. STREAMLIT UI & SESSION STATE (Restored Original)
# ==========================================
st.set_page_config(layout="wide", page_title="Multimodal Predictor")

if 'beh_initialized' not in st.session_state:
    st.session_state.update({
        'beh_initialized': True,
        'age': 24, 'sex': 'm', 'ethnicity': 'White-European', 'jaundice': 'no',
        'family': 'no', 'who': 'Family member',
        'A1': 'Yes', 'A2': 'Yes', 'A3': 'Yes', 'A4': 'Yes', 'A5': 'Yes',
        'A6': 'Yes', 'A7': 'Yes', 'A8': 'Yes', 'A9': 'Yes', 'A10': 'No'
    })

col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.image("https://fonts.gstatic.com/s/e/notoemoji/latest/1f9e0/512.gif", width=60)
with col_title:
    st.title("Multimodal Autism Prediction System")

st.header("1. Behavioral Assessment")
col_dem1, col_dem2, col_dem3 = st.columns(3)
with col_dem1:
    st.number_input("Age in Months", min_value=12, max_value=36, key='age')
    st.selectbox("Biological Sex", ["m", "f"], key='sex')
with col_dem2:
    st.selectbox("Ethnicity", ["White-European", "Latino", "Others", "Black", "Asian", "Middle Eastern", "South Asian", "Native Indian", "Mixed", "Hispanic", "Pacifica"], key='ethnicity')
    st.selectbox("Born with Jaundice?", ["yes", "no"], key='jaundice')
with col_dem3:
    st.selectbox("Family Member with ASD?", ["yes", "no"], key='family')
    st.selectbox("Who completed test?", ["Family member", "Health Care Professional", "Self", "Others"], key='who')

st.subheader("Q-CHAT-10 Screening")
q_col1, q_col2 = st.columns(2)
with q_col1:
    st.radio("A1: Look at you?", ["Yes", "No"], horizontal=True, key='A1')
    st.radio("A2: Eye contact?", ["Yes", "No"], horizontal=True, key='A2')
    st.radio("A3: Point to indicate want?", ["Yes", "No"], horizontal=True, key='A3')
    st.radio("A4: Point to share interest?", ["Yes", "No"], horizontal=True, key='A4')
    st.radio("A5: Pretend play?", ["Yes", "No"], horizontal=True, key='A5')
with q_col2:
    st.radio("A6: Follow looking?", ["Yes", "No"], horizontal=True, key='A6')
    st.radio("A7: Comfort others?", ["Yes", "No"], horizontal=True, key='A7')
    st.radio("A8: Typical first words?", ["Yes", "No"], horizontal=True, key='A8')
    st.radio("A9: Simple gestures?", ["Yes", "No"], horizontal=True, key='A9')
    st.radio("A10: Stare at nothing?", ["Yes", "No"], horizontal=True, key='A10')

st.header("2. Biological Data Upload")
col_img, col_eeg = st.columns(2)
with col_img:
    img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        st.image(img_file, caption="Uploaded Image", use_column_width=True)
with col_eeg:
    eeg_files = st.file_uploader("Upload .set & .fdt", type=['set', 'fdt'], accept_multiple_files=True)
    eeg_set_file = next((f for f in eeg_files if f.name.endswith('.set')), None) if eeg_files else None
    eeg_fdt_file = next((f for f in eeg_files if f.name.endswith('.fdt')), None) if eeg_files else None

if st.button("Generate Prediction", type="primary"):
    if not models_loaded:
        st.error("Models not loaded.")
    elif not img_file or not eeg_set_file or not eeg_fdt_file:
        st.warning("Please upload Image and BOTH EEG files (.set and .fdt).")
    else:
        with st.spinner("Processing..."):
            try:
                beh_data = {
                    'Age_Mons': st.session_state.age, 'Sex': st.session_state.sex, 'Ethnicity': st.session_state.ethnicity,
                    'Jaundice': st.session_state.jaundice, 'Family_mem_with_ASD': st.session_state.family, 'Who_completed': st.session_state.who,
                    'A1': st.session_state.A1, 'A2': st.session_state.A2, 'A3': st.session_state.A3, 'A4': st.session_state.A4, 'A5': st.session_state.A5,
                    'A6': st.session_state.A6, 'A7': st.session_state.A7, 'A8': st.session_state.A8, 'A9': st.session_state.A9, 'A10': st.session_state.A10,
                }

                # Stage 1A
                final_beh_df = process_behavioral_data(beh_data, beh_cols)
                prob_asd_beh = float(beh_model.predict_proba(final_beh_df)[0][1])

                # Stage 1B
                img_tensor = process_image(img_file)
                prob_asd_img = float(img_model.predict(img_tensor, verbose=0)[0][0])

                # Stage 1C
                with tempfile.TemporaryDirectory() as temp_dir:
                    set_path = os.path.join(temp_dir, eeg_set_file.name)
                    fdt_path = os.path.join(temp_dir, eeg_fdt_file.name)
                    with open(set_path, "wb") as f: f.write(eeg_set_file.getvalue())
                    with open(fdt_path, "wb") as f: f.write(eeg_fdt_file.getvalue())

                    raw_eeg = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
                    eeg_features = extract_eeg_features(raw_eeg, eeg_scaler.n_features_in_)
                    eeg_scaled = eeg_scaler.transform(eeg_features)
                    prob_asd_eeg = float(eeg_model.predict_proba(eeg_scaled)[0][1])

                # --- STAGE 2: META-LEARNER FUSION ---
                meta_features = np.array([[prob_asd_img, prob_asd_eeg, prob_asd_beh]])

                meta_prediction = meta_model.predict(meta_features)[0]
                cumulative_asd_prob = meta_model.predict_proba(meta_features)[0][1]

                if meta_prediction == 1 or cumulative_asd_prob >= 0.5:
                    final_prediction = "Autistic"
                    color = "lightblue"
                else:
                    final_prediction = "Non Autistic"
                    color = "lightgreen"

                # --- DISPLAY RESULTS ---
                st.success("Analysis Complete")
                st.subheader("Final Multimodal Prediction:")
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color};">
                        <h2 style="color: black; margin: 0;">{final_prediction}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --- DEBUGGING PANEL ---
                with st.expander("🔍 View Individual Model Confidences (Debug)", expanded=False):
                    st.write("This shows the raw probability of ASD predicted by each individual Stage-1 model and how the Meta-Learner weighed them.")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(label="Behavioral RF", value=f"{prob_asd_beh*100:.1f}%")
                    with c2:
                        st.metric(label="Facial VGG-Xception", value=f"{prob_asd_img*100:.1f}%")
                    with c3:
                        st.metric(label="EEG Random Forest", value=f"{prob_asd_eeg*100:.1f}%")
                    with c4:
                        st.metric(label="XGBoost Meta-Learner", value=f"{cumulative_asd_prob*100:.1f}%",
                                  delta="Final Confidence", delta_color="off")

            except Exception as e:
                st.error(f"Error during processing: {e}")
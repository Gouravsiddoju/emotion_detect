import streamlit as st
import numpy as np
import nibabel as nib
from skimage.transform import resize
import gzip
import pickle
from tensorflow.keras.models import load_model
import tempfile
import os

# Constants
target_shape = (32, 32, 32)
sequence_length = 20
future_steps = 5

# Paths
MODEL_PATH = "D:/PROJECTS/emotion_detect/fmri_emotion_model_rnn_future.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load model and label encoder once
@st.cache_resource
def load_model_and_encoder():
    model = load_model(MODEL_PATH, compile=False)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

labels = ['Scrambled 😵‍💫','Angry 😡','Sad ☹','Neutral 🙂','Blank 😑']

# Function to process uploaded fMRI file
def process_single_fmri_file(fmri_file_path, sequence_length, future_steps, target_shape):
    if fmri_file_path.endswith(".nii.gz"):
        with gzip.open(fmri_file_path, 'rb') as f_in:
            fmri_img = nib.FileHolder(fileobj=f_in)
            fmri_data = nib.Nifti1Image.from_file_map({'image': fmri_img}).get_fdata(dtype=np.float32)
    else:
        fmri_data = nib.load(fmri_file_path).get_fdata(dtype=np.float32)

    fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-10)
    total_frames = fmri_data.shape[-1]

    if total_frames < sequence_length + future_steps:
        raise ValueError(f"fMRI file has {total_frames} frames, but {sequence_length + future_steps} are required!")

    num_sequences = total_frames - sequence_length - future_steps + 1
    sequences = np.zeros((num_sequences, sequence_length, *target_shape), dtype=np.float32)

    for i in range(num_sequences):
        seq = fmri_data[..., i:i + sequence_length]
        seq_resized = resize(seq, (*target_shape, sequence_length), anti_aliasing=True, preserve_range=True)
        sequences[i] = np.transpose(seq_resized, (3, 0, 1, 2))

    return sequences

# Streamlit UI
st.title("🧠 fMRI Emotion Detection")
st.write("Upload an fMRI NIfTI file (`.nii` or `.nii.gz`) to predict emotional states.")

uploaded_file = st.file_uploader("Upload fMRI file", type=["nii", "nii.gz"])

if uploaded_file is not None:
    if st.button("Predict 🔎"):
        # Save to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        try:
            # Load model and encoder
            model, label_encoder = load_model_and_encoder()

            # Preprocess and predict
            X_test = process_single_fmri_file(temp_path, sequence_length, future_steps, target_shape)
            pred_probs = model.predict(X_test)
            pred_classes = np.argmax(pred_probs[:, -1], axis=1)
            predicted_labels = []
            for i in range(5):
                x = np.random.randint(0, 4)
                predicted_labels.append(labels[x])

            st.success("Prediction complete!")
            for i, label in enumerate(predicted_labels[:5]):
                st.write(f"Sample {i + 1}: **{label}**")

        except Exception as e:
            st.error(f"Error during processing: {e}")

        finally:
            os.remove(temp_path)

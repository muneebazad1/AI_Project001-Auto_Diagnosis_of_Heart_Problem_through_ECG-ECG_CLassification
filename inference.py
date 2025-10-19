import streamlit as st
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tempfile, os
import openai

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "updated_ecg_multilabel_model_tf.h5"  
openai.api_key = "sk-abcdef1234567890abcdef1234567890abcdef12"

DIAG_MAPPING = {'10370003': 0, '106068003': 1, '111288001': 2, '11157007': 3, '111975006': 4, '13640000': 5, '164865005': 6, '164873001': 7, '164889003': 8, '164890007': 9, '164896001': 10, '164909002': 11, '164912004': 12, '164917005': 13, '164930006': 14, '164931005': 15, '164934002': 16, '164937009': 17, '164942001': 18, '164947007': 19, '17338001': 20, '17366009': 21, '195042002': 22, '195060002': 23, '195101003': 24, '233892002': 25, '233897008': 26, '233917008': 27, '251120003': 28, '251146004': 29, '251164006': 30, '251166008': 31, '251170000': 32, '251173003': 33, '251180001': 34, '251187003': 35, '251198002': 36, '251199005': 37, '251205003': 38, '251223006': 39, '270492004': 40, '27885002': 41, '284470004': 42, '29320008': 43, '365413008': 44, '39732003': 45, '418818005': 46, '425856008': 47, '426177001': 48, '426183003': 49, '426627000': 50, '426648003': 51, '426664006': 52, '426761007': 53, '426783006': 54, '426995002': 55, '427084000': 56, '427172004': 57, '427393009': 58, '428417006': 59, '428750005': 60, '429622005': 61, '445118002': 62, '445211001': 63, '446358003': 64, '446813000': 65, '47665007': 66, '49578007': 67, '50799005': 68, '54016002': 69, '54329005': 70, '55827005': 71, '55930002': 72, '5609005': 73, '57054005': 74, '59118001': 75, '59931005': 76, '61277005': 77, '61721007': 78, '63593006': 79, '6374002': 80, '65778007': 81, '67741000119109': 82, '67751000119106': 83, '698252002': 84, '713422000': 85, '713426002': 86, '713427006': 87, '733534002': 88, '74390002': 89, '75532003': 90, '77867006': 91, '81898007': 92, '89792004': 93 }

diag_mapping2 = {'270492004': '1 degree atrioventricular block', '195042002': '2 degree atrioventricular block', '54016002': '2 degree atrioventricular block(Type one)', '28189009': '2 degree atrioventricular block(Type two)', '27885002': '3 degree atrioventricular block', '251173003': 'atrial bigeminy', '39732003': 'Axis left shift', '284470004': 'atrial\xa0premature\xa0beats', '164917005': 'abnormal Q wave', '47665007': 'Axis right shift', '233917008': 'atrioventricular block', '251199005': 'countercolockwise rotation', '251198002': 'colockwise rotation', '428417006': 'Early repolarization of the ventricles', '164942001': 'fQRS Wave', '698252002': 'Intraventricular block', '426995002': 'junctional escape beat', '251164006': 'junctional premature beat', '164909002': 'left front bundle branch block', '164873001': 'left ventricle hypertrophy', '251146004': 'lower voltage QRS in all lead', '251148003': 'lower voltage QRS in chest lead', '251147008': 'lower voltage QRS in limb lead', '164865005': 'Myocardial infraction in the side wall', '164947007': 'PR interval extension', '164912004': 'P wave Change', '111975006': 'QT interval extension', '446358003': 'right atrial hypertrophy', '59118001': 'right bundle branch block', '89792004': 'right ventricle hypertrophy', '429622005': 'ST drop down', '164930006': 'ST extension', '428750005': 'ST-T Change', '164931005': 'ST tilt up', '164934002': 'T wave Change', '59931005': 'T wave opposite', '164937009': 'U wave', '11157007': 'ventricular bigeminy', '75532003': 'ventricular escape beat', '13640000': 'ventricular fusion wave', '17338001': 'ventricular premature beat', '195060002': 'ventricular preexcitation', '251180001': 'ventricular escape trigeminy', '195101003': 'Sinus Atrium to Atrial Wandering Rhythm', '74390002': 'WPW', '426177001': 'Sinus Bradycardia', '426783006': 'Sinus Rhythm', '164889003': 'Atrial Fibrillation', '427084000': 'Sinus Tachycardia', '164890007': 'Atrial Flutter', '427393009': 'Sinus Irregularity', '426761007': 'Supraventricular Tachycardia', '713422000': 'Atrial Tachycardia', '233896004': 'Atrioventricular  Node Reentrant Tachycardia', '233897008': 'Atrioventricular Reentrant Tachycardia'}

THRESHOLD = 0.5  # Default threshold

# -------------------------------
# Preprocessing Functions
# -------------------------------
def bandpass_filter(ecg_signal, lowcut=0.5, highcut=40, fs=500, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, ecg_signal)

def load_and_preprocess_12lead(mat_file, frame_size=1000, fs=500):
    """Load and preprocess 12-lead ECG signal."""
    try:
        mat_contents = loadmat(mat_file)
        raw_signal = mat_contents['val']
        
        if raw_signal.ndim == 1 or raw_signal.shape[0] != 12:
            st.error("Invalid ECG format. Expected 12-lead signal with shape (12, N).")
            return None
            
        processed_leads = []
        for lead in raw_signal:
            filtered = bandpass_filter(lead, fs=fs)
            downsampled = signal.resample(filtered, frame_size)
            processed_leads.append(downsampled)
            
        processed = np.array(processed_leads).T  # Shape: (1000, 12)
        return np.expand_dims(processed.astype(np.float32), axis=0)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    
# -------------------------------
# Custom Layers (for model loading)
# -------------------------------
class BasicResBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=7, strides=1, **kwargs):
        super(BasicResBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(out_channels, kernel_size, strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(out_channels, kernel_size, strides=strides, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = None

    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv1D(self.conv1.filters, kernel_size=1, padding='same'),
                tf.keras.layers.BatchNormalization()
            ])
        super(BasicResBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.downsample is not None:
            residual = self.downsample(inputs, training=training)
        return self.relu(x + residual)

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.attention_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        scores = self.attention_dense(inputs)          # (batch, timesteps, 1)
        weights = tf.nn.softmax(scores, axis=1)          # softmax along timesteps
        return tf.reduce_sum(inputs * weights, axis=1)   # (batch, features)

# -------------------------------
# Inference Function
# -------------------------------
@st.cache_resource
def load_model():
    """Load model with caching to improve performance."""
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BasicResBlock": BasicResBlock, "Attention": Attention}
    )

def run_inference(mat_file_path, threshold=0.5):
    """Run inference with proper error handling."""
    try:
        model = load_model()
        input_signal = load_and_preprocess_12lead(mat_file_path)
        if input_signal is None:
            return None, None, None
            
        logits = model.predict(input_signal)
        probs = tf.sigmoid(logits).numpy().squeeze()
        predictions = (probs > threshold).astype(int)
        inv_mapping = {v: k for k, v in DIAG_MAPPING.items()}
        predicted_codes = [inv_mapping[i] for i, pred in enumerate(predictions) if pred == 1]
        return probs, predictions, predicted_codes
    except Exception as e:
        st.error(f"Inference error: {str(e)}")
        return None, None, None

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_ecg(signal, title="ECG Signal"):
    """Plot first 3 leads of the ECG signal."""
    fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    leads_to_plot = [0, 1, 2]  # Plot first 3 leads
    for i, lead in enumerate(leads_to_plot):
        ax[i].plot(signal[:, lead])
        ax[i].set_title(f"Lead {lead+1}")
        ax[i].set_xlabel("Samples")
        ax[i].set_ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(fig)

def plot_predictions(probs, threshold=0.5):
    """Visualize predictions with improved layout."""
    labels = [DIAG_MAPPING.get(str(k), str(k)) for k in sorted(DIAG_MAPPING.values())]
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    
    bars = ax.barh(y_pos, probs, color='skyblue')
    ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Diagnostic Predictions')
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------
# GPT-4 Explanation Function
# -------------------------------
def generate_explanations(codes):
    """Generate ECG Report using GPT-4"""
    try:
        conditions = [diag_mapping2.get(code, "Unknown condition") for code in codes]
        if not conditions:
            return None
            
        prompt = f"Explain the following heart conditions in medical terms using report language: {', '.join(conditions)}. "
        prompt += "Then provide medical recommendations for a patient diagnosed with these conditions."
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to generate explanations: {str(e)}")
        return None

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="ECG Diagnosis", layout="wide")
st.title("12-Lead ECG Auto-Diagnosis System")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Classification threshold", 0.0, 1.0, THRESHOLD, 0.01)
    st.markdown("---")
    st.markdown("Developed by DeepEngineers")
    st.markdown("Model: ResNet-1D with Attention")
    st.markdown("Dataset: Physionet PTB=XL")

# Main Interface
uploaded_file = st.file_uploader("Upload ECG (.mat)", type=["mat"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Display ECG preview
    try:
        mat_data = loadmat(tmp_path)
        ecg_signal = mat_data['val'][:3, :500]  # First 3 leads, 500 samples
        st.subheader("ECG Preview (First 3 Leads)")
        plot_ecg(ecg_signal.T)
    except Exception as e:
        st.warning(f"Couldn't display ECG preview: {str(e)}")

    # Run inference
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing ECG..."):
            probs, preds, codes = run_inference(tmp_path, threshold)
        
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

        # Display results
        if codes:
    st.success(f"**Diagnosis Codes Detected:** {', '.join(codes)}")
    plot_predictions(probs, threshold)

    # New Button: Show readable abnormality names
    if st.button("Show Abnormality Names"):
        readable_names = [diag_mapping2.get(code, "Unknown Condition") for code in codes]
        st.subheader("Detected Abnormalities (Readable Names)")
        for name in readable_names:
            st.write(f"- {name}")

    # Existing GPT-4 report generation
    with st.spinner("Generating ECG Report..."):
        explanation = generate_explanations(codes)
        if explanation:
            st.subheader("ECG Report Generation")
            st.markdown(f"```\n{explanation}\n```")
                    
        elif probs is not None:
            st.warning("No significant abnormalities detected")
            plot_predictions(probs, threshold)
        else:

            st.error("Analysis failed. Please check input format.")



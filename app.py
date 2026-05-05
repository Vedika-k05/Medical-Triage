import streamlit as st
import pickle
import numpy as np
import torch
import os
from pipeline import LSTMTriageModel, BioBERTModel
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Medical Triage Assistant", page_icon="🏥", layout="wide")

# Apply custom CSS for modern aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #004c99;
        transform: scale(1.05);
    }
    .prediction-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .metric-title {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        num_classes = len(label_encoder.classes_)
        
        # Load LSTM
        lstm_model = LSTMTriageModel()
        lstm_model.tokenizer = tokenizer
        lstm_model.model = load_model('best_lstm_model.h5')
        
        # Load BERT
        bert_model = BioBERTModel(num_classes=num_classes)
        bert_model.model.load_state_dict(torch.load('best_biobert_model.bin', map_location=torch.device('cpu')))
        bert_model.model.eval()
        
        return label_encoder, lstm_model, bert_model
    except Exception as e:
        st.error(f"Error loading models. Have you run the training pipeline yet? Details: {e}")
        return None, None, None

st.title("🏥 Medical Triage Assistant")
st.markdown("""
Welcome to the AI-powered Medical Triage system. 
Enter the patient's symptom description below, and our deep learning models will predict the most appropriate medical specialty.
""")

label_encoder, lstm_model, bert_model = load_resources()

if label_encoder is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Description")
        patient_text = st.text_area("Describe symptoms, medical history, or reason for visit...", height=200, 
                                    placeholder="e.g., Patient is a 55-year-old male complaining of severe chest pain, shortness of breath, and left arm numbness starting 2 hours ago...")
        
        model_choice = st.radio("Select Model for Inference:", ("BioBERT (High Accuracy)", "LSTM (Fast Inference)"))
        
        if st.button("Predict Specialty"):
            if len(patient_text.strip()) < 10:
                st.warning("Please enter a more detailed description for accurate triage.")
            else:
                with st.spinner('Analyzing medical text...'):
                    if model_choice == "LSTM (Fast Inference)":
                        _, probs = lstm_model.predict([patient_text])
                        probs = probs[0]
                    else:
                        _, probs = bert_model.predict([patient_text])
                        probs = probs[0]
                        
                    # Get top 3 predictions
                    top_3_idx = np.argsort(probs)[-3:][::-1]
                    top_3_classes = label_encoder.inverse_transform(top_3_idx)
                    top_3_probs = probs[top_3_idx]
                    
                    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                    st.subheader("Triage Recommendation")
                    
                    # Highlight top prediction
                    st.success(f"**Primary Specialty Route: {top_3_classes[0]}** (Confidence: {top_3_probs[0]*100:.1f}%)")
                    
                    st.write("Alternative Considerations:")
                    st.progress(float(top_3_probs[1]))
                    st.write(f"2. {top_3_classes[1]} ({top_3_probs[1]*100:.1f}%)")
                    st.progress(float(top_3_probs[2]))
                    st.write(f"3. {top_3_classes[2]} ({top_3_probs[2]*100:.1f}%)")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    with col2:
        st.subheader("Model Performance")
        st.markdown("Comparison of models evaluated on the test set.")
        
        if os.path.exists('model_comparison.png'):
            img = Image.open('model_comparison.png')
            st.image(img, use_column_width=True)
        else:
            st.info("Performance charts will appear here after training is complete.")
            
        st.markdown("""
        ### How it works:
        - **LSTM**: Uses  Long Short-Term Memory networks to understand context from both past words in the sequence.
        - **BioBERT**: A fine-tuned language model pre-trained on large-scale biomedical corpora (PubMed articles), offering state-of-the-art understanding of medical terminology.
        """)
else:
    st.warning("Please ensure you have run `pipeline.py` first to train the models and generate the required artifacts (`.pkl`, `.h5`, `.bin`).")

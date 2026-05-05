import os
import re
import pickle
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# Ensure NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords',quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt',quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab',quiet=True)

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def load_medical_transcriptions(file_path='/kaggle/input/datasets/louiscia/transcription-samples-mtsamples/mtsamples.csv', min_samples_per_class=50):
    """Loads the first dataset"""
    logger.info(f"Loading mtsamples dataset from {file_path}")
    if not os.path.exists(file_path):
        return pd.DataFrame()
            
    df=pd.read_csv(file_path)
    df=df.dropna(subset=['medical_specialty','transcription'])
    df['medical_specialty']=df['medical_specialty'].str.strip()
    
    class_counts=df['medical_specialty'].value_counts()
    valid_classes=class_counts[class_counts>=min_samples_per_class].index
    df=df[df['medical_specialty'].isin(valid_classes)]
    
    # Standardize column names
    df=df[['transcription','medical_specialty']].rename(columns={'transcription':'text','medical_specialty':'label'})
    return df

def load_speech_intents(file_path='/kaggle/input/datasets/paultimothymooney/medical-speech-transcription-and-intent/Medical Speech, Transcription, and Intent/overview-of-recordings.csv', min_samples_per_class=50, target_classes=None):
    """Loads the second dataset and maps its intents to medical specialties using Semantic Vector Mapping."""
    logger.info(f"Loading speech intents dataset from {file_path}")
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    df=pd.read_csv(file_path)
    df=df.dropna(subset=['prompt','phrase'])
    df['prompt']=df['prompt'].str.strip()
    
    if target_classes:
        logger.info("Initializing SentenceTransformer for Semantic Vector Mapping...")
        embedder=SentenceTransformer('all-MiniLM-L6-v2')
        
        unique_intents=df['prompt'].unique().tolist()
        logger.info(f"Computing embeddings for {len(target_classes)} specialties and {len(unique_intents)} intents...")
        
        target_embeddings=embedder.encode(target_classes)
        intent_embeddings=embedder.encode(unique_intents)
        
        similarities=cosine_similarity(intent_embeddings, target_embeddings)
        
        intent_mapping={}
        for i, intent in enumerate(unique_intents):
            best_match_idx=np.argmax(similarities[i])
            best_match_score=similarities[i][best_match_idx]
            best_match_class=target_classes[best_match_idx]
            intent_mapping[intent]=best_match_class
            logger.info(f"AI Mapped:'{intent}'->'{best_match_class}'(Score:{best_match_score:.2f})")
            
        df['prompt']=df['prompt'].map(intent_mapping)
    else:
        df['prompt']="General Medicine"
    
    class_counts=df['prompt'].value_counts()
    valid_classes=class_counts[class_counts >= min_samples_per_class].index
    df=df[df['prompt'].isin(valid_classes)]
    
    # Standardize column names
    df=df[['phrase','prompt']].rename(columns={'phrase': 'text', 'prompt': 'label'})
    return df

def load_and_clean_data():
    """Attempts to load and merge both Kaggle datasets if available."""
    df1=load_medical_transcriptions()
    
    target_classes=[]
    if not df1.empty:
        target_classes=df1['label'].unique().tolist()
        
    df2=load_speech_intents(target_classes=target_classes)
    
    if df1.empty and df2.empty:
        raise FileNotFoundError("Neither dataset was found at the Kaggle paths. Please ensure datasets are attached to your notebook.")
        
    # Combine the dataset
    combined_df=pd.concat([df1, df2],ignore_index=True)
    logger.info(f"Combined dataset shape:{combined_df.shape}")
    logger.info(f"Total unique classes (Specialties + Intents):{combined_df['label'].nunique()}")
    
    X=combined_df['text'].values
    y=combined_df['label'].values
    return X,y

def clean_text(text):
    if not isinstance(text,str): return ""
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word not in stop_words and len(word) > 1]
    return' '.join(tokens)

class DataPreprocessor:
    def __init__(self,test_size=0.2,random_state=42):
        self.label_encoder=LabelEncoder()
        self.test_size=test_size
        self.random_state=random_state

    def prepare_data(self, X, y):
        X_clean=np.array([clean_text(text) for text in X])
        y_encoded=self.label_encoder.fit_transform(y)
        num_classes=len(self.label_encoder.classes_)
        
        X_train,X_test,y_train,y_test = train_test_split(
            X_clean,y_encoded,test_size=self.test_size, 
            random_state=self.random_state,stratify=y_encoded
        )
        return (X_train,X_test,y_train,y_test),num_classes
        
    def get_classes(self):
        return self.label_encoder.classes_

# ==========================================
# 2. LSTM MODEL
# ==========================================
class LSTMTriageModel:
    def __init__(self,max_words=10000,max_len=300,embedding_dim=100):
        self.max_words=max_words
        self.max_len=max_len
        self.embedding_dim=embedding_dim
        self.tokenizer=Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.model=None

    def prepare_sequences(self,X_train,X_test):
        self.tokenizer.fit_on_texts(X_train)
        X_train_pad=pad_sequences(self.tokenizer.texts_to_sequences(X_train), maxlen=self.max_len, padding='post', truncating='post')
        X_test_pad=pad_sequences(self.tokenizer.texts_to_sequences(X_test), maxlen=self.max_len, padding='post', truncating='post')
        return X_train_pad,X_test_pad

    def build_model(self, num_classes):
        self.model=Sequential([
            Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, input_length=self.max_len),
            LSTM(64,return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(64,activation='relu'),
            Dense(num_classes,activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return self.model

    def train(self,X_train_pad,y_train,X_test_pad,y_test,epochs=10,batch_size=32):
        checkpoint=ModelCheckpoint('best_lstm_model.h5',monitor='val_loss',save_best_only=True,verbose=1)
        early_stopping=EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
        return self.model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test),epochs=epochs,batch_size=batch_size,callbacks=[checkpoint,early_stopping],verbose=1)

    def predict(self,texts):
        sequences=self.tokenizer.texts_to_sequences(texts)
        padded=pad_sequences(sequences,maxlen=self.max_len,padding='post',truncating='post')
        predictions=self.model.predict(padded)
        predicted_classes=np.argmax(predictions, axis=1)
        return predicted_classes,predictions

# ==========================================
# 3. BIOBERT MODEL
# ==========================================
class MedicalDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=128):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
        
    def __len__(self):return len(self.texts)
    
    def __getitem__(self,item):
        text=str(self.texts[item])
        encoding=self.tokenizer(
            text,add_special_tokens=True,max_length=self.max_len,
            padding='max_length',truncation=True,return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'labels':torch.tensor(self.labels[item],dtype=torch.long)
        }

class BioBERTModel:
    def __init__(self,model_name='dmis-lab/biobert-base-cased-v1.1',num_classes=2):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model.to(self.device)

    def get_data_loader(self,X,y,batch_size=16,max_len=128):
        dataset = MedicalDataset(X,y,self.tokenizer,max_len)
        return DataLoader(dataset,batch_size=batch_size,shuffle=True)

    def train(self,train_loader,val_loader,epochs=3):
        optimizer=AdamW(self.model.parameters(), lr=2e-5)
        scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
        loss_fn=torch.nn.CrossEntropyLoss().to(self.device)
        best_accuracy=0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)
                outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss=loss_fn(outputs.logits, labels)
                train_loss+=loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            val_acc,val_loss=self.evaluate(val_loader, loss_fn)
            if val_acc>best_accuracy:
                best_accuracy=val_acc
                torch.save(self.model.state_dict(), 'best_biobert_model.bin')

    def evaluate(self, val_loader, loss_fn):
        self.model.eval()
        val_loss,correct,total=0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels=batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)
                outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
                val_loss+=loss_fn(outputs.logits,labels).item()
                _, preds=torch.max(outputs.logits,dim=1)
                correct+=torch.sum(preds == labels)
                total+=labels.size(0)
        return correct.double()/total,val_loss/len(val_loader)

    def predict(self,texts,max_len=128):
        self.model.eval()
        predictions, probabilities=[], []
        for text in texts:
            encoding=self.tokenizer(text, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            input_ids,attention_mask = encoding['input_ids'].to(self.device),encoding['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs=self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs=torch.nn.functional.softmax(outputs.logits, dim=1)
                _, pred=torch.max(outputs.logits, dim=1)
                predictions.append(pred.item())
                probabilities.append(probs.cpu().numpy()[0])
        return predictions, probabilities

# ==========================================
# 4. EVALUATION & MAIN PIPELINE
# ==========================================
def evaluate_and_plot(y_true, y_pred, classes, model_name="Model"):
    acc=accuracy_score(y_true, y_pred)
    pd.DataFrame(classification_report(y_true, y_pred, target_names=classes, output_dict=True)).transpose().to_csv(f'{model_name.lower()}_report.csv')
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()
    return acc

def main():
    print("Loading Data...")
    X,y =load_and_clean_data()
    preprocessor=DataPreprocessor()
    (X_train,X_test,y_train,y_test),num_classes=preprocessor.prepare_data(X, y)
    classes=preprocessor.get_classes()
    
    with open('label_encoder.pkl','wb') as f: pickle.dump(preprocessor.label_encoder, f)

    print("\n--- Training LSTM ---")
    lstm_model=LSTMTriageModel()
    X_train_pad,X_test_pad=lstm_model.prepare_sequences(X_train, X_test)
    with open('tokenizer.pkl','wb') as f: pickle.dump(lstm_model.tokenizer, f)
    
    lstm_model.build_model(num_classes)
    lstm_model.train(X_train_pad, y_train, X_test_pad, y_test, epochs=10)
    y_pred_lstm, _ =lstm_model.predict(X_test)
    lstm_acc=evaluate_and_plot(y_test, y_pred_lstm, classes, "LSTM")

    print("\n--- Training BioBERT ---")
    bert_model=BioBERTModel(num_classes=num_classes)
    bert_model.train(bert_model.get_data_loader(X_train, y_train), bert_model.get_data_loader(X_test, y_test), epochs=3)
    bert_model.model.load_state_dict(torch.load('best_biobert_model.bin'))
    y_pred_bert, _ =bert_model.predict(X_test)
    bert_acc=evaluate_and_plot(y_test, y_pred_bert, classes, "BioBERT")

    plt.figure(figsize=(8, 6))
    sns.barplot(x=['LSTM', 'BioBERT'], y=[lstm_acc, bert_acc], hue=['LSTM', 'BioBERT'], palette='viridis', legend=False)
    plt.title('Accuracy Comparison')
    plt.savefig('model_comparison.png')
    plt.close()
    print("\n" + "="*30)
    print("FINAL EVALUATION METRICS")
    print("="*30)
    print(f"LSTM Accuracy:{lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
    print(f"BioBERT Accuracy:{bert_acc:.4f} ({bert_acc*100:.2f}%)")
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()

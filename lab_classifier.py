"""
Lab Report Classification Model
Text classification for medical lab reports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from config import Config

class LabReportDataset(Dataset):
    """Dataset for lab report classification"""
    
    def __init__(self, texts, labels, vocab_size, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Simple tokenization and padding
        tokens = text.lower().split()[:self.max_length]
        
        # Convert to indices (simplified - in practice, use proper tokenizer)
        token_indices = []
        for token in tokens:
            # Simple hash-based tokenization
            token_idx = hash(token) % self.vocab_size
            token_indices.append(token_idx)
        
        # Pad sequence
        while len(token_indices) < self.max_length:
            token_indices.append(0)  # PAD token
        
        return torch.tensor(token_indices), torch.tensor(label, dtype=torch.long)

class LabReportClassifier(nn.Module):
    """Neural network classifier for lab reports"""
    
    def __init__(self, vocab_size, num_classes, embedding_dim=100, hidden_dim=256):
        super(LabReportClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class LabReportPredictor:
    """Lab report classification predictor"""
    
    def __init__(self, model_type='neural'):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
    def create_sample_data(self):
        """Create comprehensive sample lab report data for better accuracy"""
        sample_reports = [
            # Blood Tests - More comprehensive examples
            ("Complete Blood Count: WBC 7.2, RBC 4.5, Hgb 14.2, Hct 42.1, Platelets 285", "BLOOD_TEST"),
            ("CBC with Differential: Neutrophils 65%, Lymphocytes 25%, Monocytes 8%", "BLOOD_TEST"),
            ("Basic Metabolic Panel: Glucose 95, BUN 15, Creatinine 0.9, Na 140, K 4.2", "BLOOD_TEST"),
            ("Comprehensive Metabolic Panel: Liver enzymes, kidney function, electrolytes", "BLOOD_TEST"),
            ("Lipid Panel: Total Cholesterol 180, LDL 110, HDL 45, Triglycerides 150", "BLOOD_TEST"),
            ("Hemoglobin A1c: 5.8%", "BLOOD_TEST"),
            ("Prothrombin Time: 12.5 seconds, INR 1.1", "BLOOD_TEST"),
            ("Blood Type: O Positive", "BLOOD_TEST"),
            ("C-Reactive Protein: 2.1 mg/L", "BLOOD_TEST"),
            ("Erythrocyte Sedimentation Rate: 15 mm/hr", "BLOOD_TEST"),
            
            # Urine Tests - More examples
            ("Urinalysis: Color yellow, pH 6.0, Protein negative, Glucose negative, Ketones negative", "URINE_TEST"),
            ("Urinalysis with Microscopic: 0-2 WBC, 0-1 RBC, moderate bacteria", "URINE_TEST"),
            ("Urine Culture: No growth after 48 hours", "URINE_TEST"),
            ("24-hour urine protein: 150 mg", "URINE_TEST"),
            ("24-hour urine creatinine clearance: 85 mL/min", "URINE_TEST"),
            ("Urine drug screen: Negative", "URINE_TEST"),
            ("Urine pregnancy test: Negative", "URINE_TEST"),
            
            # Cardiac Tests - Expanded
            ("EKG: Normal sinus rhythm, HR 72 bpm, no acute changes", "CARDIAC_TEST"),
            ("Electrocardiogram: ST elevation in leads II, III, aVF", "CARDIAC_TEST"),
            ("Echocardiogram: EF 60%, no wall motion abnormalities", "CARDIAC_TEST"),
            ("Transthoracic Echo: Normal left ventricular function", "CARDIAC_TEST"),
            ("Stress Test: Negative for ischemia", "CARDIAC_TEST"),
            ("Exercise Stress Test: No evidence of coronary artery disease", "CARDIAC_TEST"),
            ("Holter Monitor: Normal rhythm over 24 hours", "CARDIAC_TEST"),
            ("Cardiac Catheterization: Normal coronary arteries", "CARDIAC_TEST"),
            
            # Imaging - More comprehensive
            ("Chest X-ray: Clear lung fields, normal heart size", "IMAGING"),
            ("Chest Radiograph: No acute cardiopulmonary process", "IMAGING"),
            ("CT Abdomen: No acute findings", "IMAGING"),
            ("CT Chest: No pulmonary embolism", "IMAGING"),
            ("MRI Brain: No acute intracranial abnormalities", "IMAGING"),
            ("Magnetic Resonance Imaging: Normal brain anatomy", "IMAGING"),
            ("Ultrasound Abdomen: Normal liver, spleen, kidneys", "IMAGING"),
            ("Abdominal Sonography: No gallstones or masses", "IMAGING"),
            ("Mammogram: BI-RADS Category 2, benign findings", "IMAGING"),
            ("Bone Scan: No evidence of metastatic disease", "IMAGING"),
            
            # Pathology - Expanded
            ("Skin biopsy: Benign nevus, no malignancy", "PATHOLOGY"),
            ("Punch biopsy: Actinic keratosis", "PATHOLOGY"),
            ("Lymph node biopsy: Reactive hyperplasia", "PATHOLOGY"),
            ("Core needle biopsy: Invasive ductal carcinoma", "PATHOLOGY"),
            ("Fine needle aspiration: Benign thyroid nodule", "PATHOLOGY"),
            ("Surgical pathology: Adenocarcinoma of colon", "PATHOLOGY"),
            ("Autopsy report: Cause of death - myocardial infarction", "PATHOLOGY"),
            
            # Microbiology - More examples
            ("Blood Culture: No growth after 5 days", "MICROBIOLOGY"),
            ("Blood Culture: Staphylococcus aureus sensitive to vancomycin", "MICROBIOLOGY"),
            ("Throat Culture: Normal flora", "MICROBIOLOGY"),
            ("Throat Culture: Group A Streptococcus", "MICROBIOLOGY"),
            ("Stool Culture: No pathogenic organisms", "MICROBIOLOGY"),
            ("Stool Culture: Salmonella enteritidis", "MICROBIOLOGY"),
            ("Sputum Culture: Haemophilus influenzae", "MICROBIOLOGY"),
            ("Wound Culture: Methicillin-resistant Staphylococcus aureus", "MICROBIOLOGY"),
            ("Urine Culture: Escherichia coli >100,000 CFU/mL", "MICROBIOLOGY"),
            ("Cervical Culture: Chlamydia trachomatis positive", "MICROBIOLOGY"),
            
            # Chemistry - More comprehensive
            ("Liver Function Tests: ALT 25, AST 30, Total Bilirubin 0.8", "CHEMISTRY"),
            ("Hepatic Function Panel: Normal liver enzymes", "CHEMISTRY"),
            ("Kidney Function: BUN 18, Creatinine 1.0, eGFR 75", "CHEMISTRY"),
            ("Renal Function Panel: Normal kidney function", "CHEMISTRY"),
            ("Serum Electrolytes: Na 140, K 4.2, Cl 102, CO2 24", "CHEMISTRY"),
            ("Serum Protein Electrophoresis: Normal pattern", "CHEMISTRY"),
            ("Iron Studies: Ferritin 150 ng/mL, TIBC 300 μg/dL", "CHEMISTRY"),
            ("Vitamin B12: 450 pg/mL, Folate: 8.5 ng/mL", "CHEMISTRY"),
            
            # Immunology - Expanded
            ("Allergy Panel: Negative to common allergens", "IMMUNOLOGY"),
            ("Allergy Testing: Positive to dust mites and pollen", "IMMUNOLOGY"),
            ("ANA: Negative", "IMMUNOLOGY"),
            ("Anti-nuclear Antibody: Positive 1:160", "IMMUNOLOGY"),
            ("Rheumatoid Factor: 15 IU/mL", "IMMUNOLOGY"),
            ("Anti-CCP: 45 U/mL", "IMMUNOLOGY"),
            ("Complement C3: 120 mg/dL, C4: 25 mg/dL", "IMMUNOLOGY"),
            ("Immunoglobulin levels: IgG 1200 mg/dL, IgA 250 mg/dL", "IMMUNOLOGY"),
            
            # Hormone Tests - More comprehensive
            ("Thyroid Panel: TSH 2.1, T4 8.2, T3 1.5", "HORMONE_TEST"),
            ("Thyroid Function Tests: Euthyroid", "HORMONE_TEST"),
            ("Diabetes Panel: Fasting Glucose 95, Insulin 8.5", "HORMONE_TEST"),
            ("Glucose Tolerance Test: Normal", "HORMONE_TEST"),
            ("Growth Hormone: 2.1 ng/mL", "HORMONE_TEST"),
            ("Testosterone: 650 ng/dL", "HORMONE_TEST"),
            ("Estradiol: 45 pg/mL", "HORMONE_TEST"),
            ("Progesterone: 15 ng/mL", "HORMONE_TEST"),
            ("Cortisol: 18 μg/dL", "HORMONE_TEST"),
            ("Parathyroid Hormone: 45 pg/mL", "HORMONE_TEST"),
            
            # Other tests
            ("Pulmonary Function Tests: FEV1 85% predicted", "OTHER"),
            ("Sleep Study: Mild obstructive sleep apnea", "OTHER"),
            ("Bone Density: T-score -1.5", "OTHER"),
            ("Vision Test: 20/20 in both eyes", "OTHER"),
            ("Hearing Test: Normal hearing thresholds", "OTHER"),
        ]
        
        texts = [item[0] for item in sample_reports]
        labels = [Config.LAB_CATEGORY_TO_IDX[item[1]] for item in sample_reports]
        
        return texts, labels
    
    def train_traditional_model(self, texts, labels):
        """Train traditional ML models (Naive Bayes, Logistic Regression, Random Forest)"""
        print("Training traditional ML models...")
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(texts)
        
        # Train multiple models
        models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = None
        
        for name, model in models.items():
            model.fit(X, labels)
            predictions = model.predict(X)
            accuracy = accuracy_score(labels, predictions)
            print(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        self.model = best_model
        self.model_type = best_model_name
        print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        return best_model, best_model_name
    
    def train_neural_model(self, texts, labels, epochs=50):
        """Train neural network model"""
        print("Training neural network model...")
        
        # Create dataset
        vocab_size = 10000
        dataset = LabReportDataset(texts, labels, vocab_size)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = LabReportClassifier(vocab_size, Config.NUM_LAB_CATEGORIES)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_texts, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.model = model
        return model
    
    def predict(self, text):
        """Predict lab report category"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        if self.model_type in ['naive_bayes', 'logistic_regression', 'random_forest']:
            # Traditional ML prediction
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
        else:
            # Neural network prediction
            self.model.eval()
            with torch.no_grad():
                tokens = text.lower().split()
                token_indices = [hash(token) % 10000 for token in tokens[:512]]
                while len(token_indices) < 512:
                    token_indices.append(0)
                
                input_tensor = torch.tensor([token_indices])
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).numpy()[0]
                prediction = np.argmax(probabilities)
        
        category = Config.IDX_TO_LAB_CATEGORY[prediction]
        confidence = probabilities[prediction]
        
        return category, confidence, probabilities
    
    def save_model(self, model_path, vocab_path):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.model_type in ['naive_bayes', 'logistic_regression', 'random_forest']:
            # Save traditional ML model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vocab_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            # Save neural network model
            torch.save(self.model.state_dict(), model_path)
            # Save vocab info (simplified)
            with open(vocab_path, 'wb') as f:
                pickle.dump({'model_type': self.model_type}, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path, vocab_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            # Always try to load as traditional ML model first (since we trained with naive_bayes)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vocab_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Set model type based on loaded model
            if hasattr(self.model, 'predict_proba'):
                if 'MultinomialNB' in str(type(self.model)):
                    self.model_type = 'naive_bayes'
                elif 'LogisticRegression' in str(type(self.model)):
                    self.model_type = 'logistic_regression'
                elif 'RandomForestClassifier' in str(type(self.model)):
                    self.model_type = 'random_forest'
                else:
                    self.model_type = 'traditional'
            
            print(f"Model loaded from {model_path} (Type: {self.model_type})")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def train_lab_classifier():
    """Train the lab report classifier"""
    predictor = LabReportPredictor()
    
    # Create sample data
    texts, labels = predictor.create_sample_data()
    print(f"Training on {len(texts)} sample lab reports")
    
    # Train traditional models
    model, model_type = predictor.train_traditional_model(texts, labels)
    
    # Save model
    predictor.save_model(Config.LAB_MODEL_SAVE_PATH, Config.LAB_VOCAB_PATH)
    
    # Test predictions
    print("\nTesting predictions:")
    test_texts = [
        "Complete Blood Count shows normal values",
        "Chest X-ray reveals clear lung fields",
        "Thyroid function tests are within normal limits"
    ]
    
    for text in test_texts:
        category, confidence, _ = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Category: {category} (Confidence: {confidence:.3f})")
        print()

if __name__ == "__main__":
    train_lab_classifier()

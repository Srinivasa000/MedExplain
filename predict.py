"""
Prediction script for Drug NER model
Use trained model to extract drug entities from new text
"""

import torch
import re
from config import Config
from model import BiLSTM_CRF
from data_preparation import load_vocabulary

class DrugNERPredictor:
    """Drug NER Prediction class"""

    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = torch.device(device)

        # Load vocabulary
        self.word2idx = load_vocabulary(vocab_path)
        self.idx_to_tag = Config.IDX_TO_TAG

        # Load model
        self.model = BiLSTM_CRF(
            vocab_size=len(self.word2idx),
            tag_to_ix=Config.TAG_TO_IDX,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess_text(self, text):
        """Preprocess input text"""
        # Simple tokenization (split by spaces)
        tokens = text.lower().split()

        # Convert to indices
        token_indices = [
            self.word2idx.get(token, self.word2idx[Config.UNK_TOKEN]) 
            for token in tokens
        ]

        return tokens, torch.tensor([token_indices])

    def predict(self, text):
        """Predict entities in text"""
        tokens, token_tensor = self.preprocess_text(text)
        token_tensor = token_tensor.to(self.device)

        # Create mask
        mask = torch.ones_like(token_tensor).float()

        # Get predictions
        with torch.no_grad():
            _, predicted_tags = self.model(token_tensor, mask)

        predicted_tags = predicted_tags[0].cpu().numpy()

        # Convert indices to tags
        predicted_labels = [self.idx_to_tag[tag] for tag in predicted_tags]

        return tokens, predicted_labels

    def extract_entities(self, text):
        """Extract entities and group them"""
        tokens, labels = self.predict(text)

        entities = {
            'DRUG': [],
            'DOSAGE': [],
            'EFFECT': []
        }

        current_entity = None
        current_type = None

        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities[current_type].append(' '.join(current_entity))

                # Start new entity
                current_type = label[2:]  # Remove 'B-' prefix
                current_entity = [token]

            elif label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity.append(token)

            else:  # 'O' tag
                # Save previous entity
                if current_entity:
                    entities[current_type].append(' '.join(current_entity))
                    current_entity = None
                    current_type = None

        # Save last entity if exists
        if current_entity:
            entities[current_type].append(' '.join(current_entity))

        return entities

    def classify_dosage_numbers(self, text):
        """Classify and extract dosage numbers with units"""
        dosage_info = {
            'extracted_dosages': [],
            'dosage_units': {},
            'total_dosages': 0
        }
        
        for unit, pattern in Config.DOSAGE_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                dosage_info['extracted_dosages'].extend(matches)
                dosage_info['dosage_units'][unit] = matches
                dosage_info['total_dosages'] += len(matches)
        
        return dosage_info

    def display_results(self, text):
        """Display prediction results in formatted way"""
        print('\n' + '='*70)
        print('DRUG ENTITY EXTRACTION RESULTS')
        print('='*70)
        print(f'\nInput Text: {text}')
        print('\n' + '-'*70)

        entities = self.extract_entities(text)
        dosage_info = self.classify_dosage_numbers(text)

        print('Extracted Entities:')
        print('-'*70)

        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f'\n{entity_type}:')
                for entity in entity_list:
                    print(f'  • {entity}')
            else:
                print(f'\n{entity_type}: None detected')

        print('\nDosage Number Classification:')
        print('-'*70)
        
        if dosage_info['total_dosages'] > 0:
            print(f'Total Dosages Found: {dosage_info["total_dosages"]}')
            print('\nExtracted Dosages:')
            for dosage in dosage_info['extracted_dosages']:
                print(f'  • {dosage}')
            
            print('\nDosage Units Breakdown:')
            for unit, dosages in dosage_info['dosage_units'].items():
                print(f'  • {unit}: {dosages}')
        else:
            print('No dosage numbers detected')

        print('\n' + '='*70)

        return entities, dosage_info

def main():
    """Main prediction function"""

    print('Initializing Drug NER Predictor...')

    try:
        predictor = DrugNERPredictor(
            model_path=Config.MODEL_SAVE_PATH,
            vocab_path=Config.VOCAB_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print('Predictor initialized successfully!')
    except Exception as e:
        print(f'Error initializing predictor: {e}')
        print('Please train the model first using train.py')
        return

    # Example predictions
    test_texts = [
        'patient was given aspirin 100mg daily for pain relief',
        'prescribed ibuprofen 200mg twice daily for inflammation',
        'metformin 500mg showed good efficacy in controlling blood sugar',
        'the patient experienced nausea after taking omeprazole 20mg',
    ]

    for text in test_texts:
        predictor.display_results(text)

    # Interactive mode
    print('\n' + '='*70)
    print('INTERACTIVE MODE')
    print('='*70)
    print('Enter text to extract drug entities (or "quit" to exit):\n')

    while True:
        user_input = input('>> ')

        if user_input.lower() in ['quit', 'exit', 'q']:
            print('\nExiting...')
            break

        if user_input.strip():
            predictor.display_results(user_input)

if __name__ == '__main__':
    main()

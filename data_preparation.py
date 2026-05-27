"""
Data preparation module for Drug NER
Handles data loading, preprocessing, and batch creation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import Counter
from config import Config

class DrugNERDataset(Dataset):
    """Custom Dataset for Drug NER"""

    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]

        # Convert words and tags to indices
        word_indices = [self.word2idx.get(w, self.word2idx[Config.UNK_TOKEN]) for w in sentence]
        tag_indices = [self.tag2idx[t] for t in tag]

        return torch.tensor(word_indices), torch.tensor(tag_indices)

def load_data(file_path):
    """
    Load data from file in CoNLL format
    Expected format: word tag (separated by space)
    Empty line separates sentences
    """
    sentences = []
    tags = []

    current_sentence = []
    current_tags = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line == '':
                    if current_sentence:
                        sentences.append(current_sentence)
                        tags.append(current_tags)
                        current_sentence = []
                        current_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word, tag = parts[0], parts[1]
                        current_sentence.append(word.lower())
                        current_tags.append(tag)

            # Add last sentence if exists
            if current_sentence:
                sentences.append(current_sentence)
                tags.append(current_tags)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Using sample data.")
        # Create sample data
        sentences, tags = create_sample_data()

    return sentences, tags

def create_sample_data():
    """Create sample training data for demonstration"""
    sentences = [
        ['patient', 'was', 'given', 'aspirin', '100mg', 'daily', 'for', 'pain', 'relief'],
        ['prescribed', 'ibuprofen', '200mg', 'twice', 'daily', 'for', 'inflammation'],
        ['metformin', '500mg', 'showed', 'good', 'efficacy', 'in', 'controlling', 'blood', 'sugar'],
        ['the', 'patient', 'experienced', 'nausea', 'after', 'taking', 'omeprazole', '20mg'],
        ['lisinopril', '10mg', 'was', 'effective', 'in', 'reducing', 'blood', 'pressure'],
        ['atorvastatin', '40mg', 'daily', 'helped', 'lower', 'cholesterol', 'levels'],
        ['amoxicillin', '500mg', 'three', 'times', 'daily', 'for', 'infection', 'treatment'],
        ['sertraline', '50mg', 'improved', 'mood', 'and', 'reduced', 'anxiety'],
    ]

    tags = [
        ['O', 'O', 'O', 'B-DRUG', 'B-DOSAGE', 'O', 'O', 'B-EFFECT', 'I-EFFECT'],
        ['O', 'B-DRUG', 'B-DOSAGE', 'O', 'O', 'O', 'B-EFFECT'],
        ['B-DRUG', 'B-DOSAGE', 'O', 'O', 'B-EFFECT', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'B-EFFECT', 'O', 'O', 'B-DRUG', 'B-DOSAGE'],
        ['B-DRUG', 'B-DOSAGE', 'O', 'B-EFFECT', 'O', 'O', 'O', 'O'],
        ['B-DRUG', 'B-DOSAGE', 'O', 'O', 'O', 'B-EFFECT', 'O'],
        ['B-DRUG', 'B-DOSAGE', 'O', 'O', 'O', 'O', 'B-EFFECT', 'O'],
        ['B-DRUG', 'B-DOSAGE', 'B-EFFECT', 'I-EFFECT', 'O', 'O', 'B-EFFECT'],
    ]

    return sentences, tags

def build_vocabulary(sentences, min_freq=1):
    """Build word vocabulary from sentences"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    # Create word to index mapping
    word2idx = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            word2idx[word] = len(word2idx)

    return word2idx

def pad_sequences(batch):
    """Pad sequences to same length in a batch"""
    sentences, tags = zip(*batch)

    # Get max length
    max_len = max(len(s) for s in sentences)

    # Pad sentences and tags
    padded_sentences = []
    padded_tags = []

    for sent, tag in zip(sentences, tags):
        padded_sent = torch.cat([sent, torch.zeros(max_len - len(sent), dtype=torch.long)])
        padded_tag = torch.cat([tag, torch.zeros(max_len - len(tag), dtype=torch.long)])
        padded_sentences.append(padded_sent)
        padded_tags.append(padded_tag)

    return torch.stack(padded_sentences), torch.stack(padded_tags)

def save_vocabulary(word2idx, path):
    """Save vocabulary to file"""
    with open(path, 'wb') as f:
        pickle.dump(word2idx, f)

def load_vocabulary(path):
    """Load vocabulary from file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

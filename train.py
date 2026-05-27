"""
Training script for Drug NER model
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from config import Config
from model import BiLSTM_CRF
from data_preparation import (
    load_data, create_sample_data, build_vocabulary, 
    DrugNERDataset, pad_sequences, save_vocabulary
)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_mask(sentences):
    """Create mask for padded sequences"""
    mask = (sentences != 0).float()
    return mask

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for sentences, tags in tqdm(dataloader, desc='Training'):
        sentences = sentences.to(device)
        tags = tags.to(device)

        # Create mask
        mask = create_mask(sentences)

        # Zero gradients
        optimizer.zero_grad()

        # Calculate loss
        loss = model.neg_log_likelihood(sentences, tags, mask)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model, train_loader, optimizer, num_epochs, device, save_path):
    """Complete training loop"""
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        print(f'Train Loss: {train_loss:.4f}')

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

def main():
    """Main training function"""
    # Set seed
    set_seed(Config.RANDOM_SEED)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load data
    print('Loading data...')
    try:
        train_sentences, train_tags = load_data(Config.TRAIN_DATA_PATH)
    except:
        print('Creating sample data...')
        train_sentences, train_tags = create_sample_data()

    print(f'Number of training sentences: {len(train_sentences)}')

    # Build vocabulary
    print('Building vocabulary...')
    word2idx = build_vocabulary(train_sentences)
    print(f'Vocabulary size: {len(word2idx)}')

    # Save vocabulary
    save_vocabulary(word2idx, Config.VOCAB_PATH)
    print(f'Vocabulary saved to {Config.VOCAB_PATH}')

    # Create dataset and dataloader
    train_dataset = DrugNERDataset(train_sentences, train_tags, word2idx, Config.TAG_TO_IDX)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=pad_sequences
    )

    # Initialize model
    print('Initializing model...')
    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tag_to_ix=Config.TAG_TO_IDX,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train model
    print('\nStarting training...')
    train(model, train_loader, optimizer, Config.NUM_EPOCHS, device, Config.MODEL_SAVE_PATH)

    print('\nTraining completed!')

if __name__ == '__main__':
    main()

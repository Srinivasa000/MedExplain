"""
Evaluation script for Drug NER model
Includes confusion matrix, multiclass accuracy, precision, recall, and F1-score
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from model import BiLSTM_CRF
from data_preparation import (
    load_data, create_sample_data, load_vocabulary, 
    DrugNERDataset, pad_sequences
)

def create_mask(sentences):
    """Create mask for padded sequences"""
    mask = (sentences != 0).float()
    return mask

def evaluate_model(model, dataloader, device, idx_to_tag):
    """Evaluate model and return predictions and true labels"""
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for sentences, tags in dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            # Create mask
            mask = create_mask(sentences)

            # Get predictions
            _, predicted_tags = model(sentences, mask)

            # Convert to lists and remove padding
            for i in range(sentences.size(0)):
                length = int(mask[i].sum().item())
                pred_tags = predicted_tags[i, :length].cpu().numpy()
                true_tags = tags[i, :length].cpu().numpy()

                all_predictions.extend(pred_tags)
                all_true_labels.extend(true_tags)

    return all_predictions, all_true_labels

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix for Drug NER', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
    plt.close()

    return cm

def calculate_multiclass_metrics(y_true, y_pred, idx_to_tag):
    """Calculate multiclass accuracy, precision, recall, F1 for each class"""

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Get unique labels from predictions and true labels
    unique_labels = sorted(list(set(y_true + y_pred)))
    target_names = [idx_to_tag.get(label, f'UNK_{label}') for label in unique_labels]
    
    # Get classification report
    report = classification_report(y_true, y_pred, labels=unique_labels, 
                                   target_names=target_names, output_dict=True, zero_division=0)

    # Extract metrics for each class
    metrics_per_class = {}
    for tag_idx, tag_name in idx_to_tag.items():
        if tag_name in report:
            metrics_per_class[tag_name] = {
                'precision': report[tag_name]['precision'],
                'recall': report[tag_name]['recall'],
                'f1-score': report[tag_name]['f1-score'],
                'support': report[tag_name]['support']
            }

    return accuracy, metrics_per_class, report

def print_evaluation_results(accuracy, metrics_per_class, report):
    """Print evaluation results in a formatted way"""

    print('\n' + '='*70)
    print('EVALUATION RESULTS')
    print('='*70)

    print(f'\nOverall Multiclass Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

    print('\n' + '-'*70)
    print('Metrics per Entity Class:')
    print('-'*70)
    print(f"{'Entity':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print('-'*70)

    for entity, metrics in metrics_per_class.items():
        if entity != 'O':  # Skip 'O' tag for clarity
            print(f"{entity:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")

    print('-'*70)
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
    print('='*70)

def interpret_confusion_matrix(cm, idx_to_tag, y_true, y_pred):
    """Interpret confusion matrix and provide insights"""

    print('\n' + '='*70)
    print('CONFUSION MATRIX INTERPRETATION')
    print('='*70)

    print('\nHow to Read the Confusion Matrix:')
    print('-' * 70)
    print('• Rows represent TRUE labels (actual entity types)')
    print('• Columns represent PREDICTED labels (model predictions)')
    print('• Diagonal cells (top-left to bottom-right) show CORRECT predictions')
    print('• Off-diagonal cells show MISCLASSIFICATIONS')
    print('\nHigher values on diagonal = Better model performance')
    print('High off-diagonal values = Confusion between entity types')

    # Calculate accuracy per class
    print('\n' + '-'*70)
    print('Class-wise Accuracy (Correct Predictions):')
    print('-'*70)

    # Get unique labels from the confusion matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    for i, label in enumerate(unique_labels):
        if i < cm.shape[0]:
            tag_name = idx_to_tag.get(label, f'UNK_{label}')
            total = cm[i].sum()
            if total > 0:
                correct = cm[i, i]
                class_accuracy = correct / total
                print(f'{tag_name:<15}: {correct}/{total} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)')

    # Identify common misclassifications
    print('\n' + '-'*70)
    print('Most Common Misclassifications:')
    print('-'*70)

    misclass = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], idx_to_tag[i], idx_to_tag[j]))

    misclass.sort(reverse=True)
    for count, true_label, pred_label in misclass[:5]:
        print(f'{true_label} misclassified as {pred_label}: {count} times')

    print('='*70)

def main():
    """Main evaluation function"""

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load vocabulary
    print('Loading vocabulary...')
    try:
        word2idx = load_vocabulary(Config.VOCAB_PATH)
    except:
        print('Error: Vocabulary file not found. Please train the model first.')
        return

    # Load test data
    print('Loading test data...')
    try:
        test_sentences, test_tags = load_data(Config.TEST_DATA_PATH)
    except:
        print('Test data not found. Using training data for evaluation...')
        test_sentences, test_tags = create_sample_data()

    print(f'Number of test sentences: {len(test_sentences)}')

    # Create dataset and dataloader
    test_dataset = DrugNERDataset(test_sentences, test_tags, word2idx, Config.TAG_TO_IDX)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=pad_sequences
    )

    # Load model
    print('Loading model...')
    model = BiLSTM_CRF(
        vocab_size=len(word2idx),
        tag_to_ix=Config.TAG_TO_IDX,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)

    try:
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print('Model loaded successfully!')
    except:
        print('Error: Model file not found. Please train the model first.')
        return

    # Evaluate
    print('\nEvaluating model...')
    predictions, true_labels = evaluate_model(model, test_loader, device, Config.IDX_TO_TAG)

    # Calculate metrics
    print('\nCalculating metrics...')
    accuracy, metrics_per_class, report = calculate_multiclass_metrics(
        true_labels, predictions, Config.IDX_TO_TAG
    )

    # Print results
    print_evaluation_results(accuracy, metrics_per_class, report)

    # Plot confusion matrix
    print('\nGenerating confusion matrix...')
    labels = [Config.IDX_TO_TAG[i] for i in sorted(Config.IDX_TO_TAG.keys())]
    cm = plot_confusion_matrix(true_labels, predictions, labels)

    # Interpret confusion matrix
    interpret_confusion_matrix(cm, Config.IDX_TO_TAG, true_labels, predictions)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from typing import Dict, List, Tuple, Optional
import random
import pandas as pd
import argparse
from sklearn.metrics import f1_score, precision_score,  recall_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveBERTModel(nn.Module):
    """
    BERT-based model for contrastive learning in AI-generated text detection.
    Uses contrastive loss to learn discriminative representations.
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 hidden_dim: int = 768,
                 projection_dim: int = 256,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 temperature: float = 0.07):
        super(ContrastiveBERTModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Improved projection head with batch normalization
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Classification head for supervised learning
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, return_logits=False):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Project to contrastive space
        projections = self.projection_head(cls_embeddings)
        
        # L2 normalize projections for better contrastive learning
        projections = F.normalize(projections, p=2, dim=1)
        
        if return_logits:
            logits = self.classifier(projections)
            return projections, logits
        
        return projections

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with AI-generated text detection.
    """
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 sources: List[str],
                 tokenizer,
                 max_length: int = 512,
                 augment_prob: float = 0.3):
        self.texts = texts
        self.labels = labels
        self.sources = sources
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_prob = augment_prob
        
        # Group texts by source for better contrastive sampling
        self.source_groups = {}
        for i, source in enumerate(sources):
            if source not in self.source_groups:
                self.source_groups[source] = []
            self.source_groups[source].append(i)
        
        # Also group by labels for hard negative mining
        self.label_groups = {}
        for i, label in enumerate(labels):
            if label not in self.label_groups:
                self.label_groups[label] = []
            self.label_groups[label].append(i)
    
    def __len__(self):
        return len(self.texts)
    
    def _simple_augment(self, text: str) -> str:
        """Simple text augmentation: random word dropout"""
        if random.random() > self.augment_prob:
            return text
        
        words = text.split()
        if len(words) <= 3:
            return text
        
        # Randomly drop 10-20% of words
        drop_ratio = random.uniform(0.1, 0.2)
        keep_count = max(1, int(len(words) * (1 - drop_ratio)))
        kept_indices = sorted(random.sample(range(len(words)), keep_count))
        
        return ' '.join([words[i] for i in kept_indices])
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        source = self.sources[idx]
        
        # Apply augmentation
        augmented_text = self._simple_augment(text)
        
        # Tokenize text
        encoding = self.tokenizer(
            augmented_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Sample positive: same source
        positive_idx = self._sample_positive(idx, source)
        
        # Sample negative: different source (hard negative mining)
        negative_idx = self._sample_hard_negative(idx, source, label)
        
        # Tokenize positive and negative examples
        pos_encoding = self.tokenizer(
            self._simple_augment(self.texts[positive_idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        neg_encoding = self.tokenizer(
            self._simple_augment(self.texts[negative_idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'source': source,
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
        }
    
    def _sample_positive(self, idx, source):
        """Sample a positive example (same source)"""
        candidates = [i for i in self.source_groups[source] if i != idx]
        if not candidates:
            return idx
        return random.choice(candidates)
    
    def _sample_hard_negative(self, idx, source, label):
        """Sample hard negative: prioritize same label but different source"""
        # First try: same label, different source (harder negative)
        hard_candidates = []
        for other_source, indices in self.source_groups.items():
            if other_source != source:
                for i in indices:
                    if self.labels[i] == label:  # same label, different source
                        hard_candidates.append(i)
        
        if hard_candidates:
            return random.choice(hard_candidates)
        
        # Fallback: any different source
        all_candidates = []
        for other_source, indices in self.source_groups.items():
            if other_source != source:
                all_candidates.extend(indices)
        
        return random.choice(all_candidates) if all_candidates else idx

class ImprovedContrastiveLoss(nn.Module):
    """
    Improved contrastive loss combining InfoNCE and supervised classification.
    """
    
    def __init__(self, temperature=0.07, alpha=0.5, margin=0.5):
        super(ImprovedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance between contrastive and classification loss
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, anchor, positive, negative, labels=None, logits=None):
        batch_size = anchor.size(0)
        
        # InfoNCE contrastive loss
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature
        
        # Create logits for InfoNCE
        contrastive_logits = torch.stack([pos_sim, neg_sim], dim=1)
        contrastive_targets = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        contrastive_loss = self.ce_loss(contrastive_logits, contrastive_targets)
        
        # Add margin-based triplet loss component
        triplet_loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        total_loss = contrastive_loss + 0.1 * triplet_loss
        
        # Add supervised classification loss if available
        if labels is not None and logits is not None:
            classification_loss = self.ce_loss(logits, labels)
            total_loss = (1 - self.alpha) * total_loss + self.alpha * classification_loss
            return total_loss, contrastive_loss, classification_loss
        
        return total_loss, contrastive_loss, torch.tensor(0.0)

class AITextDetector:
    """
    Main class for AI-generated text detection using contrastive learning.
    """
    
    def __init__(self, 
                 sources: List[str],
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 2,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = ContrastiveBERTModel(
            model_name=model_name,
            num_classes=num_classes
        ).to(self.device)
        self.criterion = ImprovedContrastiveLoss()
        self.source_centroids = {}
        self.sources = sources
        
    def prepare_data(self, 
                    texts: List[str], 
                    labels: List[int],
                    sources: List[str],
                    batch_size: int = 16,
                    max_length: int = 512):
        """Prepare data for training/evaluation"""
        dataset = ContrastiveDataset(
            texts=texts,
            labels=labels,
            sources=sources,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader
    
    def train(self, 
              train_loader,
              val_loader=None,
              epochs: int = 5,
              learning_rate: float = 2e-5,
              warmup_ratio: float = 0.1):
        """Train the model with improved training strategy"""
        
        # Use different learning rates for BERT and projection head
        bert_params = list(self.model.bert.parameters())
        other_params = list(self.model.projection_head.parameters()) + list(self.model.classifier.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': learning_rate},  # Lower LR for BERT
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_contrastive_loss = 0
            total_classification_loss = 0
            
            # Collect embeddings for centroid computation in last epoch
            epoch_embeddings = {source: [] for source in self.sources}
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                sources_batch = batch['source']
                
                pos_input_ids = batch['pos_input_ids'].to(self.device)
                pos_attention_mask = batch['pos_attention_mask'].to(self.device)
                neg_input_ids = batch['neg_input_ids'].to(self.device)
                neg_attention_mask = batch['neg_attention_mask'].to(self.device)
                
                # Forward pass
                anchor_proj, anchor_logits = self.model(input_ids, attention_mask, return_logits=True)
                pos_proj = self.model(pos_input_ids, pos_attention_mask)
                neg_proj = self.model(neg_input_ids, neg_attention_mask)
                
                # Store embeddings for centroid computation
                if epoch == epochs - 1:  # Last epoch
                    for i, source in enumerate(sources_batch):
                        epoch_embeddings[source].append(anchor_proj[i].detach().cpu())
                
                # Compute loss
                total_loss_batch, contrastive_loss, classification_loss = self.criterion(
                    anchor_proj, pos_proj, neg_proj, labels, anchor_logits
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                if batch_idx < warmup_steps:
                    scheduler.step()
                
                total_loss += total_loss_batch.item()
                total_contrastive_loss += contrastive_loss.item()
                total_classification_loss += classification_loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                              f'Total Loss: {total_loss_batch.item():.4f}, '
                              f'Contrastive: {contrastive_loss.item():.4f}, '
                              f'Classification: {classification_loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            avg_contrastive = total_contrastive_loss / len(train_loader)
            avg_classification = total_classification_loss / len(train_loader)
            
            logger.info(f'Epoch {epoch+1}/{epochs} completed. '
                       f'Avg Total Loss: {avg_loss:.4f}, '
                       f'Avg Contrastive: {avg_contrastive:.4f}, '
                       f'Avg Classification: {avg_classification:.4f}')
            
            # Validation
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                logger.info(f'Validation - Accuracy: {val_metrics["accuracy"]:.4f}, '
                          f'F1: {val_metrics["f1"]:.4f}')
                
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    logger.info(f'New best validation F1: {best_val_f1:.4f}')
            
            # Compute centroids from last epoch
            if epoch == epochs - 1:
                for source in self.sources:
                    if epoch_embeddings[source]:
                        embeddings_tensor = torch.stack(epoch_embeddings[source])
                        self.source_centroids[source] = embeddings_tensor.mean(dim=0)
                    else:
                        logger.warning(f"No embeddings collected for source: {source}")
                        # Create a random centroid as fallback
                        self.source_centroids[source] = torch.randn(256)
        
        return self.source_centroids
    
    def evaluate(self, dataloader):
        """Evaluate the model using classification head"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask, return_logits=True)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, centroids, texts: List[str], batch_size: int = 16, use_classifier: bool = True, ensemble_weight: float = 0.5):
        """Predict source attribution using both classifier and centroids"""
        self.model.eval()
        
        # Create simple dataset for prediction
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_projections = []
        all_logits = []
        all_classifier_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Get both projections and classifier logits
                projections, logits = self.model(input_ids, attention_mask, return_logits=True)
                
                all_projections.append(projections.cpu())
                if use_classifier:
                    all_logits.append(logits.cpu())
                    # Convert logits to probabilities for binary classification
                    probs = F.softmax(logits, dim=1)
                    all_classifier_probs.append(probs.cpu())
        
        test_projections = torch.cat(all_projections, dim=0)
        
        if use_classifier and len(all_logits) > 0:
            test_logits = torch.cat(all_logits, dim=0)
            test_classifier_probs = torch.cat(all_classifier_probs, dim=0)
            
        
        # Method 1: Centroid-based prediction
        source_names = list(centroids.keys())
        centroid_matrix = torch.stack([centroids[src] for src in source_names])
        
        # Compute distances between test samples and centroids
        distances = torch.cdist(test_projections, centroid_matrix, p=2)
        centroid_predictions = distances.argmin(dim=1)
        
        if not use_classifier or len(all_logits) == 0:
            # Use only centroid-based prediction
            predicted_sources = [source_names[i] for i in centroid_predictions]
            return predicted_sources
        
        print("using classifeir")
        
        # Method 2: Classifier-based prediction (binary: human vs AI)
        classifier_predictions = test_logits.argmax(dim=1)  # 0 = human, 1 = AI
        classifier_confidence = test_classifier_probs.max(dim=1)[0]  # Max probability
        
        # Method 3: Ensemble approach
        predicted_sources = []
        
        for i in range(len(texts)):
            centroid_pred_idx = centroid_predictions[i].item()
            centroid_pred_source = source_names[centroid_pred_idx]
            classifier_pred = classifier_predictions[i].item()  # 0 or 1
            confidence = classifier_confidence[i].item()
            
            # If classifier is very confident about human (class 0)
            if classifier_pred == 0 and confidence > 0.8:
                predicted_sources.append('human')
            # If classifier predicts AI (class 1), use centroid to determine which AI
            elif classifier_pred == 1:
                # Among AI sources, find the closest centroid
                ai_sources = [src for src in source_names if src != 'human']
                if ai_sources:
                    ai_indices = [source_names.index(src) for src in ai_sources]
                    ai_distances = distances[i, ai_indices]
                    closest_ai_idx = ai_indices[ai_distances.argmin().item()]
                    predicted_sources.append(source_names[closest_ai_idx])
                else:
                    predicted_sources.append(centroid_pred_source)
            # If classifier is uncertain, trust centroid prediction
            else:
                predicted_sources.append(centroid_pred_source)
        return predicted_sources
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'source_centroids': self.source_centroids,
            'sources': self.sources
        }, path)
        logger.info(f'Model saved to {path}')
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.source_centroids = checkpoint.get('source_centroids', {})
        self.sources = checkpoint.get('sources', [])
        logger.info(f'Model loaded from {path}')

def get_train_data(path):
    df = pd.read_csv(path)
    df['label_binary'] = (df['label'] != 'human').astype(int)
    texts = df['Generation'].to_numpy()
    labels = df['label_binary'].to_numpy()
    sources = df['label'].to_numpy()
    return texts, labels, sources

def get_test_data(path):
    df = pd.read_csv(path)
    df['label_binary'] = (df['label'] != 'human').astype(int)
    df.columns = ['texts', 'sources', 'labels']
    return df

def train(model_path, data_path):
    """Train the AI text detector"""
    texts, labels, sources = get_train_data(data_path)
    
    # Initialize detector
    detector = AITextDetector(np.unique(sources), num_classes=2)
    
    # Prepare data with smaller batch size for better contrastive learning
    train_loader = detector.prepare_data(texts, labels, sources, batch_size=16)
    
    # Train model
    print("Training model...")
    centroids = detector.train(train_loader, epochs=3)  # Increased epochs
    
    # Save both model and centroids
    detector.save_model(model_path)
    torch.save(centroids, model_path.replace('.pt', '_centroids.pt'))

def test(model_path, data_path):
    df = get_test_data(data_path)
    
    detector = AITextDetector(df['sources'].unique(), num_classes=2)
    detector.load_model(model_path)
    
    # Load centroids
    centroids_path = model_path.replace('.pt', '_centroids.pt')
    try:
        centroids = torch.load(centroids_path)
    except FileNotFoundError:
        logger.error(f"Centroids file not found: {centroids_path}")
        logger.info("Using centroids from model checkpoint")
        centroids = detector.source_centroids
    
    f1_scores_weighted = {}
    p_scores_weighted = {}
    r_scores_weighted = {}
    
    A_TT = []
    P_TT = []
    R_TT = []
    F1_TT = []

    for ai_name in df['sources'].unique():
        if ai_name == "human":
            continue
            
        subset = df[(df['labels'] == 0) | (df['sources'] == ai_name)]
        
        if subset.empty or len(subset['labels'].unique()) < 2:
            continue

        y_true = subset['sources'].values
        y_pred = detector.predict(centroids, subset['texts'].tolist())
        
        y_true_binary = subset['labels'].values
        y_pred_binary = [0 if pred == 'human' else 1 for pred in y_pred]
        
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, average='binary')
        recall = recall_score(y_true_binary, y_pred_binary, average='binary')
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
        
        A_TT.append(accuracy)
        P_TT.append(precision)
        R_TT.append(recall)
        F1_TT.append(f1)

        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        p_weighted = precision_score(y_true, y_pred, average='weighted')
        r_weighted = recall_score(y_true, y_pred, average='weighted')
        
        f1_scores_weighted[ai_name] = f1_weighted
        p_scores_weighted[ai_name] = p_weighted
        r_scores_weighted[ai_name] = r_weighted
        
        logger.info(f'{ai_name} - Weighted F1: {f1_weighted:.4f}')
        logger.info(f'{ai_name} - Weighted P: {p_weighted:.4f}')
        logger.info(f'{ai_name} - Weighted R: {r_weighted:.4f}')
    
    avg_f1_weighted = np.mean(list(f1_scores_weighted.values()))
    avg_p_weighted = np.mean(list(p_scores_weighted.values()))
    avg_r_weighted = np.mean(list(r_scores_weighted.values()))
    
    avg_f1_TT = np.mean(F1_TT)
    avg_p_TT = np.mean(P_TT)
    avg_r_TT = np.mean(R_TT)
    avg_a_TT = np.mean(A_TT)
    
    logger.info(f'Average Weighted F1: {avg_f1_weighted:.4f}')
    logger.info(f'Average Weighted P: {avg_p_weighted:.4f}')
    logger.info(f'Average Weighted R: {avg_r_weighted:.4f}')
    
    logger.info(f'Average Weighted F1 TT: {avg_f1_TT:.4f}')
    logger.info(f'Average Weighted P TT: {avg_p_TT:.4f}')
    logger.info(f'Average Weighted R TT: {avg_r_TT:.4f}')
    logger.info(f'Average Weighted A TT: {avg_a_TT:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the improved contrastive AI text detector")
    parser.add_argument("--mode", choices=['train', 'test'], default='test', help="Choose the mode")
    args = parser.parse_args()
    
    if args.mode == 'train':
        model_path = "./models/improved_contrastive_bert.pt"
        data_path = "all_gpt_train.csv"
        train(model_path, data_path)
    elif args.mode == 'test':
        model_path = "./models/improved_contrastive_bert.pt"
        data_path = "all_gpt_test.csv"
        test(model_path, data_path)
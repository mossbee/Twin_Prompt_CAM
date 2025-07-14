import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


class TwinVerificationMetrics:
    """
    Metrics calculator for twin face verification
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics for new evaluation"""
        self.predictions = []
        self.labels = []
        self.scores = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        """
        Update metrics with batch predictions
        
        Args:
            predictions: Binary predictions (0 or 1)
            labels: Ground truth labels (0 or 1)
            scores: Confidence scores/probabilities
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.labels.extend(labels.flatten())
        self.scores.extend(scores.flatten())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all verification metrics
        
        Returns:
            Dictionary containing all metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        scores = np.array(self.scores)
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Equal Error Rate (EER)
        eer, eer_threshold = self._compute_eer(fpr, tpr, thresholds)
        
        # False Acceptance Rate (FAR) at different thresholds
        far_1 = self._compute_far_at_threshold(labels, scores, 0.01)
        far_01 = self._compute_far_at_threshold(labels, scores, 0.001)
        far_001 = self._compute_far_at_threshold(labels, scores, 0.0001)
        
        # Verification accuracy at EER threshold
        verification_accuracy = self._compute_verification_accuracy(labels, scores, eer_threshold)
        
        # Average Precision
        avg_precision = average_precision_score(labels, scores)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'far_1': far_1,
            'far_01': far_01,
            'far_001': far_001,
            'verification_accuracy': verification_accuracy,
            'average_precision': avg_precision
        }
    
    def _compute_eer(self, fpr: np.ndarray, tpr: np.ndarray, 
                    thresholds: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER)
        
        Args:
            fpr: False Positive Rate
            tpr: True Positive Rate
            thresholds: Thresholds
            
        Returns:
            EER value and corresponding threshold
        """
        fnr = 1 - tpr  # False Negative Rate
        
        # Find the threshold where FPR and FNR are closest
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold
    
    def _compute_far_at_threshold(self, labels: np.ndarray, scores: np.ndarray, 
                                 threshold: float) -> float:
        """
        Compute False Acceptance Rate at given threshold
        
        Args:
            labels: Ground truth labels
            scores: Prediction scores
            threshold: Threshold value
            
        Returns:
            FAR value
        """
        predictions = (scores >= threshold).astype(int)
        
        # FAR = FP / (FP + TN) = FP / Total_Negatives
        true_negatives = (labels == 0)
        false_positives = (predictions == 1) & (labels == 0)
        
        if np.sum(true_negatives) == 0:
            return 0.0
        
        far = np.sum(false_positives) / np.sum(true_negatives)
        return far
    
    def _compute_verification_accuracy(self, labels: np.ndarray, scores: np.ndarray, 
                                     threshold: float) -> float:
        """
        Compute verification accuracy at given threshold
        
        Args:
            labels: Ground truth labels
            scores: Prediction scores
            threshold: Threshold value
            
        Returns:
            Verification accuracy
        """
        predictions = (scores >= threshold).astype(int)
        return accuracy_score(labels, predictions)
    
    def get_roc_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data for plotting
        
        Returns:
            FPR, TPR, and thresholds
        """
        labels = np.array(self.labels)
        scores = np.array(self.scores)
        
        return roc_curve(labels, scores)
    
    def get_precision_recall_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data for plotting
        
        Returns:
            Precision, Recall, and thresholds
        """
        labels = np.array(self.labels)
        scores = np.array(self.scores)
        
        return precision_recall_curve(labels, scores)
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        fpr, tpr, _ = self.get_roc_curve_data()
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Twin Face Verification')
        ax.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            matplotlib figure
        """
        precision, recall, _ = self.get_precision_recall_curve_data()
        avg_precision = average_precision_score(np.array(self.labels), np.array(self.scores))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Twin Face Verification')
        ax.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def compute_twin_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                        scores: torch.Tensor) -> Dict[str, float]:
    """
    Compute twin verification metrics for a batch
    
    Args:
        predictions: Binary predictions (0 or 1)
        labels: Ground truth labels (0 or 1)
        scores: Confidence scores/probabilities
        
    Returns:
        Dictionary containing metrics
    """
    metrics = TwinVerificationMetrics()
    metrics.update(predictions, labels, scores)
    return metrics.compute_metrics()


def compute_dataset_statistics(dataset_info: Dict, twin_pairs: List) -> Dict[str, float]:
    """
    Compute dataset statistics for logging
    
    Args:
        dataset_info: Dataset information dictionary
        twin_pairs: List of twin pairs
        
    Returns:
        Dictionary containing dataset statistics
    """
    total_persons = len(dataset_info)
    total_images = sum(len(images) for images in dataset_info.values())
    
    images_per_person = [len(images) for images in dataset_info.values()]
    images_per_person_avg = np.mean(images_per_person)
    images_per_person_min = np.min(images_per_person)
    images_per_person_max = np.max(images_per_person)
    
    # Calculate total possible positive pairs
    total_positive_pairs = sum(
        len(images) * (len(images) - 1) // 2 
        for images in dataset_info.values() 
        if len(images) >= 2
    )
    
    # Calculate twin pairs
    twin_pairs_count = len(twin_pairs)
    
    # Estimate total negative pairs (rough calculation)
    total_negative_pairs = total_positive_pairs  # Assuming 1:1 ratio
    
    return {
        'total_persons': total_persons,
        'total_images': total_images,
        'images_per_person_avg': images_per_person_avg,
        'images_per_person_min': images_per_person_min,
        'images_per_person_max': images_per_person_max,
        'total_pairs': total_positive_pairs + total_negative_pairs,
        'positive_pairs': total_positive_pairs,
        'negative_pairs': total_negative_pairs,
        'twin_pairs': twin_pairs_count
    }


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 
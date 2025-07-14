import json
import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from data.dataset.utils import get_transformation, default_loader
from itertools import combinations


class TwinPairDataset(Dataset):
    """
    Dataset for twin face verification task.
    Generates positive pairs (same person) and negative pairs (different person + twin pairs as hard negatives).
    """
    
    def __init__(self, train_dataset_infor_path, train_twin_pairs_path, transform=None, phase='base', 
                 positive_ratio=0.5, twin_negative_ratio=0.3):
        """
        Args:
            train_dataset_infor_path: Path to train_dataset_infor.json
            train_twin_pairs_path: Path to train_twin_pairs.json
            transform: Image transformations
            phase: Training phase ('base', 'twin_focused', 'attention_refine')
            positive_ratio: Ratio of positive pairs in the dataset
            twin_negative_ratio: Ratio of twin pairs among negative pairs
        """
        self.transform = transform
        self.phase = phase
        self.positive_ratio = positive_ratio
        self.twin_negative_ratio = twin_negative_ratio
        self.loader = default_loader
        
        # Load dataset information
        with open(train_dataset_infor_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        # Load twin pairs information
        with open(train_twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Convert twin pairs to set for faster lookup
        self.twin_pairs_set = set()
        for pair in self.twin_pairs:
            self.twin_pairs_set.add((pair[0], pair[1]))
            self.twin_pairs_set.add((pair[1], pair[0]))  # Add reverse pair
        
        # Get all person IDs
        self.person_ids = list(self.dataset_info.keys())
        
        # Generate pairs based on phase
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self):
        """Generate positive and negative pairs based on training phase"""
        pairs = []
        
        if self.phase == 'base':
            # Phase 1: Balanced positive/negative pairs
            pairs = self._generate_balanced_pairs()
        elif self.phase == 'twin_focused':
            # Phase 2: Emphasize twin pairs as hard negatives
            pairs = self._generate_twin_focused_pairs()
        elif self.phase == 'attention_refine':
            # Phase 3: All data with focus on challenging cases
            pairs = self._generate_attention_refine_pairs()
        
        return pairs
    
    def _generate_balanced_pairs(self):
        """Generate balanced positive and negative pairs"""
        pairs = []
        
        # Generate positive pairs (same person)
        positive_pairs = []
        for person_id, image_paths in self.dataset_info.items():
            if len(image_paths) >= 2:
                # Generate all combinations of images for this person
                for img1, img2 in combinations(image_paths, 2):
                    positive_pairs.append((img1, img2, 1))  # Label 1 for same person
        
        # Generate negative pairs (different person)
        negative_pairs = []
        target_negative_count = int(len(positive_pairs) / self.positive_ratio * (1 - self.positive_ratio))
        
        # Calculate how many twin pairs to include
        twin_negative_count = int(target_negative_count * self.twin_negative_ratio)
        regular_negative_count = target_negative_count - twin_negative_count
        
        # Add twin pairs as hard negatives
        for pair in self.twin_pairs:
            if twin_negative_count <= 0:
                break
            person1_id, person2_id = pair
            if person1_id in self.dataset_info and person2_id in self.dataset_info:
                img1 = random.choice(self.dataset_info[person1_id])
                img2 = random.choice(self.dataset_info[person2_id])
                negative_pairs.append((img1, img2, 0))  # Label 0 for different person
                twin_negative_count -= 1
        
        # Add regular negative pairs
        for _ in range(regular_negative_count):
            person1_id = random.choice(self.person_ids)
            person2_id = random.choice(self.person_ids)
            
            # Ensure different persons and not twins
            if (person1_id != person2_id and 
                (person1_id, person2_id) not in self.twin_pairs_set):
                img1 = random.choice(self.dataset_info[person1_id])
                img2 = random.choice(self.dataset_info[person2_id])
                negative_pairs.append((img1, img2, 0))
        
        # Combine and shuffle
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)
        return pairs
    
    def _generate_twin_focused_pairs(self):
        """Generate pairs with emphasis on twin pairs as hard negatives"""
        pairs = []
        
        # Generate positive pairs (same person)
        positive_pairs = []
        for person_id, image_paths in self.dataset_info.items():
            if len(image_paths) >= 2:
                for img1, img2 in combinations(image_paths, 2):
                    positive_pairs.append((img1, img2, 1))
        
        # Generate negative pairs with higher ratio of twin pairs
        negative_pairs = []
        target_negative_count = len(positive_pairs)
        
        # Increase twin negative ratio for this phase
        twin_negative_count = int(target_negative_count * 0.7)  # 70% twin pairs
        regular_negative_count = target_negative_count - twin_negative_count
        
        # Add twin pairs as hard negatives (sample multiple pairs per twin)
        twin_pairs_added = 0
        for pair in self.twin_pairs:
            if twin_pairs_added >= twin_negative_count:
                break
            person1_id, person2_id = pair
            if person1_id in self.dataset_info and person2_id in self.dataset_info:
                # Sample multiple image pairs for each twin pair
                for _ in range(min(3, (twin_negative_count - twin_pairs_added))):
                    img1 = random.choice(self.dataset_info[person1_id])
                    img2 = random.choice(self.dataset_info[person2_id])
                    negative_pairs.append((img1, img2, 0))
                    twin_pairs_added += 1
        
        # Add regular negative pairs
        for _ in range(regular_negative_count):
            person1_id = random.choice(self.person_ids)
            person2_id = random.choice(self.person_ids)
            
            if (person1_id != person2_id and 
                (person1_id, person2_id) not in self.twin_pairs_set):
                img1 = random.choice(self.dataset_info[person1_id])
                img2 = random.choice(self.dataset_info[person2_id])
                negative_pairs.append((img1, img2, 0))
        
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)
        return pairs
    
    def _generate_attention_refine_pairs(self):
        """Generate pairs for attention refinement phase"""
        # Similar to balanced pairs but with all available data
        return self._generate_balanced_pairs()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self.loader(img1_path)
        img2 = self.loader(img2_path)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_twin_dataset(params, mode='train', phase='base'):
    """
    Create twin verification dataset
    
    Args:
        params: Parameters object containing data paths
        mode: 'train' or 'test'
        phase: Training phase ('base', 'twin_focused', 'attention_refine')
    """
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    
    if mode == 'train':
        transform = get_transformation('train', mean, std)
        dataset_infor_path = os.path.join(params.data_path, 'train_dataset_infor.json')
        twin_pairs_path = os.path.join(params.data_path, 'train_twin_pairs.json')
    elif mode == 'test':
        transform = get_transformation('test', mean, std)
        dataset_infor_path = os.path.join(params.data_path, 'test_dataset_infor.json')
        twin_pairs_path = os.path.join(params.data_path, 'test_twin_pairs.json')
    else:
        raise ValueError("Mode must be 'train' or 'test'")
    
    return TwinPairDataset(
        train_dataset_infor_path=dataset_infor_path,
        train_twin_pairs_path=twin_pairs_path,
        transform=transform,
        phase=phase
    )


def get_twin(params, mode='train', phase='base'):
    """
    Main function to get twin dataset following the existing pattern
    
    Args:
        params: Parameters object
        mode: 'train' or 'test'
        phase: Training phase for different training strategies
    """
    params.class_num = 2  # Binary classification: same/different person
    
    return get_twin_dataset(params, mode, phase) 
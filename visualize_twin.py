import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
from experiment.build_model import get_model
from experiment.build_loader_twin import get_twin_loader_for_phase
from utils.setup_logging import get_logger
from utils.misc import set_seed, load_yaml, override_args_with_yaml
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from data.dataset.utils import get_transformation

logger = get_logger("Prompt_CAM")


class TwinVisualizationTool:
    """
    Visualization tool for twin face verification attention maps
    """
    
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.device = params.device
        self.model.eval()
        
        # Set up transforms
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        self.transform = get_transformation('test', mean, std)
    
    def load_image_pair(self, img1_path, img2_path):
        """Load and preprocess image pair"""
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1_tensor = self.transform(img1).unsqueeze(0)
            img2_tensor = self.transform(img2).unsqueeze(0)
        else:
            img1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        return img1_tensor, img2_tensor, img1, img2
    
    def denormalize_tensor(self, tensor):
        """Denormalize tensor for visualization"""
        mean = torch.tensor(IMAGENET_INCEPTION_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_INCEPTION_STD).view(3, 1, 1)
        
        tensor = tensor.clone()
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
    
    def get_attention_maps(self, img1_tensor, img2_tensor):
        """Get attention maps for image pair"""
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        with torch.no_grad():
            # Enable attention visualization
            original_vis_attn = getattr(self.params, 'vis_attn', False)
            self.params.vis_attn = True
            
            verification_score, attn1, attn2 = self.model(img1_tensor, img2_tensor)
            
            # Restore original setting
            self.params.vis_attn = original_vis_attn
        
        return verification_score.cpu(), attn1, attn2
    
    def visualize_attention_pair(self, img1_path, img2_path, save_path=None, title=None):
        """Visualize attention maps for a twin pair"""
        
        # Load images
        img1_tensor, img2_tensor, img1_pil, img2_pil = self.load_image_pair(img1_path, img2_path)
        
        # Get attention maps
        verification_score, attn1, attn2 = self.get_attention_maps(img1_tensor, img2_tensor)
        
        # Denormalize for visualization
        img1_norm = self.denormalize_tensor(img1_tensor.squeeze(0))
        img2_norm = self.denormalize_tensor(img2_tensor.squeeze(0))
        
        # Convert to numpy
        img1_np = img1_norm.permute(1, 2, 0).numpy()
        img2_np = img2_norm.permute(1, 2, 0).numpy()
        
        # Create visualization
        fig = self._create_attention_visualization(
            img1_np, img2_np, attn1, attn2, 
            verification_score.item(), title
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved: {save_path}")
        
        return fig, verification_score.item()
    
    def _create_attention_visualization(self, img1_np, img2_np, attn1, attn2, 
                                      verification_score, title=None):
        """Create attention map visualization"""
        
        # Determine number of attention heads to show
        max_heads = 6
        if attn1 is not None and len(attn1.shape) >= 3:
            num_heads = min(max_heads, attn1.shape[0])
        else:
            num_heads = 0
        
        # Create subplots
        cols = max(4, num_heads + 2)
        fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 6))
        
        # Ensure axes is 2D
        if len(axes.shape) == 1:
            axes = axes.reshape(2, -1)
        
        # Plot original images
        axes[0, 0].imshow(img1_np)
        axes[0, 0].set_title('Image 1')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img2_np)
        axes[1, 0].set_title('Image 2')
        axes[1, 0].axis('off')
        
        # Plot attention maps
        if attn1 is not None and attn2 is not None and num_heads > 0:
            attn1_np = attn1.detach().cpu().numpy()
            attn2_np = attn2.detach().cpu().numpy()
            
            for i in range(num_heads):
                col_idx = i + 1
                if col_idx < cols:
                    # Resize attention map to match image size
                    attn1_resized = self._resize_attention_map(attn1_np[i], img1_np.shape[:2])
                    attn2_resized = self._resize_attention_map(attn2_np[i], img2_np.shape[:2])
                    
                    # Image 1 attention
                    axes[0, col_idx].imshow(img1_np)
                    axes[0, col_idx].imshow(attn1_resized, alpha=0.6, cmap='jet')
                    axes[0, col_idx].set_title(f'Attention Head {i+1}')
                    axes[0, col_idx].axis('off')
                    
                    # Image 2 attention
                    axes[1, col_idx].imshow(img2_np)
                    axes[1, col_idx].imshow(attn2_resized, alpha=0.6, cmap='jet')
                    axes[1, col_idx].set_title(f'Attention Head {i+1}')
                    axes[1, col_idx].axis('off')
        
        # Hide unused subplots
        for i in range(num_heads + 1, cols):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        # Set main title
        prediction_text = f"Verification Score: {verification_score:.4f}"
        same_different = "Same Person" if verification_score > 0.5 else "Different Person"
        main_title = f"{title}\n{prediction_text} ({same_different})" if title else f"{prediction_text} ({same_different})"
        
        fig.suptitle(main_title, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _resize_attention_map(self, attention_map, target_size):
        """Resize attention map to target size"""
        from scipy.ndimage import zoom
        
        h_ratio = target_size[0] / attention_map.shape[0]
        w_ratio = target_size[1] / attention_map.shape[1]
        
        resized = zoom(attention_map, (h_ratio, w_ratio), order=1)
        return resized
    
    def visualize_twin_pairs(self, train_dataset_infor_path, train_twin_pairs_path, 
                           num_pairs=5, save_dir=None):
        """Visualize multiple twin pairs"""
        
        # Load dataset info
        with open(train_dataset_infor_path, 'r') as f:
            dataset_info = json.load(f)
        
        with open(train_twin_pairs_path, 'r') as f:
            twin_pairs = json.load(f)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = []
        
        # Visualize twin pairs
        logger.info(f"Visualizing {num_pairs} twin pairs...")
        for i, pair in enumerate(twin_pairs[:num_pairs]):
            person1_id, person2_id = pair
            
            if person1_id in dataset_info and person2_id in dataset_info:
                # Get random images for each person
                img1_path = np.random.choice(dataset_info[person1_id])
                img2_path = np.random.choice(dataset_info[person2_id])
                
                title = f"Twin Pair {i+1}: {person1_id} vs {person2_id}"
                save_path = os.path.join(save_dir, f"twin_pair_{i+1}.png") if save_dir else None
                
                fig, score = self.visualize_attention_pair(img1_path, img2_path, save_path, title)
                
                results.append({
                    'pair_id': i+1,
                    'person1_id': person1_id,
                    'person2_id': person2_id,
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'verification_score': score,
                    'prediction': 'Same Person' if score > 0.5 else 'Different Person'
                })
                
                plt.close(fig)
        
        # Visualize same person pairs for comparison
        logger.info(f"Visualizing {num_pairs} same person pairs...")
        same_person_count = 0
        for person_id, image_paths in dataset_info.items():
            if len(image_paths) >= 2 and same_person_count < num_pairs:
                # Get two random images of the same person
                img1_path, img2_path = np.random.choice(image_paths, 2, replace=False)
                
                title = f"Same Person {same_person_count+1}: {person_id}"
                save_path = os.path.join(save_dir, f"same_person_{same_person_count+1}.png") if save_dir else None
                
                fig, score = self.visualize_attention_pair(img1_path, img2_path, save_path, title)
                
                results.append({
                    'pair_id': f'same_{same_person_count+1}',
                    'person1_id': person_id,
                    'person2_id': person_id,
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'verification_score': score,
                    'prediction': 'Same Person' if score > 0.5 else 'Different Person'
                })
                
                plt.close(fig)
                same_person_count += 1
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Twin Face Verification Visualization')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    # Data
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--use_test_data', action='store_true',
                       help='Use test dataset instead of training dataset')
    
    # Visualization options
    parser.add_argument('--num_pairs', type=int, default=5,
                       help='Number of pairs to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--single_pair', nargs=2, type=str,
                       help='Visualize single pair: img1_path img2_path')
    
    # System
    parser.add_argument('--random_seed', default=42, type=int,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)
    
    set_seed(args.random_seed)
    
    # Load model
    logger.info("Loading model...")
    model, _, _ = get_model(args)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create visualization tool
    visualizer = TwinVisualizationTool(model, args)
    
    if args.single_pair:
        # Visualize single pair
        img1_path, img2_path = args.single_pair
        save_path = os.path.join(args.save_dir, 'single_pair_visualization.png')
        os.makedirs(args.save_dir, exist_ok=True)
        
        fig, score = visualizer.visualize_attention_pair(img1_path, img2_path, save_path)
        logger.info(f"Verification score: {score:.4f}")
        plt.show()
    
    else:
        # Visualize multiple pairs
        if args.use_test_data:
            train_dataset_infor_path = os.path.join(args.data_path, 'test_dataset_infor.json')
            train_twin_pairs_path = os.path.join(args.data_path, 'test_twin_pairs.json')
        else:
            train_dataset_infor_path = os.path.join(args.data_path, 'train_dataset_infor.json')
            train_twin_pairs_path = os.path.join(args.data_path, 'train_twin_pairs.json')
        
        results = visualizer.visualize_twin_pairs(
            train_dataset_infor_path, train_twin_pairs_path,
            num_pairs=args.num_pairs, save_dir=args.save_dir
        )
        
        # Print results summary
        logger.info("\nVisualization Results Summary:")
        for result in results:
            logger.info(f"Pair {result['pair_id']}: {result['verification_score']:.4f} - {result['prediction']}")


if __name__ == '__main__':
    main() 
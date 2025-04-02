import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
from model import get_model

class BrainTumorDataset(Dataset):
    def __init__(self, processed_data_dir, split, transform=None):
        self.data_dir = os.path.join(processed_data_dir, split)
        self.transform = transform
        
        # Get all image paths and labels
        self.healthy_paths = self._get_image_paths('healthy')
        self.tumor_paths = self._get_image_paths('tumor')
        self.image_paths = self.healthy_paths + self.tumor_paths
        self.labels = [0] * len(self.healthy_paths) + [1] * len(self.tumor_paths)
        
        # Shuffle the data
        idx = np.random.permutation(len(self.image_paths))
        self.image_paths = np.array(self.image_paths)[idx]
        self.labels = np.array(self.labels)[idx]

    def _get_image_paths(self, category):
        category_dir = os.path.join(self.data_dir, category)
        return [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        # Create segmentation mask (placeholder)
        seg_mask = torch.zeros((1, 256, 256))
        if self.labels[idx] == 1:  # tumor case
            seg_mask = torch.ones((1, 256, 256)) * 0.5
            
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)  # Add [] to make it 1D
        
        return img, seg_mask, label

def calculate_metrics(outputs, targets, threshold=0.5):
    # For segmentation
    seg_pred = (outputs[0] > threshold).float()
    seg_acc = (seg_pred == targets[0]).float().mean()
    
    # For classification
    class_pred = (outputs[1].squeeze() > threshold).float()
    class_acc = (class_pred == targets[1]).float().mean()
    
    return seg_acc.item(), class_acc.item()

def train_model(processed_data_dir, model_save_dir, batch_size=8, epochs=50):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nTraining Setup:")
    print(f"- Using device: {device}")
    
    try:
        # Create model with explicit device
        model = get_model(device)
        print("- Model initialized successfully")
        
        # Create directories
        os.makedirs(model_save_dir, exist_ok=True)
        print("- Directories created")
        
        # Define loss functions and optimizer
        seg_criterion = nn.BCELoss().to(device)
        class_criterion = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters())
        print("- Loss functions and optimizer created")
        
        # Create datasets
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = BrainTumorDataset(processed_data_dir, 'train', transform)
        val_dataset = BrainTumorDataset(processed_data_dir, 'val', transform)
        print("- Datasets created")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print("- Data loaders created")
        
        # Test first batch
        print("\nTesting first batch:")
        images, seg_masks, labels = next(iter(train_loader))
        print(f"- Image batch device: {images.device}, shape: {images.shape}")
        images = images.to(device)
        print(f"- Image batch moved to device: {images.device}")
        
        # Training loop
        best_val_loss = float('inf')
        best_val_seg_acc = 0
        best_val_class_acc = 0
        patience = 10
        patience_counter_loss = 0
        patience_counter_seg = 0
        patience_counter_class = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_seg_loss = 0
            train_class_loss = 0
            train_seg_acc = 0
            train_class_acc = 0
            
            # Training
            for images, seg_masks, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                images = images.to(device)
                seg_masks = seg_masks.to(device)
                labels = labels.view(-1).to(device)
                
                optimizer.zero_grad()
                seg_output, class_output = model(images)
                
                class_output = class_output.view(-1)
                
                seg_loss = seg_criterion(seg_output, seg_masks)
                class_loss = class_criterion(class_output, labels)
                loss = seg_loss + 0.5 * class_loss
                
                loss.backward()
                optimizer.step()
                
                # Calculate accuracies
                seg_acc, class_acc = calculate_metrics((seg_output, class_output), (seg_masks, labels))
                
                train_loss += loss.item()
                train_seg_loss += seg_loss.item()
                train_class_loss += class_loss.item()
                train_seg_acc += seg_acc
                train_class_acc += class_acc
            
            # Validation
            model.eval()
            val_loss = 0
            val_seg_loss = 0
            val_class_loss = 0
            val_seg_acc = 0
            val_class_acc = 0
            
            with torch.no_grad():
                for images, seg_masks, labels in val_loader:
                    images = images.to(device)
                    seg_masks = seg_masks.to(device)
                    labels = labels.view(-1).to(device)
                    
                    seg_output, class_output = model(images)
                    class_output = class_output.view(-1)
                    
                    seg_loss = seg_criterion(seg_output, seg_masks)
                    class_loss = class_criterion(class_output, labels)
                    loss = seg_loss + 0.5 * class_loss
                    
                    # Calculate accuracies
                    seg_acc, class_acc = calculate_metrics((seg_output, class_output), (seg_masks, labels))
                    
                    val_loss += loss.item()
                    val_seg_loss += seg_loss.item()
                    val_class_loss += class_loss.item()
                    val_seg_acc += seg_acc
                    val_class_acc += class_acc
            
            # Calculate average metrics
            val_loss = val_loss/len(val_loader)
            val_seg_acc = val_seg_acc/len(val_loader)
            val_class_acc = val_class_acc/len(val_loader)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Train Seg Loss: {train_seg_loss/len(train_loader):.4f}")
            print(f"Train Class Loss: {train_class_loss/len(train_loader):.4f}")
            print(f"Train Seg Acc: {train_seg_acc/len(train_loader):.4f}")
            print(f"Train Class Acc: {train_class_acc/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Seg Loss: {val_seg_loss/len(val_loader):.4f}")
            print(f"Val Class Loss: {val_class_loss/len(val_loader):.4f}")
            print(f"Val Seg Acc: {val_seg_acc:.4f}")
            print(f"Val Class Acc: {val_class_acc:.4f}")
            
            # Save models based on different metrics
            # 1. Best combined loss model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter_loss = 0
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model_loss.pth'))
            else:
                patience_counter_loss += 1
            
            # 2. Best segmentation accuracy model
            if val_seg_acc > best_val_seg_acc:
                best_val_seg_acc = val_seg_acc
                patience_counter_seg = 0
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model_seg.pth'))
            else:
                patience_counter_seg += 1
            
            # 3. Best classification accuracy model
            if val_class_acc > best_val_class_acc:
                best_val_class_acc = val_class_acc
                patience_counter_class = 0
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model_class.pth'))
            else:
                patience_counter_class += 1
            
            # Early stopping if all metrics stop improving
            if patience_counter_loss >= patience and patience_counter_seg >= patience and patience_counter_class >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best validation segmentation accuracy: {best_val_seg_acc:.4f}")
                print(f"Best validation classification accuracy: {best_val_class_acc:.4f}")
                break
        
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        raise


if __name__ == "__main__":
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    processed_data_dir = os.path.join(project_dir, "data", "processed_data")
    model_save_dir = os.path.join(project_dir, "data", "models", "checkpoints")
    
    print(f"\nTraining Configuration:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"- MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("- MPS device: Apple Silicon/AMD GPU")
    print(f"- Data directory: {processed_data_dir}")
    print(f"- Model save directory: {model_save_dir}\n")
    
    # Train model
    train_model(processed_data_dir, model_save_dir) 

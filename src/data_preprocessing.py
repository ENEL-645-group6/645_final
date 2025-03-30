import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, data_dir, processed_dir, target_size=(256, 256)):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.target_size = target_size
        
        # Create processed data directories
        for split in ['train', 'val', 'test']:
            for category in ['healthy', 'tumor']:
                os.makedirs(os.path.join(processed_dir, split, category), exist_ok=True)

    def process_images(self):
        # Load and process healthy images
        healthy_dir = os.path.join(self.data_dir, 'Brain Tumor CT scan Images', 'Healthy')
        healthy_images = self._load_images(healthy_dir, label='healthy')

        # Load and process tumor images
        tumor_dir = os.path.join(self.data_dir, 'Brain Tumor CT scan Images', 'Tumor')
        tumor_images = self._load_images(tumor_dir, label='tumor')

        # Split datasets
        return self._split_and_save_data(healthy_images, tumor_images)

    def _load_images(self, directory, label):
        images = []
        for filename in tqdm(os.listdir(directory), desc=f'Processing {label} images'):
            if filename.endswith('.jpg'):
                path = os.path.join(directory, filename)
                try:
                    # Read and resize image
                    img = cv2.imread(path)
                    img = cv2.resize(img, self.target_size)
                    images.append({
                        'image': img,
                        'filename': filename,
                        'label': label
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return images

    def _split_and_save_data(self, healthy_images, tumor_images):
        # Combine and shuffle
        all_images = healthy_images + tumor_images
        np.random.shuffle(all_images)

        # Split into train (70%), validation (15%), and test (15%)
        train_data, temp_data = train_test_split(all_images, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Save splits
        self._save_split(train_data, 'train')
        self._save_split(val_data, 'val')
        self._save_split(test_data, 'test')

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

    def _save_split(self, data, split_name):
        for item in tqdm(data, desc=f'Saving {split_name} set'):
            save_path = os.path.join(
                self.processed_dir,
                split_name,
                item['label'],
                item['filename']
            )
            cv2.imwrite(save_path, item['image'])

if __name__ == "__main__":
    # Set up paths using parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")
    processed_dir = os.path.join(project_dir, "data", "processed_data")
    
    # Initialize and run preprocessing
    preprocessor = DataPreprocessor(data_dir, processed_dir)
    splits = preprocessor.process_images()
    
    # Print dataset statistics
    for split_name, split_data in splits.items():
        print(f"\n{split_name} set:")
        healthy_count = sum(1 for x in split_data if x['label'] == 'healthy')
        tumor_count = sum(1 for x in split_data if x['label'] == 'tumor')
        print(f"Healthy: {healthy_count}, Tumor: {tumor_count}")

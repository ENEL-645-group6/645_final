# Brain Tumor Detection and Segmentation

This project implements a deep learning solution for brain tumor detection and segmentation using CT scan images. The model combines a UNet architecture for segmentation with a classification branch for tumor detection.

## Project Structure

```
brain_tumor/
├── data/
│   ├── Brain Tumor CT scan Images/
│   │   ├── Healthy/
│   │   └── Tumor/
│   └── processed_data/
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── check_gpu.py
└── environment.yml
```

## Data Processing

The `data_preprocessing.py` script handles the following tasks:
1. Loads CT scan images from the raw data directory
2. Resizes images to 256x256 pixels
3. Splits the dataset into:
   - Training set (70%)
   - Validation set (15%)
   - Test set (15%)
4. Organizes processed images into appropriate directories

## Model Architecture

The model (`model.py`) implements a dual-task architecture:
1. **UNet for Segmentation**:
   - Encoder path with 4 downsampling blocks
   - Bridge layer
   - Decoder path with 4 upsampling blocks
   - Skip connections between corresponding encoder and decoder levels

2. **Classification Branch**:
   - Global average pooling on the bridge features
   - Two fully connected layers with dropout
   - Binary classification output (tumor/no tumor)

## Training Process

The training script (`train.py`) implements:
1. **Data Loading**:
   - Uses PyTorch DataLoader with batch size of 8
   - Applies image normalization
   - Handles both segmentation masks and classification labels

2. **Training Loop**:
   - Combined loss function (segmentation + classification)
   - Adam optimizer
   - Multi-metric early stopping with patience of 10 epochs
   - Three separate model checkpoints:
     - `best_model_loss.pth`: Best combined validation loss
     - `best_model_seg.pth`: Best segmentation accuracy
     - `best_model_class.pth`: Best classification accuracy

3. **Metrics**:
   - Segmentation accuracy
   - Classification accuracy
   - Combined loss
   - Individual task losses

4. **Early Stopping**:
   - Monitors three separate metrics
   - Stops training when all metrics stop improving
   - Prints best achieved values for each metric

## GPU Acceleration

The project utilizes CUDA acceleration for faster training:
- Checks GPU availability
- Moves model and data to GPU
- Optimized for NVIDIA GPUs

## Setup and Usage

1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Activate environment:
```bash
conda activate brain_tumor
```

3. Run data preprocessing:
```bash
python src/data_preprocessing.py
```

4. Train the model:
```bash
python src/train.py
```

To restart training from scratch:
```bash
rm data/models/checkpoints/best_model_*.pth
python src/train.py
```

## Performance

- Training time: ~2.5 minutes per epoch on RTX 4060
- Multi-metric early stopping
- Saves three specialized model checkpoints
- Optimized for both segmentation and classification tasks

## Future Improvements

1. Implement real segmentation masks instead of placeholders
2. Add data augmentation
3. Experiment with different model architectures
4. Add learning rate scheduling
5. Implement cross-validation
6. Add model loading from specific checkpoints
7. Implement different patience values for each metric
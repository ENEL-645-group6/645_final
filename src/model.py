import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithClassification(nn.Module):
    def __init__(self, input_channels=3, initial_filters=64, device=None):
        super(UNetWithClassification, self).__init__()
        self.initial_filters = initial_filters
        self.device = device if device is not None else torch.device('cpu')
        
        # Encoder (Contracting Path)
        # Block 1
        self.enc1 = self._conv_block(input_channels, initial_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.enc2 = self._conv_block(initial_filters, initial_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.enc3 = self._conv_block(initial_filters * 2, initial_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Block 4
        self.enc4 = self._conv_block(initial_filters * 4, initial_filters * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = self._conv_block(initial_filters * 8, initial_filters * 16)
        
        # Decoder (Expanding Path)
        # Block 4
        self.up4 = self._upconv_block(initial_filters * 16, initial_filters * 8)
        
        # Block 3
        self.up3 = self._upconv_block(initial_filters * 8, initial_filters * 4)
        
        # Block 2
        self.up2 = self._upconv_block(initial_filters * 4, initial_filters * 2)
        
        # Block 1
        self.up1 = self._upconv_block(initial_filters * 2, initial_filters)
        
        # Segmentation output
        self.seg_output = nn.Conv2d(initial_filters, 1, kernel_size=1)
        
        # Classification branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(initial_filters * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        
    def _conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block.to(self.device)
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            self._conv_block(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool4(enc4))
        
        # Decoder
        # Block 4
        up4 = self.up4(bridge)
        dec4 = torch.cat([up4, enc4], dim=1)
        conv_block4 = self._conv_block(dec4.size(1), self.initial_filters * 8).to(self.device)
        dec4 = conv_block4(dec4)
        
        # Block 3
        up3 = self.up3(dec4)
        dec3 = torch.cat([up3, enc3], dim=1)
        conv_block3 = self._conv_block(dec3.size(1), self.initial_filters * 4).to(self.device)
        dec3 = conv_block3(dec3)
        
        # Block 2
        up2 = self.up2(dec3)
        dec2 = torch.cat([up2, enc2], dim=1)
        conv_block2 = self._conv_block(dec2.size(1), self.initial_filters * 2).to(self.device)
        dec2 = conv_block2(dec2)
        
        # Block 1
        up1 = self.up1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        conv_block1 = self._conv_block(dec1.size(1), self.initial_filters).to(self.device)
        dec1 = conv_block1(dec1)
        
        # Segmentation output
        seg_output = torch.sigmoid(self.seg_output(dec1))
        
        # Classification output
        gap = self.gap(bridge)
        gap = gap.view(gap.size(0), -1)
        fc1 = F.relu(self.fc1(gap))
        fc1 = self.dropout(fc1)
        class_output = torch.sigmoid(self.fc2(fc1))
        
        return seg_output, class_output

def get_model(device=None):
    # Determine device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"\nModel Initialization:")
    print(f"- Using device: {device}")
    
    # Create model with device
    model = UNetWithClassification(device=device)
    model = model.to(device)
    
    print("- Model created")
    
    # Debug: Check parameter device types
    print("\nChecking parameter devices:")
    for name, param in model.named_parameters():
        print(f"- {name}: {param.device}, {param.dtype}")
        param.data = param.data.to(device)
    
    # Debug: Check buffer devices
    print("\nChecking buffer devices:")
    for name, buffer in model.named_buffers():
        print(f"- {name}: {buffer.device}, {buffer.dtype}")
        buffer.data = buffer.data.to(device)
    
    # Create and check dummy input
    print("\nTesting with dummy input:")
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    print(f"- Dummy input device: {dummy_input.device}, {dummy_input.dtype}")
    
    try:
        _ = model(dummy_input)
        print("- Forward pass successful")
    except Exception as e:
        print(f"- Forward pass failed: {str(e)}")
        raise
    
    return model

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    print(model) 
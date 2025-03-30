import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithClassification(nn.Module):
    def __init__(self, input_channels=3, initial_filters=64):
        super(UNetWithClassification, self).__init__()
        self.initial_filters = initial_filters
        
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
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
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
        up4 = self.up4(bridge)
        up4 = torch.cat([up4, enc4], dim=1)
        up4 = self._conv_block(up4.size(1), up4.size(1)//2).to(x.device)(up4)
        
        up3 = self.up3(up4)
        up3 = torch.cat([up3, enc3], dim=1)
        up3 = self._conv_block(up3.size(1), up3.size(1)//2).to(x.device)(up3)
        
        up2 = self.up2(up3)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self._conv_block(up2.size(1), up2.size(1)//2).to(x.device)(up2)
        
        up1 = self.up1(up2)
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self._conv_block(up1.size(1), up1.size(1)//2).to(x.device)(up1)
        
        # Segmentation output
        seg_output = torch.sigmoid(self.seg_output(up1))
        
        # Classification output
        gap = self.gap(bridge).view(bridge.size(0), -1)
        fc1 = F.relu(self.fc1(gap))
        fc1 = self.dropout(fc1)
        class_output = torch.sigmoid(self.fc2(fc1))
        
        return seg_output, class_output

def get_model(device='cuda'):
    model = UNetWithClassification()
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    print(model) 
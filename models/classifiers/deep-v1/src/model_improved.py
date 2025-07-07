# Copyright 2025 Brandon Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        
        return out


class ConvBlock(nn.Module):
    """Convolutional block with multiple 3x3 convolutions."""
    
    def __init__(self, in_channels, out_channels, num_convs=2, use_residual=False):
        super(ConvBlock, self).__init__()
        
        self.use_residual = use_residual
        layers = []
        
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_block = nn.Sequential(*layers)
        
        # Optional residual connection
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        elif use_residual:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            out = out + self.shortcut(x)
        return out


class DeepLearningV1Improved(nn.Module):
    """Improved CNN architecture with 3x3 kernels and residual connections."""
    
    def __init__(self, num_classes=70, input_channels=3, dropout_rate=0.5, use_residual=True):
        super(DeepLearningV1Improved, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_residual = use_residual
        
        # Feature extraction layers - using 3x3 kernels throughout
        if use_residual:
            # Residual network architecture
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            self.block1 = ResidualBlock(32, 64, stride=2)
            self.block2 = ResidualBlock(64, 128, stride=2)
            self.block3 = ResidualBlock(128, 256, stride=2)
            self.block4 = ResidualBlock(256, 512, stride=2)
            self.block5 = ResidualBlock(512, 512, stride=2)
        else:
            # Standard convolutional architecture with 3x3 kernels
            self.conv1 = ConvBlock(input_channels, 32, num_convs=2)
            self.block1 = ConvBlock(32, 64, num_convs=2)
            self.block2 = ConvBlock(64, 128, num_convs=2)
            self.block3 = ConvBlock(128, 256, num_convs=2)
            self.block4 = ConvBlock(256, 512, num_convs=2)
            self.block5 = ConvBlock(512, 512, num_convs=2)
            
            # Pooling for non-residual blocks
            self.pool = nn.MaxPool2d(2, 2)
        
        # Global pooling and classification
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head with additional hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input: 128x128x3
        
        if self.use_residual:
            # Residual network forward pass
            x = self.conv1(x)           # 128x128x32
            x = self.block1(x)          # 64x64x64
            x = self.block2(x)          # 32x32x128
            x = self.block3(x)          # 16x16x256
            x = self.block4(x)          # 8x8x512
            x = self.block5(x)          # 4x4x512
        else:
            # Standard network forward pass
            x = self.conv1(x)           # 128x128x32
            x = self.pool(x)            # 64x64x32
            
            x = self.block1(x)          # 64x64x64
            x = self.pool(x)            # 32x32x64
            
            x = self.block2(x)          # 32x32x128
            x = self.pool(x)            # 16x16x128
            
            x = self.block3(x)          # 16x16x256
            x = self.pool(x)            # 8x8x256
            
            x = self.block4(x)          # 8x8x512
            x = self.pool(x)            # 4x4x512
            
            x = self.block5(x)          # 4x4x512
        
        # Global average pooling and classification
        x = self.adaptive_pool(x)       # 1x1x512
        x = x.view(x.size(0), -1)      # Flatten
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x, layer_names=None):
        """Extract feature maps from intermediate layers."""
        features = {}
        
        # Initial convolution
        x = self.conv1(x)
        features['conv1'] = x
        
        # Blocks
        if self.use_residual:
            x = self.block1(x)
            features['block1'] = x
            
            x = self.block2(x)
            features['block2'] = x
            
            x = self.block3(x)
            features['block3'] = x
            
            x = self.block4(x)
            features['block4'] = x
            
            x = self.block5(x)
            features['block5'] = x
        else:
            x = self.pool(x)
            x = self.block1(x)
            features['block1'] = x
            
            x = self.pool(x)
            x = self.block2(x)
            features['block2'] = x
            
            x = self.pool(x)
            x = self.block3(x)
            features['block3'] = x
            
            x = self.pool(x)
            x = self.block4(x)
            features['block4'] = x
            
            x = self.pool(x)
            x = self.block5(x)
            features['block5'] = x
        
        return features if layer_names is None else {k: v for k, v in features.items() if k in layer_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "DeepLearningV1Improved",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "use_residual": self.use_residual,
            "layers": {
                "conv_layers": "5 blocks (2-3 conv per block)",
                "batch_norm_layers": "Multiple per block",
                "pooling_layers": "Stride-2 convs or MaxPool",
                "dropout_layers": 2,
                "fc_layers": 2
            }
        }
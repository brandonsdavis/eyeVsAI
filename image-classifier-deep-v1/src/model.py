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


class DeepLearningV1(nn.Module):
    """Custom CNN architecture for image classification with 128x128 input."""
    
    def __init__(self, num_classes=70, input_channels=3, dropout_rate=0.5):
        super(DeepLearningV1, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Feature extraction layers - optimized for 128x128 input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=6, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=6, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=6, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
        
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
        
        # Block 1: 128x128 -> 64x64
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: 64x64 -> 32x32  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Block 5: 8x8 -> 4x4  
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Global average pooling: 4x4 -> 1x1
        x = self.adaptive_pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x, layer_names=None):
        """Extract feature maps from intermediate layers."""
        features = {}
        
        # Block 1
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        features['block1'] = x1
        
        # Block 2
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        features['block2'] = x2
        
        # Block 3
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        features['block3'] = x3
        
        # Block 4
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))
        features['block4'] = x4
        
        # Block 5
        x5 = self.pool(F.relu(self.bn5(self.conv5(x4))))
        features['block5'] = x5
        
        return features if layer_names is None else {k: v for k, v in features.items() if k in layer_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "DeepLearningV1",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "layers": {
                "conv_layers": 5,
                "batch_norm_layers": 5,
                "pooling_layers": 5,
                "dropout_layers": 1,
                "fc_layers": 1
            }
        }
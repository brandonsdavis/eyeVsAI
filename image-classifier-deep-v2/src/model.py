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
import numpy as np
from typing import Dict, Any


class AttentionBlock(nn.Module):
    """Self-attention mechanism for enhanced feature representation."""
    
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel=7):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        
        # Learnable attention weight
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        
        # Apply channel attention
        x_channel = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        
        # Apply spatial attention with learnable weight
        out = x + self.gamma * (x_channel * spatial_att)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeepLearningV2(nn.Module):
    """Advanced CNN with ResNet architecture, attention mechanisms, and modern techniques."""
    
    def __init__(self, num_classes=1000, input_channels=3, dropout_rates=(0.5, 0.3, 0.2), 
                 attention_reduction=8, spatial_kernel=7, residual_dropout=0.1):
        super(DeepLearningV2, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout_rate=residual_dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_rate=residual_dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_rate=residual_dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_rate=residual_dropout)
        
        # Attention mechanisms
        self.attention1 = AttentionBlock(128, attention_reduction, spatial_kernel)
        self.attention2 = AttentionBlock(256, attention_reduction, spatial_kernel)
        self.attention3 = AttentionBlock(512, attention_reduction, spatial_kernel)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced classifier with LayerNorm for better batch size flexibility
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout_rate=0.1):
        """Create a residual layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, dropout_rate))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.attention1(x)  # Apply attention after layer2
        
        x = self.layer3(x)
        x = self.attention2(x)  # Apply attention after layer3
        
        x = self.layer4(x)
        x = self.attention3(x)  # Apply attention after layer4
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x, layer_names=None):
        """Extract feature maps at different levels for visualization."""
        features = {}
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        x = self.attention1(x)
        features['attention1'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        x = self.attention2(x)
        features['attention2'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        x = self.attention3(x)
        features['attention3'] = x
        
        return features if layer_names is None else {k: v for k, v in features.items() if k in layer_names}
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization."""
        attention_weights = {}
        
        # Forward pass to layer2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Get attention1 weights
        attention_weights['attention1_gamma'] = self.attention1.gamma.item()
        
        x = self.attention1(x)
        x = self.layer3(x)
        
        # Get attention2 weights
        attention_weights['attention2_gamma'] = self.attention2.gamma.item()
        
        x = self.attention2(x)
        x = self.layer4(x)
        
        # Get attention3 weights
        attention_weights['attention3_gamma'] = self.attention3.gamma.item()
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "DeepLearningV2",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "features": [
                "residual_connections", 
                "self_attention", 
                "channel_attention", 
                "spatial_attention",
                "layer_normalization",
                "advanced_dropout"
            ],
            "layers": {
                "conv_layers": 9,  # Initial + 8 residual blocks
                "attention_blocks": 3,
                "batch_norm_layers": 9,
                "layer_norm_layers": 2,
                "dropout_layers": 5,
                "fc_layers": 3
            }
        }
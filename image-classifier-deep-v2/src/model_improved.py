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
from typing import Dict, Any, Optional, Tuple


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        # Channel attention
        self.channel_attention = SqueezeExcitation(in_channels, reduction)
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        
        return x * spatial


class ImprovedResidualBlock(nn.Module):
    """Improved residual block with better normalization and regularization."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, dropout_rate: float = 0.0,
                 use_se: bool = True, reduction: int = 16):
        super(ImprovedResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SqueezeExcitation(out_channels, reduction) if use_se else nn.Identity()
        
        # Shortcut
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE block
        out = self.se(out)
        
        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add and activate
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        
        return out


class DenseBlock(nn.Module):
    """Dense block for better gradient flow."""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, dropout_rate: float = 0.0):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DeepLearningV2Improved(nn.Module):
    """Improved Deep Learning v2 with advanced architecture and optimization techniques."""
    
    def __init__(self, num_classes: int = 1000, input_channels: int = 3,
                 dropout_rates: Tuple[float, float, float] = (0.5, 0.3, 0.2),
                 use_cbam: bool = True, use_se: bool = True,
                 residual_dropout: float = 0.1, architecture: str = "resnet"):
        super(DeepLearningV2Improved, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.architecture = architecture
        
        # Stem layers - lighter initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Build architecture
        if architecture == "resnet":
            self.features = self._build_resnet(64, use_se, residual_dropout)
            final_channels = 512
        elif architecture == "densenet":
            self.features = self._build_densenet(64, residual_dropout)
            final_channels = 512
        else:  # hybrid
            self.features = self._build_hybrid(64, use_se, residual_dropout)
            final_channels = 512
        
        # Attention mechanism
        self.attention = CBAM(final_channels) if use_cbam else nn.Identity()
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Advanced classifier with better regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(final_channels * 2, 1024),  # *2 for concat of avg and max pool
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_resnet(self, in_channels: int, use_se: bool, dropout: float) -> nn.Sequential:
        """Build ResNet-style feature extractor."""
        layers = []
        
        # Layer 1: 64 -> 128
        layers.append(self._make_residual_layer(in_channels, 128, 3, stride=2, 
                                               dropout=dropout, use_se=use_se))
        
        # Layer 2: 128 -> 256
        layers.append(self._make_residual_layer(128, 256, 4, stride=2,
                                               dropout=dropout, use_se=use_se))
        
        # Layer 3: 256 -> 512
        layers.append(self._make_residual_layer(256, 512, 6, stride=2,
                                               dropout=dropout, use_se=use_se))
        
        # Layer 4: 512 -> 512 (no stride)
        layers.append(self._make_residual_layer(512, 512, 3, stride=1,
                                               dropout=dropout, use_se=use_se))
        
        return nn.Sequential(*layers)
    
    def _build_densenet(self, in_channels: int, dropout: float) -> nn.Sequential:
        """Build DenseNet-style feature extractor."""
        layers = []
        channels = in_channels
        
        # Dense block 1
        layers.append(DenseBlock(channels, growth_rate=32, num_layers=6, dropout_rate=dropout))
        channels += 32 * 6
        layers.append(self._make_transition(channels, channels // 2))
        channels = channels // 2
        
        # Dense block 2
        layers.append(DenseBlock(channels, growth_rate=32, num_layers=12, dropout_rate=dropout))
        channels += 32 * 12
        layers.append(self._make_transition(channels, channels // 2))
        channels = channels // 2
        
        # Dense block 3
        layers.append(DenseBlock(channels, growth_rate=32, num_layers=24, dropout_rate=dropout))
        channels += 32 * 24
        layers.append(self._make_transition(channels, channels // 2))
        channels = channels // 2
        
        # Dense block 4
        layers.append(DenseBlock(channels, growth_rate=32, num_layers=16, dropout_rate=dropout))
        channels += 32 * 16
        
        # Final transition to 512 channels
        layers.append(nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 512, kernel_size=1, bias=False)
        ))
        
        return nn.Sequential(*layers)
    
    def _build_hybrid(self, in_channels: int, use_se: bool, dropout: float) -> nn.Sequential:
        """Build hybrid architecture combining ResNet and DenseNet."""
        layers = []
        
        # ResNet-style layers
        layers.append(self._make_residual_layer(in_channels, 128, 2, stride=2,
                                               dropout=dropout, use_se=use_se))
        
        # Dense block
        layers.append(DenseBlock(128, growth_rate=32, num_layers=6, dropout_rate=dropout))
        channels = 128 + 32 * 6
        layers.append(self._make_transition(channels, 256))
        
        # More ResNet layers
        layers.append(self._make_residual_layer(256, 512, 3, stride=2,
                                               dropout=dropout, use_se=use_se))
        
        return nn.Sequential(*layers)
    
    def _make_residual_layer(self, in_channels: int, out_channels: int, blocks: int,
                            stride: int = 1, dropout: float = 0.0, use_se: bool = True) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ImprovedResidualBlock(in_channels, out_channels, stride,
                                          downsample, dropout, use_se))
        
        for _ in range(1, blocks):
            layers.append(ImprovedResidualBlock(out_channels, out_channels,
                                              dropout_rate=dropout, use_se=use_se))
        
        return nn.Sequential(*layers)
    
    def _make_transition(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Transition layer for DenseNet."""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        """Initialize network weights with improved schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # Features
        x = self.features(x)
        
        # Attention
        x = self.attention(x)
        
        # Global pooling (concat avg and max for richer representation)
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """Extract feature maps at different levels."""
        features = {}
        
        # Stem
        x = self.stem(x)
        features['stem'] = x
        
        # Features with intermediate outputs
        for i, layer in enumerate(self.features):
            x = layer(x)
            features[f'layer_{i}'] = x
        
        # Attention
        x = self.attention(x)
        features['attention'] = x
        
        return features if layer_names is None else {k: v for k, v in features.items() if k in layer_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": f"DeepLearningV2Improved-{self.architecture}",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "features": [
                "improved_residual_blocks",
                "squeeze_excitation" if hasattr(self, 'se') else None,
                "cbam_attention",
                "batch_normalization",
                "dual_pooling",
                "architecture_type: " + self.architecture
            ],
            "memory_efficiency": "High - uses checkpoint gradients if enabled"
        }
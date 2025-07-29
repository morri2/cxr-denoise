import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEModule(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out)  # Apply SE attention
        
        out += residual
        out = self.relu(out)
        
        return out

class AttentionGate(nn.Module):
    """Attention Gate for skip connections"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super(AttentionGate, self).__init__()
        
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.attention_conv = nn.Conv2d(inter_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.bn_gate = nn.BatchNorm2d(inter_channels)
        self.bn_skip = nn.BatchNorm2d(inter_channels)
    
    def forward(self, gate, skip):
        # Ensure gate and skip have same spatial dimensions
        if gate.size()[2:] != skip.size()[2:]:
            gate = F.interpolate(gate, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        gate_conv = self.bn_gate(self.gate_conv(gate))
        skip_conv = self.bn_skip(self.skip_conv(skip))
        
        attention = self.relu(gate_conv + skip_conv)
        attention = self.attention_conv(attention)
        attention = self.sigmoid(attention)
        
        return skip * attention

class AttentionResUNet(nn.Module):
    """Attention ResUNet for Image Denoising"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(AttentionResUNet, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path with residual blocks
        self.encoder1 = ResidualBlock(base_channels, base_channels)
        self.encoder2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.encoder3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.encoder4 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)
        
        # Bottleneck with adaptive average pooling
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 16, stride=2),
            nn.AdaptiveAvgPool2d((8, 8)),  # Adaptive pooling as shown in diagram
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )
        
        # Attention gates for skip connections
        self.att_gate1 = AttentionGate(base_channels * 16, base_channels * 8, base_channels * 4)
        self.att_gate2 = AttentionGate(base_channels * 8, base_channels * 4, base_channels * 2)
        self.att_gate3 = AttentionGate(base_channels * 4, base_channels * 2, base_channels)
        self.att_gate4 = AttentionGate(base_channels * 2, base_channels, base_channels // 2)
        
        # Decoder path with upsampling
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        self.decoder1_conv = ResidualBlock(base_channels * 16, base_channels * 8)  # 8 + 8 from skip
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.decoder2_conv = ResidualBlock(base_channels * 8, base_channels * 4)  # 4 + 4 from skip
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder3_conv = ResidualBlock(base_channels * 4, base_channels * 2)  # 2 + 2 from skip
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder4_conv = ResidualBlock(base_channels * 2, base_channels)  # 1 + 1 from skip
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 1, 1, 0)
        )
        
        # Residual connection from input to output (subtract operation in diagram)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # Store input for residual connection
        input_residual = x
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder path
        enc1 = self.encoder1(x)      # 64 channels
        enc2 = self.encoder2(enc1)   # 128 channels, 1/2 size
        enc3 = self.encoder3(enc2)   # 256 channels, 1/4 size
        enc4 = self.encoder4(enc3)   # 512 channels, 1/8 size
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 1024 channels, 1/16 size
        
        # Decoder path with attention gates
        dec1 = self.decoder1(bottleneck)  # Upsample to 1/8 size
        att1 = self.att_gate1(dec1, enc4)  # Apply attention to skip connection
        dec1 = torch.cat([dec1, att1], dim=1)  # Concatenate
        dec1 = self.decoder1_conv(dec1)
        
        dec2 = self.decoder2(dec1)  # Upsample to 1/4 size
        att2 = self.att_gate2(dec2, enc3)
        dec2 = torch.cat([dec2, att2], dim=1)
        dec2 = self.decoder2_conv(dec2)
        
        dec3 = self.decoder3(dec2)  # Upsample to 1/2 size
        att3 = self.att_gate3(dec3, enc2)
        dec3 = torch.cat([dec3, att3], dim=1)
        dec3 = self.decoder3_conv(dec3)
        
        dec4 = self.decoder4(dec3)  # Upsample to original size
        att4 = self.att_gate4(dec4, enc1)
        dec4 = torch.cat([dec4, att4], dim=1)
        dec4 = self.decoder4_conv(dec4)
        
        # Final convolution to get denoised output
        denoised = self.final_conv(dec4)
        
        # Residual connection: output = input - noise (subtract operation in diagram)
        if isinstance(self.residual_conv, nn.Identity):
            output = input_residual - denoised
        else:
            output = self.residual_conv(input_residual) - denoised
        
        return output

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = AttentionResUNet(in_channels=1, out_channels=1, base_channels=64)

    
    # Test with random input
    x = torch.randn(1, 1, 256, 256)  # Batch=1, Channels=1, Height=256, Width=256

    import torchinfo

    #torchinfo.summary(model, (1,1,256, 256))
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with different input sizes
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    for h, w in test_sizes:
        test_input = torch.randn(1, 1, h, w)
        try:
            test_output = model(test_input)
            print(f"✓ Input {h}x{w} -> Output {test_output.shape[2]}x{test_output.shape[3]}")
        except Exception as e:
            print(f"✗ Input {h}x{w} failed: {e}")

    

# A UNet with attention and residual blocks.

import torch
import torch.nn as nn
import torchxrayvision as xrv


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    Applies channel-wise attention to input features.
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze
        se_weight = self.avg_pool(x)
        se_weight = self.fc(se_weight)
        # Excitation
        return x * se_weight


class ConvBlock(nn.Module):
    """
    Residual convolutional block: two conv layers + BN + ReLU,
    followed by an optional SE block and a residual connection.
    """
    def __init__(self, in_ch, out_ch, use_se=True, se_reduction=16):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_ch, reduction=se_reduction)
        # If input/output channels differ, use a 1x1 conv for residual
        self.residual_conv = None
        if in_ch != out_ch:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.use_se:
            out = self.se(out)
        if self.residual_conv:
            identity = self.residual_conv(identity)
        out += identity
        return out


class UNetSE(nn.Module):
    """
    UNet model with Squeeze-and-Excitation (SE) blocks.
    Input: 1-channel image (e.g., CXR) of size HxW.
    Output: 1-channel denoised image.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], se_reduction=16):
        if type(features) is int:
            features = [features, features * 2, features * 4, features * 8]

        super(UNetSE, self).__init__()
        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feat in features:
            self.downs.append(ConvBlock(in_channels, feat, use_se=True, se_reduction=se_reduction))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, use_se=True, se_reduction=se_reduction)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = features[::-1]
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(feat * 2, feat, use_se=True, se_reduction=se_reduction))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pathway
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # Decoder pathway
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip = skip_connections[idx]
            # If needed, pad x to match skip size
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.up_convs[idx](x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick unit test
    model = UNetSE(in_channels=1, out_channels=1, features=32)
    x = torch.randn((1, 1, 1024, 1024))
    preds = model(x)
    print(f"Input shape: {x.shape}, Output shape: {preds.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    #print(torchinfo.summary(model, input_size=(1, 1, 1024, 1024), col_names=["input_size", "output_size", "num_params"]))

    import matplotlib.pyplot as plt
    plt.imshow(preds[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.show()
    import torchinfo
    print(torchinfo.summary(model, input_size=(8, 1, 256, 256)))


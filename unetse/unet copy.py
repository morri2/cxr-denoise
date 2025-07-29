import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class EncoderStep(nn.Module):
    """Encodes, Halves width and height"""
    def __init__(self, in_channels, out_channels):
        super(EncoderStep, self).__init__()

        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        """returns the (<Pooled Output>, <Unpooled/Skip Output>)"""
        skip_con = self.doubleconv(x)
        y = self.pool(skip_con)
        return y, skip_con

class DecoderStep(nn.Module):
    """Decodes (output is 2x width and height of input, but equal to skip_con)"""
    def __init__(self, in_channels, out_channels):
        super(DecoderStep, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(out_channels * 2, out_channels)
    
    def forward(self, x, skip_con):
        """takes both x and skip"""
        x = self.upconv(x)
        x = torch.cat((skip_con, x), dim=1)
        # TODO: handle non-clean concats (either with output_padding or interpolation)
        y = self.doubleconv(x)
        return y


class UNet(nn.Module):
    
    """Custom UNet"""
    def __init__(self, layers=4, base_channels=64):
        super(UNet, self).__init__()

        
        self.base_channels = base_channels
        
        self.encoder_steps = nn.ModuleList(
            [EncoderStep(
                1 if i == 0 else ( base_channels * 2 ** (i-1) ), 
                base_channels * 2 ** i
                ) for i in range(layers) ],
        )

        self.bottleneck = DoubleConv(
            base_channels * 2 ** (layers-1), 
            base_channels * 2 ** layers
            )
        
        self.decoder_steps = nn.ModuleList(
            [DecoderStep(
                base_channels * 2 ** i, 
                base_channels * 2 ** (i-1)
                ) for i in range(layers, 0, -1) ],
        )
        
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    

    def name(self):
        return "unet-s{}k{}".format(self.base_channels)

    def forward(self, x):
        skip_cons = []

        # encoder
        for encoder_step in self.encoder_steps:
            x, skip_con = encoder_step(x)
            skip_cons.append(skip_con)
        
        x = self.bottleneck(x)

        # decoder
        skip_cons = skip_cons[::-1]
        for i, decoder_step in enumerate( self.decoder_steps ):
            x = decoder_step(x, skip_cons[i])

        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

# Example usage
if __name__ == "__main__":
    import torchinfo

    model = UNet(layers=4, base_channels=64
                 )
    x = torch.randn((1, 1, 512, 512))  # Example input tensor (Batch, Width, Height)
    t = time.time()
    out = model(x)

    torchinfo.summary(model, (1,1,512,512))

    print("time =", time.time() - t)
    print("in shape", out.shape)  # Should be (1, 256, 256)
    print("out shape", x.shape)
    #import matplotlib.pyplot as plt
    #plt.imshow(out.detach().numpy(). squeeze() , cmap="gray")
    #plt.show()
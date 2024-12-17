import torch
from torch import nn
from einops import rearrange

from backbones.mambaroll.ssm import SSM
from backbones.transforms import DataConsistencyKspace, DataConsistencySinogram


class ShuffledSSM(nn.Module):
    def __init__(
        self,
        in_channels,
        shuffle_factor=4,
        d_state=64
    ):
        super().__init__()
        self.shuffle_factor = shuffle_factor
        self.shuffle = nn.PixelShuffle(self.shuffle_factor)
        self.unshuffle = nn.PixelUnshuffle(self.shuffle_factor)
        self.norm = nn.LayerNorm(in_channels*self.shuffle_factor**2)
        self.ssm = SSM(
            d_model=in_channels*self.shuffle_factor**2,
            d_state=d_state
        )

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = self.unshuffle(x)

        # Spatial to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # SSM forward computation
        x = self.norm(x)
        x = self.ssm(x)
        
        # Sequence to spatial
        x = rearrange(
            x, 'b (h w) c -> b c h w',
            h=H//self.shuffle_factor,
            w=W//self.shuffle_factor,
            c=C*self.shuffle_factor**2
        )

        out = self.shuffle(x)

        return out + res


class PSSM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        scale,
        shuffle_factor,
        d_state
    ):
        super().__init__()

        if scale == 0.25:
            kernel, stride = 6, 4
        elif scale == 0.5:
            kernel, stride = 4, 2
        elif scale == 1:
            kernel, stride = 3, 1
        else:
            raise ValueError("Invalid scale. Must be one of [0.25, 0.5, 1]")
        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=model_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=model_channels,
                kernel_size=kernel,
                stride=stride,
                padding=1
            ),
            nn.SiLU(inplace=True)
        )

        self.ssm = ShuffledSSM(
            in_channels=model_channels,
            shuffle_factor=shuffle_factor,
            d_state=d_state
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=model_channels,
                out_channels=model_channels,
                kernel_size=kernel,
                stride=stride,
                padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=model_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x, dc_layer, **kwargs):
        x = self.encoder(x)
        x = self.ssm(x)
        x = self.decoder(x)
        x_dc = dc_layer(x, **kwargs)
        out = torch.cat((x, x_dc), dim=1)
        return out
   

class MambaRoll(nn.Module):
    def __init__(
        self,
        nroll,
        in_channels, 
        out_channels,
        model_channels,
        scales,
        shuffle_factors,
        d_state
    ):
        super(MambaRoll, self).__init__()
        self.nroll = nroll

        self.pssm1 = PSSM(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels[0],
            scale=scales[0],
            shuffle_factor=shuffle_factors[0],
            d_state=d_state
        )

        self.pssm2 = PSSM(
            in_channels=in_channels*2,
            out_channels=out_channels,
            model_channels=model_channels[1],
            scale=scales[1],
            shuffle_factor=shuffle_factors[1],
            d_state=d_state
        )

        self.pssm3 = PSSM(
            in_channels=in_channels*4,
            out_channels=out_channels,
            model_channels=model_channels[2],
            scale=scales[2],
            shuffle_factor=shuffle_factors[2],
            d_state=d_state
        )        

        self.refinement = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels*2,
                out_channels=model_channels[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=model_channels[2],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )


class MambaRollMRI(MambaRoll):
    def __init__(self, **params):
        super().__init__(**params)
        self.dc = DataConsistencyKspace()

    def forward(self, x, target, mask, coilmap):
        B, C, H, W = x.shape
        for _ in range(self.nroll):
            x1 = self.pssm1(
                x,
                dc_layer=self.dc,
                target=target,
                mask=mask,
                coilmap=coilmap
            )
            
            x2 = self.pssm2(
                x1,
                dc_layer=self.dc,
                target=target,
                mask=mask,
                coilmap=coilmap
            )
            
            x3 = self.pssm3(
                torch.cat((x1, x2), dim=1),
                dc_layer=self.dc,
                target=target,
                mask=mask,
                coilmap=coilmap
            )
            
            x = self.refinement(x3)
            x = self.dc(x, target, mask, coilmap)
            
        # Extract scale-specific decoder outputs for the last cascade
        decoder_out = torch.cat((x1[:,2:], x2[:,2:], x3[:,2:]), dim=1)

        return x, decoder_out


class MambaRollCT(MambaRoll):
    def __init__(self, **params):
        super().__init__(**params)
        self.dc = None

    def forward(self, x, s_target, theta, us_factor):
        # Create DC layer if does not exist
        if self.dc is None:
            self.dc = DataConsistencySinogram(
                image_size=x.shape[-1],
                theta=theta,
                us_factor=us_factor,
                device=x.device
            )

        for _ in range(self.nroll):
            x1 = self.pssm1(x, dc_layer=self.dc, s_target=s_target)
            x2 = self.pssm2(x1, dc_layer=self.dc, s_target=s_target)
            x3 = self.pssm3(torch.cat((x1, x2), dim=1), dc_layer=self.dc, s_target=s_target)
            x = self.refinement(x3)

        # Extract scale-specific decoder outputs for the last cascade
        decoder_out = torch.cat((x1[:,1:], x2[:,1:], x3[:,1:]), dim=1)

        return x, decoder_out

import torch
from torch import nn
from einops import rearrange

from backbones.mambaroll.ssm import SSM
from backbones.transforms import DataConsistencyKspace, DataConsistencySinogram


class Identity(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x


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
        d_state,
        dropout
    ):
        super().__init__()

        if scale == 0.0625:
            kernel, stride = 18, 16
        elif scale == 0.125:
            kernel, stride = 10, 8
        elif scale == 0.25:
            kernel, stride = 6, 4
        elif scale == 0.5:
            kernel, stride = 4, 2
        elif scale == 1:
            kernel, stride = 3, 1
        else:
            raise ValueError("Invalid scale. Must be one of [0.0625, 0.125, 0.25, 0.5, 1]")
        
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
            nn.Dropout2d(dropout),
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
        d_state,
        dropout
    ):
        super(MambaRoll, self).__init__()
        self.nroll = nroll
        self.scales = scales

        self.pssms = nn.ModuleList()
        for i in range(len(scales)):
            self.pssms.append(
                PSSM(
                    in_channels=in_channels if i == 0 else (2*i+1)*in_channels,
                    out_channels=out_channels,
                    model_channels=model_channels[i],
                    scale=scales[i],
                    shuffle_factor=shuffle_factors[i],
                    d_state=d_state,
                    dropout=dropout
                )
            )

        self.refinement = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels*2,
                out_channels=model_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=model_channels[-1],
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
        for _ in range(self.nroll):
            decoder_out = []
            for i, pssm in enumerate(self.pssms):
                out = pssm(
                    x,
                    dc_layer=self.dc,
                    target=target,
                    mask=mask,
                    coilmap=coilmap
                )
                
                # Aggregate outputs from all scales
                x = torch.cat((x, out), dim=1)
                
                # Aggregate decoder outputs for autoregressive loss
                decoder_out.append(out[:,2:])
            
            x = self.refinement(out)
            x = self.dc(x, target, mask, coilmap)
            
        # Extract scale-specific decoder outputs for the last cascade
        decoder_out = torch.cat(decoder_out, dim=1)

        return x, decoder_out


class MambaRollCT(MambaRoll):
    def __init__(self, **params):
        super().__init__(**params)
        self.dc = None
        self.identity = Identity()

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
            # x1 = self.pssm1(x, dc_layer=self.dc, s_target=s_target)
            decoder_out = []
            for i, pssm in enumerate(self.pssms):
                if i == 0:
                    out = pssm(x, dc_layer=self.dc, s_target=s_target)
                else:
                    out = pssm(x, dc_layer=self.identity, s_target=s_target)
                
                # Aggregate outputs from all scales
                x = torch.cat((x, out), dim=1)

                # Aggregate decoder outputs for autoregressive loss
                decoder_out.append(out[:,1:])

            x = self.refinement(out)

        # Extract scale-specific decoder outputs for the last cascade
        decoder_out = torch.cat(decoder_out, dim=1)

        return x, decoder_out
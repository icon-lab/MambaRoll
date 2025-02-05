import torch
from torch import nn
from backbones.radon import IRadon

try:
    from torch_radon import Radon
except ImportError:
    from backbones.radon import Radon


def fft2c(x, dim=(-2, -1)):
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fft2(x, dim=dim)
    return torch.fft.fftshift(x, dim=dim)


def ifft2c(x, dim=(-2, -1)):
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifft2(x, dim=dim)
    return torch.fft.fftshift(x, dim=dim)


class DataConsistencyKspace(nn.Module):
    """ Data consistency layer in k-space. """

    def __init__(self):
        super().__init__()

    def forward(self, source, target, mask, coilmap):
        source = torch.complex(
            source[:, :source.shape[1]//2],
            source[:, source.shape[1]//2:]
        )

        # Forward projection
        source = source * coilmap

        # Fourier transform
        source_kspace = fft2c(source)
        target_kspace = fft2c(target)

        # Fill in k-space
        dc_kspace = (1-mask) * source_kspace + mask * target_kspace

        # Inverse Fourier transform
        out = ifft2c(dc_kspace)

        # Coil combined image
        out = (torch.conj(coilmap) * out).sum(axis=1, keepdim=True)

        # Stack real and imaginary parts
        out = torch.cat((out.real, out.imag), dim=1)

        return out


class DataConsistencySinogram(nn.Module):
    """ Data consistency layer in sinogram. """

    def __init__(self, image_size, theta, us_factor, device):
        super().__init__()

        # Create full-sampled angles
        theta = theta[0]
        theta_fs = []
        for i in range(len(theta)-1):
            theta_fs.extend(torch.linspace(theta[i], theta[i+1], us_factor+1)[:-1])

        # Add last angle
        theta_fs = torch.tensor(theta_fs + [theta[-1]]).to(device)

        self.mask = torch.zeros_like(theta_fs).bool()
        self.mask[torch.isin(theta_fs, theta)] = 1

        self.radon = Radon(
            image_size,
            theta_fs*torch.pi/180,
            det_count=int(image_size*2**0.5 + 0.5)
        )
        
        self.iradon = IRadon(
            image_size,
            theta_fs.to(device),
            circle=False,
            device=device
        )

    def forward(self, x_source, s_target):
        s_source = self.radon.forward(x_source).transpose(-1,-2)
        s_source[...,self.mask] = s_target - s_source[...,self.mask]
        x_source = self.iradon(s_source)
        return x_source

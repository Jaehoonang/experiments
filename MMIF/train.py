import torch
from models.wavelet_transforms import get_wavelet_transform
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np

if __name__=="__main__":
    xfm = DWTForward(J=1, mode="periodization", wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode="periodization", wave='haar')
    X = torch.randn(1, 3, 512, 512)

    Yl, Yh = xfm(X)
    print(Yl.shape)

    inverse = ifm((Yl, Yh))
    print(inverse.shape)


    # print(np.testing.assert_array_almost_equal(
    #     inverse.cpu().numpy(), X.cpu().numpy()
    # ))

    # print(inverse == X)
    # wt = get_wavelet_transform(backend="pytorch", wave_type="haar", device="cuda")
    # img = torch.randn(1, 3, 256, 256).cuda()



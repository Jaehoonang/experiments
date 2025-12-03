import torch
from models.wavelet_transforms import get_wavelet_transform
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
from data.dataset import ex_data
import matplotlib.pyplot as plt
from torchvision import transforms


if __name__=="__main__":
    x_visible = ex_data(root_dir = r"C:\Users\12wkd\Desktop\mri2.png")
    x_infra = ex_data(root_dir= r"C:\Users\12wkd\Desktop\pet2.png")

    xfm = DWTForward(J=1, mode="periodization", wave='haar')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode="periodization", wave='haar')

    Yl_visible, Yh_visible = xfm(x_visible)
    Yl_infrared, Yh_infrared = xfm(x_infra)

    difference_Yl = Yl_visible - Yl_infrared
    difference_Yh = [Yh_visible[i] - Yh_infrared[i] for i in range(len(Yh_visible))]

    threshold = 0.7
    masked = (np.abs(difference_Yl) >= threshold)
    print(masked)
    mask_yl = difference_Yl * masked
    print(mask_yl)
    print(mask_yl.shape)
    print(difference_Yl.shape)

    difference_Yh_mask = []
    for i in range(len(Yh_visible)):
        diff = Yh_visible[i] - Yh_infrared[i]
        mask = (diff >= threshold)
        diff = diff * mask
        difference_Yh.append(diff)

    inverse = ifm((mask_yl, difference_Yh_mask))

    inverse_img = inverse.detach().cpu().numpy()
    plt.imshow(inverse_img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

    # Yl_img = Yl.squeeze().detach().cpu().numpy()  # (H/2, W/2)
    # Yh_img = Yh[0].squeeze().detach().cpu().numpy()  # (3, H/2, W/2)
    #
    # titles = ['Yl (Low freq)', 'Yh - LH', 'Yh - HL', 'Yh - HH']
    #
    # # --- 그리기 ---
    # plt.figure(figsize=(14, 4))
    #
    # # 1) Yl (저주파)
    # plt.subplot(1, 4, 1)
    # plt.imshow(Yl_img, cmap='gray')
    # plt.title(titles[0])
    # plt.axis('off')
    #
    # # 2) Yh의 3 성분 (고주파)
    # for i in range(3):
    #     plt.subplot(1, 4, i + 2)
    #     plt.imshow(Yh_img[i], cmap='gray')
    #     plt.title(titles[i + 1])
    #     plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    # print(np.testing.assert_array_almost_equal(
    #     inverse.cpu().numpy(), X.cpu().numpy()
    # ))

    # print(inverse == X)
    # wt = get_wavelet_transform(backend="pytorch", wave_type="haar", device="cuda")
    # img = torch.randn(1, 3, 256, 256).cuda()



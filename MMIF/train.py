import torch
from dateutil.tz.win import valuestodict
import numpy as np
from models.wavelet_transforms import get_wavelet_transform
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
from data.dataset import ex_data
import matplotlib.pyplot as plt
from torchvision import transforms



# transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])



if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_visible = ex_data(root_dir = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\visible\00025N.png")
    x_infra = ex_data(root_dir= r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared\00025N.png")

    xfm = DWTForward(J=1, mode="periodization", wave='haar').to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode="periodization", wave='haar').to(device)

    print(x_visible.shape)
    Yl_visible, Yh_visible = xfm(x_visible)
    print('visible shape', Yl_visible.shape),

    Yl_infrared, Yh_infrared = xfm(x_infra)
    x= torch.concat([Yl_visible, Yl_infrared], dim=1)
    print(x.shape)

    # print(Yh_visible)
    difference_Yl = torch.abs(Yl_visible - Yl_infrared)
    difference_Yh = [Yh_visible[i] - Yh_infrared[i] for i in range(len(Yh_visible))]

    ratio = 0.3
    values, indices = torch.sort(difference_Yl.view(-1))
    k = int((1 - ratio) * len(values))
    threshold = values[k]
    print('len values:', len(values))
    print('K', k)
    print('threshold', threshold)

    mask = (difference_Yl >= threshold).float()

    print(values, indices)
    print(difference_Yl * mask)
    #
    # print(difference_Yl.shape)
    # print(values, indices)


    # threshold = 0.7
    masked = (difference_Yl) >= threshold
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

    # inverse = ifm((mask_yl, difference_Yh_mask))
    #
    # inverse_img = inverse.detach().cpu().numpy()
    # plt.imshow(inverse_img.squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.show()

    Yl_img = Yl_infrared.squeeze().detach().cpu().numpy()  # (H/2, W/2)
    Yh_img = Yh_infrared[0].squeeze().detach().cpu().numpy()  # (3, H/2, W/2)

    print(Yh_img.shape)
    titles = ['Yl (Low freq)', 'Yh - LH', 'Yh - HL', 'Yh - HH']

    # --- 그리기 ---
    plt.figure(figsize=(14, 4))

    # 1) Yl (저주파)
    plt.subplot(1, 4, 1)
    plt.imshow(Yl_img, cmap='gray')
    plt.title(titles[0])
    plt.axis('off')

    # 2) Yh의 3 성분 (고주파)
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        plt.imshow(Yh_img[i], cmap='gray')
        plt.title(titles[i + 1])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # print(np.testing.assert_array_almost_equal(
    #     inverse.cpu().numpy(), X.cpu().numpy()
    # ))

    # print(inverse == X)
    # wt = get_wavelet_transform(backend="pytorch", wave_type="haar", device="cuda")
    # img = torch.randn(1, 3, 256, 256).cuda()



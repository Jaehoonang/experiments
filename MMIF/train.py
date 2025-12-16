import torch
from dateutil.tz.win import valuestodict
import numpy as np
from models.wavelet_transforms import get_wavelet_transform
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
from data.dataset import ex_data
import matplotlib.pyplot as plt
from torchvision import transforms

import xml.etree.ElementTree as ET

# transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])

def pares_annotation(self, xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def create_annotation_mask(self, H, W, boxes):
    mask = torch.zeros((H, W), device=self.device)
    for (xmin, ymin, xmax, ymax) in boxes:
        mask[ymin:ymax, xmin:xmax] = 1.0
    return mask

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_visible = ex_data(root_dir = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\visible\00025N.png")
    x_infra = ex_data(root_dir= r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared\00025N.png")

    xfm = DWTForward(J=1, mode="periodization", wave='haar').to(device)
    ifm = DWTInverse(mode="periodization", wave='haar').to(device)

    # print(x_visible.shape)
    Yl_visible, Yh_visible = xfm(x_visible)
    # print('visible shape', Yl_visible.shape),

    Yl_infrared, Yh_infrared = xfm(x_infra)
    x= torch.concat([Yl_visible, Yl_infrared], dim=1)
    # print(x.shape)

    # print(Yh_visible)
    difference_Yl = torch.abs(Yl_visible - Yl_infrared)
    # difference_Yh =[Yh_visible[i] - Yh_infrared[i] for i in range(len(Yh_visible))]
    difference_Yh = torch.stack([torch.abs(Yh_visible[i] - Yh_infrared[i]) for i in range(len(Yh_visible))])

    ratio = 0.3
    values1, indices1 = torch.sort(difference_Yl.view(-1))
    k1 = int((1 - ratio) * len(values1))
    threshold1 = values1[k1]

    high_mask = torch.zeros_like(difference_Yh)
    for c in range(difference_Yh.size(0)):
        vals = torch.sort(difference_Yh[c].view(-1))[0]
        k2 = int(ratio * len(vals))
        threshold2 = vals[k2]
        high_mask[c] = (difference_Yh[c] >= threshold2).float()
    # print('len values:', len(values))
    # print('K', k)
    # print('threshold', threshold)

    # mask = (difference_Yl >= threshold).float()

    # print(values, indices)
    # print(difference_Yl * mask)
    # print(difference_Yl.shape)
    # print(values, indices)

    # threshold = 0.7
    masked1 = (difference_Yl) >= threshold1
    masked2 = (difference_Yh) >= high_mask

    ll_img = difference_Yl.detach().cpu().squeeze().numpy()
    ll_mask_np = masked1.detach().cpu().squeeze().numpy()

    high_img = difference_Yh.detach().cpu().squeeze().numpy()
    high_mask_np = high_mask.detach().cpu().squeeze().numpy()

    boxes = pares_annotation(xml_path)
    ann_mask = create_annotation_mask(difference_ll.shape[0], difference_ll.shape[1], boxes)
    high_ann_mask = ann_mask.unsqueeze(0).expand_as(difference_high)

    total_ll_mask = torch.clamp(ll_mask + ann_mask, max=1.0)
    total_high_mask = torch.clamp(high_mask + high_ann_mask, max=1.0)

    # combined_high_mask = (high_mask.sum(dim=0) > 0).float()
    # plt.imshow(difference_Yl.cpu(), cmap='gray')
    # plt.imshow(combined_high_mask.cpu(), cmap='jet', alpha=0.4)
    # plt.title('Bottom 30% High-frequency (Any Channel)')
    # plt.axis('off')
    # plt.show()
    # freq_names = ['LH', 'HL', 'HH']
    # for c in range(high_img.shape[0]):
    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(high_img[c], cmap='gray')
    #     plt.imshow(high_mask_np[c], cmap='jet', alpha=0.4)
    #     plt.title(f'Top 30% High-frequency ({freq_names[c]})')
    #     plt.axis('off')
    #     plt.show()












    # difference_Yh_mask = []
    # for i in range(len(Yh_visible)):
    #     diff = Yh_visible[i] - Yh_infrared[i]
    #     mask = (diff >= threshold)
    #     diff = diff * mask
    #     difference_Yh.append(diff)

    # inverse = ifm((mask_yl, difference_Yh_mask))
    #
    # inverse_img = inverse.detach().cpu().numpy()
    # plt.imshow(inverse_img.squeeze(), cmap='gray')
    # plt.axis('off')
    # plt.show()
    ####################################################################
    # Yl_img = Yl_infrared.squeeze().detach().cpu().numpy()  # (H/2, W/2)
    # Yh_img = Yh_infrared[0].squeeze().detach().cpu().numpy()  # (3, H/2, W/2)

    # print(Yh_img.shape)
    # titles = ['Yl (Low freq)', 'Yh - LH', 'Yh - HL', 'Yh - HH']

    # # --- 그리기 ---
    # plt.figure(figsize=(14, 4))

    # # 1) Yl (저주파)
    # plt.subplot(1, 4, 1)
    # plt.imshow(Yl_img, cmap='gray')
    # plt.title(titles[0])
    # plt.axis('off')

    # # 2) Yh의 3 성분 (고주파)
    # for i in range(3):
    #     plt.subplot(1, 4, i + 2)
    #     plt.imshow(Yh_img[i], cmap='gray')
    #     plt.title(titles[i + 1])
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    ####################################################################
    # print(np.testing.assert_array_almost_equal(
    #     inverse.cpu().numpy(), X.cpu().numpy()
    # ))

    # print(inverse == X)
    # wt = get_wavelet_transform(backend="pytorch", wave_type="haar", device="cuda")
    # img = torch.randn(1, 3, 256, 256).cuda()



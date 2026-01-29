import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.dataset import ex_data

def patch_devide(image1, image2, patch_num):
    assert image1.shape[-1] % patch_num == 0
    assert image2.shape[-1] % patch_num == 0

    image1 = image1.unsqueeze(1)
    image1 = torch.chunk(image1, int(patch_num), dim=-1)
    image1 = torch.cat(image1, dim=1)
    image1 = image1.unsqueeze(1)
    image1_2 = torch.chunk(image1, int(patch_num), dim=-2)
    image1_2 = torch.cat(image1_2, dim=1)

    image2 = image2.unsqueeze(1)
    image2 = torch.chunk(image2, int(patch_num), dim=-1)
    image2 = torch.cat(image2, dim=1)
    image2 = image2.unsqueeze(1)
    image2_2 = torch.chunk(image2, int(patch_num), dim=-2)
    image2_2 = torch.cat(image2_2, dim=1)

    return image1_2, image2_2

def focus_map(image1, image2, ratio=0.25):
    diff = torch.abs(image1 - image2)
    B, P, _, _, _, _ = diff.shape
    score = diff.mean(dim=(3,4,5)).view(B,-1)

    k = int(score.shape[1] * ratio)
    _, idx = torch.topk(score, k, dim=1)

    mask = torch.zeros_like(score)
    mask.scatter_(1, idx, 1.0)
    mask = mask.view(B, P, P)

    return mask


if __name__ == "__main__":
    x_visible = ex_data(root_dir=r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg")
    x_infra = ex_data(root_dir=r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg")

    # image1 = torch.randn(3, 3, 224, 224)
    # image2 = torch.randn(3, 1, 224, 224)

    patch_num = 14
    image1_2, image2_2 = patch_devide(x_visible, x_infra, patch_num)
    mask = focus_map(image1_2, image2_2)
    print(mask)

    b = 0
    mask_img = mask[b].unsqueeze(0).unsqueeze(0)
    mask_up = F.interpolate(mask_img,size=(224, 224),mode='nearest').squeeze().cpu().numpy()
    img = x_visible[b, 0].cpu().numpy()
    visible_img = x_visible[0, 0].detach().cpu().numpy()
    infra_img = x_infra[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(visible_img, cmap='gray')
    plt.title('Visible')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(infra_img, cmap='gray')
    plt.title('Infrared')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask_up, cmap='jet', alpha=0.4)
    plt.title("Focus Patch Map")
    plt.axis('off')
    plt.show()
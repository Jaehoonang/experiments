from UNet_train import train_MMunet
from models.UNet import UNet
from data.unet_data import UNet_Dataset, DWT_UNet_Dataset, MMDWT_UNet_Dataset, ex1_MMDWT_UNet_Dataset
from PIL import Image
import cv2
import torch
import torch.nn as nn
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from data.dataset import ex_data
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

low_mode = 'low'
high_mode = 'high'

data_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val"

low_in = 1
low_out = 1

high_in = 3
high_out = 3

low_model = UNet(in_channels=low_in, out_channels=low_out)
low_model.to(device)

high_model = UNet(in_channels=high_in, out_channels=high_out)
high_model.to(device)

low_model_path = r"C:\Users\12wkd\Desktop\exp_result\1217\exp1_bootom30\low\best_model.pth"
high_model_path = r"C:\Users\12wkd\Desktop\exp_result\1217\exp1_bootom30\high\best_model.pth"

low_model.eval()
checkpoint1 = torch.load(low_model_path, map_location=device)
low_model.load_state_dict(checkpoint1)
low_model.eval()
print('low pretrained loaded')

high_model.eval()
checkpoint2 = torch.load(high_model_path, map_location=device)
high_model.load_state_dict(checkpoint2)
high_model.eval()
print('high pretrained loaded')


img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared\00025N.png"
x_visible = ex_data(root_dir=img_path)

print(x_visible.shape)
xfm = DWTForward(J=1, mode="periodization", wave='haar').to(device)  # Accepts all wave types available to PyWavelets
ifm = DWTInverse(mode="periodization", wave='haar').to(device)

low_visible, high_visible = xfm(x_visible)

low_visible = low_visible.to(device)
low_visible_img = low_visible

high_visible = high_visible[0].to(device)
high_visible_img = high_visible.squeeze(0)

print("Low input:", low_visible_img.shape)
print("High input:", high_visible_img.shape)

with torch.no_grad():
    low_output = low_model(low_visible_img)
    high_output = high_model(high_visible_img)

print(high_output.shape)
LH = high_output[:, 0:1, :, :]
HL = high_output[:, 1:2, :, :]
HH = high_output[:, 2:3, :, :]
ll = low_output.to(device)
highs = [torch.stack([LH, HL, HH], dim=2).to(device)]  # => (B, 1, 3, H, W) 형태로 전달됨

inverse = ifm((ll, highs))

inv_img = inverse.squeeze().cpu().numpy()
inv_img = (inv_img * 255.0).clip(0,255).astype(np.uint8)


# 후처리
low_output_img = low_output.squeeze().cpu().numpy()
low_output_img = (low_output_img * 255).astype(np.uint8)

high_img = highs[0].squeeze().detach().cpu().numpy()
high_output_img = high_output[0].squeeze().cpu().numpy()
high_output_img = (high_output_img * 255).astype(np.uint8)


# 결과 시각화
cv2.imshow("Reconstructed", inv_img)
cv2.imshow("Low Reconstruction", low_output_img)
cv2.imshow("LH Reconstruction", high_output_img[0,:,:])
cv2.imshow("HL Reconstruction", high_output_img[1,:,:])
cv2.imshow("HH Reconstruction", high_output_img[2,:,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

save_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\results"
cv2.imwrite(f"{save_path}/total_output.png", inv_img)
cv2.imwrite(f"{save_path}/low_output.png", low_output_img)
cv2.imwrite(f"{save_path}/high_output_LH.png", high_output_img[0, :, :])
cv2.imwrite(f"{save_path}/high_output_HL.png", high_output_img[1, :, :])
cv2.imwrite(f"{save_path}/high_output_HH.png", high_output_img[2, :, :])


# titles = ['Yl (Low freq)', 'Yh - LH', 'Yh - HL', 'Yh - HH']
# # 1) Yl (저주파)
# plt.subplot(1, 4, 1)
# plt.imshow(low_visible_img, cmap='gray')
# plt.title(titles[0])
# plt.axis('off')
#
# # 2) Yh의 3 성분 (고주파)
# for i in range(3):
#     plt.subplot(1, 4, i + 2)
#     plt.imshow(high_visible_img[i], cmap='gray')
#     plt.title(titles[i + 1])
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()


# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# img = img.astype(np.float32) / 255.0
# img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)


# with torch.no_grad():
#     low_output = low_model(low_visible_img)
#     high_output = high_model(high_visible_img)
#
# # 결과 후처리
# low_output_img = low_output.squeeze().cpu().numpy()
# low_output_img = (low_output_img * 255).astype(np.uint8)
#
# high_output_img = high_output.squeeze().cpu().numpy()
# high_output_img = (high_output_img * 255).astype(np.uint8)
#
# cv2.imshow("Inference low Result", low_output_img)
# cv2.waitKey(0)
#
# cv2.imshow("Inference high Result", high_output_img)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
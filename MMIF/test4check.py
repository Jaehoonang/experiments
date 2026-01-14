from data.dataset import ex_data
import matplotlib.pyplot as plt
import numpy as np
import cv2

# x_visible = ex_data(root_dir = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\visible\00025N.png")
# x_infra = ex_data(root_dir= r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared\00025N.png")
result1 = r"C:\Users\12wkd\Desktop\exp_result\0115\200_min\modal1_81_out_img.png"
result2 = r"C:\Users\12wkd\Desktop\exp_result\0115\200_min\modal2_81_out_img.png"

x_visible = ex_data(root_dir = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg")
x_infra = ex_data(root_dir= r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg")

res_visible = ex_data(root_dir=result1)
res_infra = ex_data(root_dir=result2)
###########################################################
visible_img = x_visible[0, 0].detach().cpu().numpy()
infra_img = x_infra[0, 0].detach().cpu().numpy()

infra_norm = cv2.normalize(infra_img, None, 0, 255, cv2.NORM_MINMAX)
visible_norm = cv2.normalize(visible_img, None, 0, 255, cv2.NORM_MINMAX)

infra_uint8 = infra_norm.astype(np.uint8)
visible_uint8 = visible_norm.astype(np.uint8)

# CLAHE
clahe1 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
infra_img2 = clahe1.apply(infra_uint8)

clahe2 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
visible_img2 = clahe2.apply(visible_uint8)

modal_sum = visible_img + infra_img
clahe_sum = visible_img2 + infra_img

sum_norm = cv2.normalize(modal_sum, None, 0, 255, cv2.NORM_MINMAX)
sum_uint8 = sum_norm.astype(np.uint8)
clahe3= cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
sum_img2 = clahe3.apply(sum_uint8)

modal_minus = visible_img - infra_img
###########################################################
res_visible_img = res_visible[0, 0].detach().cpu().numpy()
res_infra_img = res_infra[0, 0].detach().cpu().numpy()

res_infra_norm = cv2.normalize(res_infra_img, None, 0, 255, cv2.NORM_MINMAX)
res_visible_norm = cv2.normalize(res_visible_img, None, 0, 255, cv2.NORM_MINMAX)

res_infra_uint8 = res_infra_norm.astype(np.uint8)
res_visible_uint8 = res_visible_norm.astype(np.uint8)

# CLAHE
clahe1_1 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
res_infra_img2 = clahe1_1.apply(res_infra_uint8)

clahe2_1 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
res_visible_img2 = clahe2_1.apply(res_visible_uint8)

res_modal_sum = res_visible_img + res_infra_img
res_clahe_sum = res_visible_img2 + res_infra_img

res_sum_norm = cv2.normalize(res_modal_sum, None, 0, 255, cv2.NORM_MINMAX)
res_sum_uint8 = res_sum_norm.astype(np.uint8)
clahe3_1= cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
res_sum_img2 = clahe3_1.apply(res_sum_uint8)

res_modal_minus = res_visible_img - res_infra_img

# 모달 합친거
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(visible_img, cmap='gray')
plt.title('Visible')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(infra_img, cmap='gray')
plt.title('Infra')
plt.axis('off')

# plt.subplot(1, 5, 3)
# plt.imshow(visible_img2, cmap='gray')
# plt.title('Clahe Visible')
# plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(modal_sum, cmap='gray')
plt.title('Modal Sum')
plt.axis('off')
#
plt.subplot(1, 4, 4)
plt.imshow(sum_img2, cmap='gray')
plt.title('Clahe Sum')
plt.axis('off')
########################################################################
plt.figure(figsize=(15, 5))
plt.subplot(2, 4, 1)
plt.imshow(res_visible_img, cmap='gray')
plt.title('Visible')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(res_infra_img, cmap='gray')
plt.title('Infra')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(res_modal_sum, cmap='gray')
plt.title('Modal Sum')
plt.axis('off')
#
plt.subplot(2, 4, 4)
plt.imshow(res_sum_img2, cmap='gray')
plt.title('Clahe Sum')
plt.axis('off')

plt.show()

# # 모달 뺀거
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(visible_img, cmap='gray')
# plt.title('Visible')
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(infra_img, cmap='gray')
# plt.title('Infra')
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.imshow(modal_minus, cmap='gray')
# plt.title('Modal Minus')
# plt.axis('off')
# plt.show()

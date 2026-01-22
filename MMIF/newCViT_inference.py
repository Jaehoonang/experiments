import torch
from models.newCViT import newCViTFlow
from data.dataset import ex_data
import numpy as np
import cv2
from PIL import Image
import torchvision

pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\newCVT_checkpoints\best_representation_model.pth"

# 81, 97
vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010097.jpg"
inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010097.jpg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = newCViTFlow(in_channels=1, embed_dim=64, patch_size=8, num_heads=8, dim=64).to(device)

model.eval()
checkpoint1 = torch.load(pt_path, map_location=device)
model.load_state_dict(checkpoint1['model_state'])
model.eval()

vis_image = ex_data(root_dir=vis_img_path)
inf_image = ex_data(root_dir=inf_img_path)

with torch.no_grad():
    out = model(vis_image, inf_image)
print(out.min(), out.max())
print('modal1 representation', out.shape)

out_img = out.squeeze().cpu().numpy()
out_img = (out_img * 255).astype(np.uint8)

save_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\results"
cv2.imwrite(f"{save_path}/modal1_out_img.png", out_img)
cv2.imshow("modal1_out_img", out_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# fused_img = fused[0, 0].detach().cpu().numpy()
# fused_img = fused_img - fused_img.min()
# fused_img = fused_img / (fused_img.max() + 1e-8)
# fused_img = (fused_img * 255).astype(np.uint8)

# fused_img = fused[0,0].cpu().numpy()
#
# # contrast stretch
# p2, p98 = np.percentile(fused_img, (2, 98))
# fused_img = np.clip((fused_img - p2) / (p98 - p2 + 1e-8), 0, 1)
#
# fused_img = (fused_img * 255).astype(np.uint8)
#
# cv2.imshow("fused image", fused_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# fused_img = fused.squeeze().cpu().numpy()
# fused_img = (fused_img * 255).astype(np.uint8)
# cv2.imshow("fused image", fused_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# fused_img = fused.squeeze().cpu().numpy()
# fused_img = (fused_img * 255).astype(np.uint8)
# cv2.imshow("fused image", fused_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
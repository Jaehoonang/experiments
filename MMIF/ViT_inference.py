import torch
from models.fusion_transformer import ViTFlow
from data.dataset import ex_data
import numpy as np
import cv2
import torchvision

pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\ViT_checkpoints\best_representation_model.pth"

vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\visible\00025N.png"
inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\val\infrared\00025N.png"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTFlow(in_channels=1, img_size=224, emb_size=768, patch_size=8, num_heads=12, out_channels=1)
model.to(device)

model.eval()
checkpoint1 = torch.load(pt_path, map_location=device)
model.load_state_dict(checkpoint1['model_state'])
model.eval()

vis_image = ex_data(root_dir=vis_img_path)
inf_image = ex_data(root_dir=inf_img_path)

with torch.no_grad():
    modal1_out, modal2_out = model(vis_image, inf_image)
print(modal1_out.min(), modal1_out.max())
print('modal1 representation', modal1_out.shape)
print('modal2 representation', modal2_out.shape)

modal1_out_img = modal1_out.squeeze().cpu().numpy()
modal2_out_img = modal2_out.squeeze().cpu().numpy()
modal1_out_img = (modal1_out_img * 255).astype(np.uint8)
modal2_out_img = (modal2_out_img * 255).astype(np.uint8)

save_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\results"
cv2.imwrite(f"{save_path}/modal1_out_img.png", modal1_out_img)
cv2.imwrite(f"{save_path}/modal2_out_img.png", modal2_out_img)

cv2.imshow("modal1_out_img", modal1_out_img)
cv2.imshow("modal2_out_img", modal2_out_img)
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
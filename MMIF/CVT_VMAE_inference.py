import torch
from models.CVT_VMAE import CVT_VMAE
from data.dataset import ex_data
import numpy as np

import matplotlib.pyplot as plt


pt_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\CVT_VMAE_checkpoints\best_representation_model.pth"

vis_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\visible\010081.jpg"
inf_img_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\test\infrared\010081.jpg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CVT_VMAE().to(device)

model.eval()
checkpoint1 = torch.load(pt_path, map_location=device)
model.load_state_dict(checkpoint1['model_state'])
samples = []

vis_image = ex_data(root_dir=vis_img_path)
inf_image = ex_data(root_dir=inf_img_path)

with torch.no_grad():
    x, posterior = model(vis_image, inf_image, sample_latent=False)
    for _ in range(20):
        x, _ = model(vis_image, inf_image, sample_latent=True, latent_scale=30)
        samples.append(x.cpu())

vis = vis_image.squeeze().cpu().numpy()
inf = inf_image.squeeze().cpu().numpy()
x_img = x.squeeze().cpu().numpy()
x_img = np.clip(x_img, 0, 1)

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Visible Input")
plt.imshow(vis, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Infrared Input")
plt.imshow(inf, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Summation feature")
plt.imshow(inf+vis, cmap='gray')
plt.axis('off')
#
# plt.subplot(2,3,4)
# plt.title("Model Output - infrared = visible")
# plt.imshow(np.abs(x_img - inf), cmap='gray')
# plt.axis('off')
#
# plt.subplot(2,3,5)
# plt.title("Model Output - visible = infrared")
# plt.imshow(np.abs(x_img - vis), cmap='gray')
# plt.axis('off')
#
plt.subplot(1,4,4)
plt.title("Model Output (Reconstruction)")
plt.imshow(x_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))
for i, img in enumerate(samples):
    plt.subplot(4, 5, i + 1)
    plt.imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title(f"sample {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# save_path = r"C:\Users\12wkd\Desktop\experiments\MMIF\onlytest\results"
# cv2.imwrite(f"{save_path}/x_img.png", x_img)
#
# cv2.imshow("x_img", x_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
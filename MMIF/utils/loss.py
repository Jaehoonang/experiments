import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self,x):
        B, C, H, W = x.shape

        weightx = self.weightx.repeat(C, 1, 1, 1)
        weighty = self.weighty.repeat(C, 1, 1, 1)

        sobelx=F.conv2d(x, weightx, padding=1, groups=C)
        sobely=F.conv2d(x, weighty, padding=1, groups=C)

        return torch.abs(sobelx)+torch.abs(sobely)

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = Sobelxy()

    def forward(self, fused, img1, img2):
        grad_fused = self.grad(fused)
        grad_1 = self.grad(img1)
        grad_2 = self.grad(img2)

        grad_max = torch.max(grad_1, grad_2)

        loss = torch.mean(torch.abs(grad_fused - grad_max))
        return loss

class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, image_fused, img1, img2):
        intensity_joint = torch.max(img1, img2)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

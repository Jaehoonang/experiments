from pathlib import Path
import numpy as np
import torch
from numpy.ma.core import masked_greater_equal
from torch.utils.data import Dataset
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
import xml.etree.ElementTree as ET

#### test dataset ####
class UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mask_ratio=0.3):
        self.data_path = Path(data_path)
        self.images = sorted(self.data_path.glob("*"))
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        gt = img.clone()
        mask = torch.rand_like(img)
        masked_img = img * mask

        return masked_img, gt

class DWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3):
        self.data_path = Path(data_path)
        self.images = sorted(self.data_path.glob("*"))
        self.transform = transform
        self.mode = mode
        self.device = device
        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        img = img.to(self.device)
        LL, High_list = self.dwt(img.unsqueeze(0))

        High = torch.cat(High_list, dim=0)

        LL = LL.squeeze(0)
        High = High.squeeze()

        mask_ll = (torch.rand_like(LL) > self.mask_ratio).float()
        mask_high = (torch.rand_like(High) > self.mask_ratio).float()

        masked_LL = LL * mask_ll
        masked_High = High * mask_high


        if self.mode == 'low':
            return masked_LL, LL
        else:
            return masked_High, High

# class MMDWT_UNet_Dataset(Dataset):
#     def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
#         self.data_path = Path(data_path)
#         self.modal1 = modal1
#         self.modal2 = modal2
#
#         self.modal1_dir = self.data_path / self.modal1
#         self.modal2_dir = self.data_path / self.modal2
#
#         self.images = sorted(self.modal1_dir.glob("*"))
#
#         self.transform = transform
#         self.mode = mode
#         self.device = device
#
#         self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
#         self.mask_ratio = mask_ratio
#
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         modal1_path = self.images[idx]
#         modal2_path = self.modal2_dir / modal1_path.name
#
#         modal1_img = Image.open(modal1_path).convert('L')
#         modal2_img = Image.open(modal2_path).convert('L')
#
#         if self.transform:
#             modal1_img = self.transform(modal1_img)
#             modal2_img = self.transform(modal2_img)
#
#         modal1_img = modal1_img.to(self.device)
#         modal2_img = modal2_img.to(self.device)
#
#         # wavelet decomposition for each modal
#         modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
#         modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))
#
#         modal1_High = torch.cat(modal1_High_list, dim=0)
#         modal2_High = torch.cat(modal2_High_list, dim=0)
#
#         modal1_LL = modal1_LL.squeeze(0)
#         modal1_High = modal1_High.squeeze()
#
#         modal2_LL = modal2_LL.squeeze(0)
#         modal2_High = modal2_High.squeeze()
#
#         # subtract the modal frequencies for find big difference gap area
#         # and also masking the top k-th big gap area
#
#         difference_ll = torch.abs(modal1_LL - modal2_LL)
#         difference_high = torch.abs(modal1_High - modal2_High)
#
#         ll_values, ll_indices = torch.sort(difference_ll.view(-1))
#         ll_k = int((1 - self.mask_ratio) * len(ll_values))
#         threshold = ll_values[ll_k]
#         ll_mask = (difference_ll >= threshold).float()
#
#         high_values, high_indices = torch.sort(difference_high.view(-1))
#         high_k = int((1 - self.mask_ratio) * len(high_values))
#         threshold = high_values[high_k]
#         high_mask = (difference_high >= threshold).float()
#         masekd_ll_dal1 = modal1_LL * ll_mask
#         masekd_ll_dal2 = modal2_LL * ll_mask
#
#         masekd_high_dal1 = modal1_High * high_mask
#         masekd_high_dal2 = modal2_High * high_mask
#
#         masked_LL = torch.stack([masekd_ll_dal1, masekd_ll_dal2], dim=0)
#         masked_High = torch.stack([masekd_high_dal1, masekd_high_dal2], dim=0)
#
#         # ground truth
#         LL = torch.stack([modal1_LL, modal2_LL], dim=0)
#         High = torch.stack([modal1_High, modal2_High], dim=0)
#
#         if self.mode == 'low':
#             return masked_LL, LL
#         else:
#             return masked_High, High

##### using dataset #####
# bottom30
class MMDWT_UNet_Bottom_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
        self.data_path = Path(data_path)
        self.modal1 = modal1
        self.modal2 = modal2

        self.modal1_dir = self.data_path / self.modal1
        self.modal2_dir = self.data_path / self.modal2

        self.images = sorted(self.modal1_dir.glob("*"))

        self.transform = transform
        self.mode = mode
        self.device = device

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        modal1_path = self.images[idx]
        modal2_path = self.modal2_dir / modal1_path.name

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        # wavelet decomposition for each modal
        modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
        modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))

        modal1_High = torch.cat(modal1_High_list, dim=0)
        modal2_High = torch.cat(modal2_High_list, dim=0)

        modal1_LL = modal1_LL.squeeze(0)
        modal1_High = modal1_High.squeeze()

        modal2_LL = modal2_LL.squeeze(0)
        modal2_High = modal2_High.squeeze()

        # subtract the modal frequencies for find big difference gap area
        # and also masking the top k-th big gap area

        difference_ll = torch.abs(modal1_LL - modal2_LL)
        difference_high = torch.abs(modal1_High - modal2_High)

        ll_values, ll_indices = torch.sort(difference_ll.view(-1))
        ll_k = int((self.mask_ratio) * len(ll_values))
        threshold = ll_values[ll_k]
        ll_mask = (difference_ll > threshold).float()

        high_mask = torch.zeros_like(difference_high)
        for c in range(difference_high.size(0)):
            vals = torch.sort(difference_high[c].view(-1))[0]
            k = int(self.mask_ratio * len(vals))
            threshold = vals[k]
            high_mask[c] = (difference_high[c] > threshold).float()

        # high_values, high_indices = torch.sort(difference_high.view(-1))
        # high_k = int((self.mask_ratio) * len(high_values))
        # threshold = high_values[high_k]
        # high_mask = (difference_high <= threshold).float()
        # print(high_mask.shape)

        # ground truth
        LL = torch.stack([modal1_LL, modal2_LL], dim=0)
        High = torch.stack([modal1_High, modal2_High], dim=0)

        if self.mode == 'low':
            return LL, ll_mask, LL #(input, masking map, GT)
        else:
            return High, high_mask, High #(input, masking map, GT) #

# top30
class MMDWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
        self.data_path = Path(data_path)
        self.modal1 = modal1
        self.modal2 = modal2

        self.modal1_dir = self.data_path / self.modal1
        self.modal2_dir = self.data_path / self.modal2

        self.images = sorted(self.modal1_dir.glob("*"))

        self.transform = transform
        self.mode = mode
        self.device = device

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        modal1_path = self.images[idx]
        modal2_path = self.modal2_dir / modal1_path.name

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        # wavelet decomposition for each modal
        modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
        modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))

        modal1_High = torch.cat(modal1_High_list, dim=0)
        modal2_High = torch.cat(modal2_High_list, dim=0)

        modal1_LL = modal1_LL.squeeze(0)
        modal1_High = modal1_High.squeeze()

        modal2_LL = modal2_LL.squeeze(0)
        modal2_High = modal2_High.squeeze()

        # subtract the modal frequencies for find big difference gap area
        # and also masking the top k-th big gap area

        difference_ll = torch.abs(modal1_LL - modal2_LL)
        difference_high = torch.abs(modal1_High - modal2_High)

        ll_values, ll_indices = torch.sort(difference_ll.view(-1))
        ll_k = int((1 - self.mask_ratio) * len(ll_values))
        threshold = ll_values[ll_k]
        ll_mask = (difference_ll >= threshold).float()

        high_mask = torch.zeros_like(difference_high)
        for c in range(difference_high.size(0)):
            vals = torch.sort(difference_high[c].view(-1))[0]
            k = int(self.mask_ratio * len(vals))
            threshold = vals[k]
            high_mask[c] = (difference_high[c] >= threshold).float()

        # ground truth
        LL = torch.stack([modal1_LL, modal2_LL], dim=0)
        High = torch.stack([modal1_High, modal2_High], dim=0)

        if self.mode == 'low':
            return LL, ll_mask, LL  # (input, masking map, GT)
        else:
            return High, high_mask, High  # (input, masking map, GT)

# annotation all masking
class ex1_MMDWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, annotation_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
        self.data_path = Path(data_path)
        self.annotation_path = Path(annotation_path)

        self.modal1 = modal1
        self.modal2 = modal2

        self.modal1_dir = self.data_path / self.modal1
        self.modal2_dir = self.data_path / self.modal2

        self.images = sorted(self.modal1_dir.glob("*"))

        self.transform = transform
        self.mode = mode
        self.device = device

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio


    def __len__(self):
        return len(self.images)

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

    def __getitem__(self, idx):
        modal1_path = self.images[idx]
        modal2_path = self.modal2_dir / modal1_path.name
        xml_path = self.annotation_path / (modal1_path.stem + ".xml")

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        # wavelet decomposition for each modal
        modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
        modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))

        modal1_High = torch.cat(modal1_High_list, dim=0)
        modal2_High = torch.cat(modal2_High_list, dim=0)

        modal1_LL = modal1_LL.squeeze(0)
        modal1_High = modal1_High.squeeze()

        modal2_LL = modal2_LL.squeeze(0)
        modal2_High = modal2_High.squeeze()

        # subtract the modal frequencies for find big difference gap area
        # and also masking the top k-th big gap area

        difference_ll = torch.abs(modal1_LL - modal2_LL)
        difference_high = torch.abs(modal1_High - modal2_High)

        ll_values, ll_indices = torch.sort(difference_ll.view(-1))
        ll_k = int((1 - self.mask_ratio) * len(ll_values))
        threshold = ll_values[ll_k]
        ll_mask = (difference_ll >= threshold).float()

        high_mask = torch.zeros_like(difference_high)
        for c in range(difference_high.size(0)):
            vals = torch.sort(difference_high[c].view(-1))[0]
            k = int(self.mask_ratio * len(vals))
            threshold = vals[k]
            high_mask[c] = (difference_high[c] >= threshold).float()

        # annotation masking
        boxes = self.pares_annotation(xml_path)
        ann_mask = self.create_annotation_mask(difference_ll.shape[0], difference_ll.shape[1], boxes)
        high_ann_mask = ann_mask.unsqueeze(0).expand_as(difference_high)

        total_ll_mask = torch.clamp(ll_mask + ann_mask, max=1.0)
        total_high_mask = torch.clamp(high_mask + high_ann_mask, max=1.0)

        # ground truth
        LL = torch.stack([modal1_LL, modal2_LL], dim=0)
        High = torch.stack([modal1_High, modal2_High], dim=0)

        if self.mode == 'low':
            return LL, total_ll_mask, LL  # (input, masking map, GT)
        else:
            return High, total_high_mask, High  # (input, masking map, GT)

# annotation weight masking
class weight_MMDWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, annotation_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
        self.data_path = Path(data_path)
        self.annotation_path = Path(annotation_path)

        self.modal1 = modal1
        self.modal2 = modal2

        self.modal1_dir = self.data_path / self.modal1
        self.modal2_dir = self.data_path / self.modal2

        self.images = sorted(self.modal1_dir.glob("*"))

        self.transform = transform
        self.mode = mode
        self.device = device

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio


    def __len__(self):
        return len(self.images)

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
            mask[ymin:ymax, xmin:xmax] = 0.5
        return mask

    def __getitem__(self, idx):
        modal1_path = self.images[idx]
        modal2_path = self.modal2_dir / modal1_path.name
        xml_path = self.annotation_path / (modal1_path.stem + ".xml")

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        # wavelet decomposition for each modal
        modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
        modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))

        modal1_High = torch.cat(modal1_High_list, dim=0)
        modal2_High = torch.cat(modal2_High_list, dim=0)

        modal1_LL = modal1_LL.squeeze(0)
        modal1_High = modal1_High.squeeze()

        modal2_LL = modal2_LL.squeeze(0)
        modal2_High = modal2_High.squeeze()

        # subtract the modal frequencies for find big difference gap area
        # and also masking the top k-th big gap area

        difference_ll = torch.abs(modal1_LL - modal2_LL)
        difference_high = torch.abs(modal1_High - modal2_High)

        boxes = self.pares_annotation(xml_path)
        ann_mask = self.create_annotation_mask(difference_ll.shape[0], difference_ll.shape[1], boxes)

        # add ann_mask info as a weight for make focusing map
        difference_annot_ll = difference_ll + ann_mask
        difference_annot_high = difference_high + ann_mask

        ll_values, ll_indices = torch.sort(difference_annot_ll.view(-1))
        ll_k = int((1 - self.mask_ratio) * len(ll_values))
        threshold = ll_values[ll_k]
        ll_mask = (difference_annot_ll >= threshold).float()

        high_mask = torch.zeros_like(difference_annot_high)
        for c in range(difference_annot_high.size(0)):
            vals = torch.sort(difference_annot_high[c].view(-1))[0]
            k = int(self.mask_ratio * len(vals))
            threshold = vals[k]
            high_mask[c] = (difference_annot_high[c] >= threshold).float()

        # ground truth
        LL = torch.stack([modal1_LL, modal2_LL], dim=0)
        High = torch.stack([modal1_High, modal2_High], dim=0)

        if self.mode == 'low':
            return LL, ll_mask, LL
        else:
            return High, high_mask, High


class Patch_MMDWT_UNet_Dataset(Dataset):
    def __init__(self, data_path, transform=None, mode='low', device='gpu', mask_ratio=0.3, modal1='visible', modal2='infrared'):
        self.data_path = Path(data_path)
        self.modal1 = modal1
        self.modal2 = modal2

        self.modal1_dir = self.data_path / self.modal1
        self.modal2_dir = self.data_path / self.modal2

        self.images = sorted(self.modal1_dir.glob("*"))

        self.transform = transform
        self.mode = mode
        self.device = device

        self.dwt = DWTForward(J=1, mode="periodization", wave='haar').to(self.device)
        self.mask_ratio = mask_ratio


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        modal1_path = self.images[idx]
        modal2_path = self.modal2_dir / modal1_path.name

        modal1_img = Image.open(modal1_path).convert('L')
        modal2_img = Image.open(modal2_path).convert('L')

        if self.transform:
            modal1_img = self.transform(modal1_img)
            modal2_img = self.transform(modal2_img)

        modal1_img = modal1_img.to(self.device)
        modal2_img = modal2_img.to(self.device)

        # wavelet decomposition for each modal
        modal1_LL, modal1_High_list = self.dwt(modal1_img.unsqueeze(0))
        modal2_LL, modal2_High_list = self.dwt(modal2_img.unsqueeze(0))

        modal1_High = torch.cat(modal1_High_list, dim=0)
        modal2_High = torch.cat(modal2_High_list, dim=0)

        modal1_LL = modal1_LL.squeeze(0)
        modal1_High = modal1_High.squeeze()

        modal2_LL = modal2_LL.squeeze(0)
        modal2_High = modal2_High.squeeze()

        # subtract the modal frequencies for find big difference gap area
        # and also masking the top k-th big gap area

        difference_ll = torch.abs(modal1_LL - modal2_LL)
        difference_high = torch.abs(modal1_High - modal2_High)

        ll_values, ll_indices = torch.sort(difference_ll.view(-1))
        ll_k = int((1 - self.mask_ratio) * len(ll_values))
        threshold = ll_values[ll_k]
        ll_mask = (difference_ll >= threshold).float()

        high_mask = torch.zeros_like(difference_high)
        for c in range(difference_high.size(0)):
            vals = torch.sort(difference_high[c].view(-1))[0]
            k = int(self.mask_ratio * len(vals))
            threshold = vals[k]
            high_mask[c] = (difference_high[c] >= threshold).float()

        # ground truth
        LL = torch.stack([modal1_LL, modal2_LL], dim=0)
        High = torch.stack([modal1_High, modal2_High], dim=0)

        if self.mode == 'low':
            return LL, ll_mask, LL  # (input, masking map, GT)
        else:
            return High, high_mask, High  # (input, masking map, GT)

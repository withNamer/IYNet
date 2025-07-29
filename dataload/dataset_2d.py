import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pywt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from scipy.ndimage.interpolation import zoom
import time

class dataset_XNetv2(Dataset):
    def __init__(self, data_dir, augmentation_1, augmentation_strong_1, 
                #  augmentation_strong_resize_1, 
                 normalize_1, wavelet_type, alpha, beta, gamma, scale, sup=True, num_images=None, **kwargs):
        super(dataset_XNetv2, self).__init__()

        img_paths_1 = []
        mask_paths = []

        image_dir_1 = data_dir + '/image'
        # image_dir_1 = data_dir + '/image_down'
        # image_dir_1 = data_dir + '/image_down_128'
        if sup:
            mask_dir = data_dir + '/mask'
            # mask_dir = data_dir + '/mask_down'
            # mask_dir = data_dir + '/mask_down_128'

        for image in os.listdir(image_dir_1):

            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.augmentation_strong_1 = augmentation_strong_1
        # self.augmentation_strong_resize_1 = augmentation_strong_resize_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.kwargs = kwargs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scale = scale
        self.wavelet_type = wavelet_type

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img = Image.open(img_path_1)
        img = np.array(img).astype(np.float32) / 255.0

        if self.sup:
            mask_path = self.mask_paths[index]
            # print(mask_path)
            # name = mask_path[:-4]
            # mask_path = name + '_segmentation' + '.png'
            mask = Image.open(mask_path).convert('L')         
            mask = np.array(mask)
            # print(mask.shape)
            mask = (mask > 128).astype(np.uint8) # 这也是防止255导致损失函数出错

        # transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #                        ], additional_targets={'image': 'LL',
        #                                               'image': 'LH', 
        #                                               'image': 'HL',
        #                                               'image': 'HH'})  # 声明额外目标为掩码
        # extra_trans = transform(image=img,)
        # img = extra_trans['image']

        # start_time = time.time()

        # w, h, _ = img.shape
        # max_wh = max(w, h)
        # max_len = 512
        # if max_wh > max_len:
        #     if h >= w:
        #         ratio = w / h
        #         img = zoom(img, (ratio * max_len / w, max_len / h, 1), order=0)
        #         mask = zoom(mask, (ratio * max_len / w, max_len / h), order=0)
        #     else:
        #         ratio = h / w
        #         img = zoom(img, (max_len / w, ratio * max_len / h, 1), order=0)
        #         mask = zoom(mask, (max_len / w, ratio * max_len / h), order=0)

        # h, w, _ = img.shape  # 获取高度和宽度（忽略通道数）
        # max_wh = max(w, h)
        # max_len = 640

        # if max_wh > max_len:
        #     if h >= w:
        #         ratio = w / h
        #         new_w = int(max_len * ratio)
        #         img = cv2.resize(img, (new_w, max_len), interpolation=cv2.INTER_NEAREST)
        #         mask = cv2.resize(mask, (new_w, max_len), interpolation=cv2.INTER_NEAREST)
        #     else:
        #         ratio = h / w
        #         new_h = int(max_len * ratio)
        #         img = cv2.resize(img, (max_len, new_h), interpolation=cv2.INTER_NEAREST)
        #         mask = cv2.resize(mask, (max_len, new_h), interpolation=cv2.INTER_NEAREST)

        # print(img.shape)

        LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet_type, axes=(0, 1))

        LL_2 = LL ** 2
        H_2 = LH ** 2 + HL ** 2 + HH ** 2  # 不要使用np.power，太慢了

        LL = (LL - np.amin(LL, (0, 1))) / (np.amax(LL, (0, 1)) - np.amin(LL, (0, 1))) * 1
        LH = (LH - np.amin(LH, (0, 1))) / (np.amax(LH, (0, 1)) - np.amin(LH, (0, 1))) * 1
        HL = (HL - np.amin(HL, (0, 1))) / (np.amax(HL, (0, 1)) - np.amin(HL, (0, 1))) * 1
        HH = (HH - np.amin(HH, (0, 1))) / (np.amax(HH, (0, 1)) - np.amin(HH, (0, 1))) * 1

        LL_2 = (LL_2 - np.amin(LL_2, (0, 1))) / (np.amax(LL_2, (0, 1)) - np.amin(LL_2, (0, 1))) * 0.5
        H_2 = (H_2 - np.amin(H_2, (0, 1))) / (np.amax(H_2, (0, 1)) - np.amin(H_2, (0, 1))) * 0.5 # 能量化

        # LL_2 = np.power(LL / 4.0, 4)  # 这儿需不需要归一化呢？？？？？

        H_ = HL + LH + HH
        H_ = (H_ - np.amin(H_, (0, 1))) / (np.amax(H_, (0, 1)) - np.amin(H_, (0, 1))) * 1

        # H_2 = np.power(H_ / 4.0, 4)  # 这样更有倒数的意思，应该来说高频会引进更多的噪声，所以这样也许更好；

        L_alpha = random.uniform(self.alpha[0], self.alpha[1])
        H_beta = random.uniform(self.beta[0], self.beta[1])

        L = LL + L_alpha * H_ #+ L_alpha * H_2  # 我就说，这个有点类似于指数的泰勒展开了
        L = (L - np.amin(L, (0, 1))) / (np.amax(L, (0, 1)) - np.amin(L, (0, 1))) * 1

        H = H_ + H_beta * LL #+ H_beta * LL_2
        H = (H - np.amin(H, (0, 1))) / (np.amax(H, (0, 1)) - np.amin(H, (0, 1))) * 1

        LH_gamma = random.uniform(self.gamma[0], self.gamma[1])
        LH_ = LH_gamma * LL + (1 - LH_gamma) * H_  # + LH_gamma * LL_2 + (1 - LH_gamma) * H_2  # 这儿我觉得应该不能改，得保持纯净
        # LH_ = (1-LH_gamma) * LL + LH_gamma * H
        LH_ = (LH_ - np.amin(LH_, (0, 1))) / (np.amax(LH_, (0, 1)) - np.amin(LH_, (0, 1))) * 1

        L = (L_alpha + 0.6) * L # + (L_alpha - self.alpha[0] - (self.alpha[1] - self.alpha[0]) / 2.0) / 10  # (0.2, 0.6)  0.5
        H = (H_beta + 0.6) * H # + (H_beta - self.beta[0] - (self.beta[1] - self.beta[0]) / 2.0) / 10   # (0.2, 0.6)  0.5
        LH_ = (LH_gamma + 0.0) * LH_  # (0.6,0.8) 0.0 这是最好的

        # L = 0.7 * L  # (0.2, 0.6)  0.5
        # H = 0.7 * H   # (0.2, 0.6)  0.5
        # LH_ = 0.7 * LH_  # (0.6,0.8) 0.0 这是最好的

        # end_time = time.time()

        # elapsed = end_time - start_time 
        # print(f"耗时: {elapsed:.4f} 秒") # 这个的速度实在是太慢了，很不舒适

        # scale_1 = random.uniform(self.scale[0], self.scale[1])
        # scale_2 = random.uniform(self.scale[0], self.scale[1])
        # scale_3 = random.uniform(self.scale[0], self.scale[1])
        # L = scale_1 * L
        # H = scale_2 * H
        # LH_ = scale_3 * LH_

        # 先做strong，然后weak增强即可
        # print(self.augmentation_strong_1 != None)
        # print(self.augmentation_strong_1)
        if self.augmentation_strong_1 is not None:
            augment_strong_1 = self.augmentation_strong_1(image=img, )           
            img_strong_1 = augment_strong_1['image']
            # cv2.imwrite("output_.jpg", cv2.cvtColor((img_strong_1* 1) .clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            # L_strong = augment_strong_1['L']
            # H_strong = augment_strong_1['H']
            # LL_strong = augment_strong_1['LL']
            # LH_strong = augment_strong_1['LH']
            # HL_strong = augment_strong_1['HL']
            # HH_strong = augment_strong_1['HH']

            LL_strong, (LH_strong, HL_strong, HH_strong) = pywt.dwt2(img_strong_1, self.wavelet_type, axes=(0, 1))
            LL_strong = (LL_strong - np.amin(LL_strong, (0, 1))) / (np.amax(LL_strong, (0, 1)) - np.amin(LL_strong, (0, 1))) * 1
            LH_strong = (LH_strong - np.amin(LH_strong, (0, 1))) / (np.amax(LH_strong, (0, 1)) - np.amin(LH_strong, (0, 1))) * 1
            HL_strong = (HL_strong - np.amin(HL_strong, (0, 1))) / (np.amax(HL_strong, (0, 1)) - np.amin(HL_strong, (0, 1))) * 1
            HH_strong = (HH_strong - np.amin(HH_strong, (0, 1))) / (np.amax(HH_strong, (0, 1)) - np.amin(HH_strong, (0, 1))) * 1

            H_strong = HL_strong + LH_strong + HH_strong
            H_strong = (H_strong - np.amin(H_strong, (0, 1))) / (np.amax(H_strong, (0, 1)) - np.amin(H_strong, (0, 1))) * 1

            # L_alpha = random.uniform(self.alpha[0], self.alpha[1])
            L_strong = LL_strong + L_alpha * H_strong
            L_strong = (L_strong - np.amin(L_strong, (0, 1))) / (np.amax(L_strong, (0, 1)) - np.amin(L_strong, (0, 1))) * 1

            # H_beta = random.uniform(self.beta[0], self.beta[1])
            H_strong = H_strong + H_beta * LL_strong
            H_strong = (H_strong - np.amin(H_strong, (0, 1))) / (np.amax(H_strong, (0, 1)) - np.amin(H_strong, (0, 1))) * 1

            # normalize_strong_1 = self.normalize_1(image=img_strong_1, L=L_strong, H=H_strong, LL=LL_strong, LH=LH_strong, HL=HL_strong, HH=HH_strong)
            # img_strong_1 = normalize_strong_1['image']
            # L_strong = normalize_strong_1['L']
            # H_strong = normalize_strong_1['H']
            # LL_strong = normalize_strong_1['LL']
            # LH_strong = normalize_strong_1['LH']
            # HL_strong = normalize_strong_1['HL']
            # HH_strong = normalize_strong_1['HH']

        if self.sup:
            # mask_path = self.mask_paths[index]
            # # print(mask_path)
            # name = mask_path[:-4]
            # mask_path = name + '_segmentation' + '.png'
            # mask = Image.open(mask_path)#.convert('L')         
            # mask = np.array(mask)
            # # print(mask.shape)
            # mask = (mask > 128).astype(np.uint8) # 这也是防止255导致损失函数出错
            # mask[mask == 255] = 1 # 这里居然没有改动，也许这不是最终版的代码


            if self.augmentation_strong_1 is not None:
                augment_1 = self.augmentation_1(image=img, mask=mask, L=L, H=H, 
                                                # LL=LL, LH=LH, HL=HL, HH=HH, 
                                        image_strong=img_strong_1, L_strong=L_strong, H_strong=H_strong, 
                                        # LL_strong=LL_strong, LH_strong=LH_strong, HL_strong=HL_strong, HH_strong=HH_strong
                                        )
                img_1 = augment_1['image']
                L = augment_1['L']
                H = augment_1['H']
                # LL = augment_1['LL']
                # LH = augment_1['LH']
                # HL = augment_1['HL']
                # HH = augment_1['HH']
                mask_1 = augment_1['mask']
                img_strong_1 = augment_1['image_strong']
                L_strong = augment_1['L_strong']
                H_strong = augment_1['H_strong']
                # LL_strong = augment_1['LL_strong']
                # LH_strong = augment_1['LH_strong']
                # HL_strong = augment_1['HL_strong']
                # HH_strong = augment_1['HH_strong']
            else:
                augment_1 = self.augmentation_1(image=img, mask=mask, L=L, H=H, LH_=LH_,
                                                # LL=LL, LH=LH, HL=HL, HH=HH
                                                )
                img_1 = augment_1['image']
                # if 'val' not in img_path_1:
                #     img_1 = A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.2, p=0.5)(image=img_1)['image'] # 注意这里有区别的，主要是val和train是不一样的
                # img_1 = A.Blur(p=0.05)(image=img_1)['image']
                mask_1 = augment_1['mask']
                L = augment_1['L']
                H = augment_1['H']
                LH_ = augment_1['LH_']
                # LL = augment_1['LL']
                # LH = augment_1['LH']
                # HL = augment_1['HL']
                # HH = augment_1['HH']
            
            if self.augmentation_strong_1 is not None:
                normalize_1 = self.normalize_1(image=img_1, mask=mask_1, L=L, H=H, 
                                            #    LL=LL, LH=LH, HL=HL, HH=HH, 
                                        image_strong=img_strong_1, L_strong=L_strong, H_strong=H_strong, 
                                        # LL_strong=LL_strong, LH_strong=LH_strong, HL_strong=HL_strong, HH_strong=HH_strong
                                        )
                img_1 = normalize_1['image']
                L = normalize_1['L']
                H = normalize_1['H']
                mask_1 = normalize_1['mask'].long()
                # LL = normalize_1['LL']
                # LH = normalize_1['LH']
                # HL = normalize_1['HL']
                # HH = normalize_1['HH']
                img_strong_1 = normalize_1['image_strong']
                L_strong = normalize_1['L_strong']
                H_strong = normalize_1['H_strong']
                # LL_strong = normalize_1['LL_strong']
                # LH_strong = normalize_1['LH_strong']
                # HL_strong = normalize_1['HL_strong']
                # HH_strong = normalize_1['HH_strong']
            else:
                normalize_1 = self.normalize_1(
                    image=img_1, 
                    mask=mask_1, 
                                            #    L=L, H=H, LH_=LH_
                                            #    LL=LL, LH=LH, HL=HL, HH=HH
                                               )
                img_1 = normalize_1['image']
                # img_strong1 = normalize_1['image_strong']
                # img_1 = (LH_gamma + 0.2) * img_1  # 将这里也改变一下吧，失败的改变
                # L = normalize_1['L']
                # H = normalize_1['H']
                # img_1 = ToTensorV2()(image=img_1)['image']
                mask_1 = normalize_1['mask'].long()
                # LH_ = normalize_1['LH_']
                # LL = normalize_1['LL']
                # LH = normalize_1['LH']
                # HL = normalize_1['HL']
                # HH = normalize_1['HH']
                L = ToTensorV2()(image=L)['image']
                H = ToTensorV2()(image=H)['image']
                LH_ = ToTensorV2()(image=LH_)['image']

            if self.augmentation_strong_1 is not None:
                sampel = {'image': img_1, 'mask': mask_1, 'L': L, 'H': H,
                        #   'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH,
                        'image_strong': img_strong_1, 'L_strong': L_strong, 'H_strong': H_strong, 
                        # 'LL_strong': LL_strong, 'LH_strong': LH_strong, 'HL_strong': HL_strong, 'HH_strong': HH_strong,  
                        'ID': os.path.split(mask_path)[1]}
            else:
                sampel = {'image': img_1, 
                        #   'image_strong': img_strong1,
                          'mask': mask_1, 'L': L, 'H': H, 'LH_': LH_,
                        #   'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH, 
                          'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img, L=L, H=H, LH_=LH_)
            img_1 = augment_1['image']
            L = augment_1['L']
            H = augment_1['H']
            LH_ = augment_1['LH_']
            # LL = augment_1['LL']
            # LH = augment_1['LH']
            # HL = augment_1['HL']
            # HH = augment_1['HH']

            normalize_1 = self.normalize_1(image=img_1, L=L, H=H, LH_=LH_)
            img_1 = normalize_1['image']
            # L = normalize_1['L']
            # H = normalize_1['H']
            # LH_ = normalize_1['LH_']

            L = ToTensorV2()(image=L)['image']
            H = ToTensorV2()(image=H)['image']
            LH_ = ToTensorV2()(image=LH_)['image']
            # LL = normalize_1['LL']
            # LH = normalize_1['LH']
            # HL = normalize_1['HL']
            # HH = normalize_1['HH']

            sampel = {'image': img_1, 'L': L, 'H': H,  'LH_': LH_, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_XNetv2(data_dir, data_transform_1, data_normalize_1, wavelet_type, alpha, beta, gamma, scale, data_strong_transform_1=None, 
                    #    data_strong_resize_1=None, 
                       sup=True, num_images=None, **kwargs):
    dataset = dataset_XNetv2(data_dir=data_dir,
                          augmentation_1=data_transform_1,
                          augmentation_strong_1=data_strong_transform_1,
                        #   augmentation_strong_resize_1=data_strong_resize_1, 
                          normalize_1=data_normalize_1,        
                          wavelet_type=wavelet_type,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma,
                          scale=scale,
                          sup=sup,
                          num_images=num_images,
                           **kwargs)
    return dataset

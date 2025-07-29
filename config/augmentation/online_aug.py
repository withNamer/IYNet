import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchio import transforms as T
import torchio as tio

def data_transform_2d(size):
    data_transforms = {
        'train': A.Compose([
            A.Resize(size[0], size[1], p=1),
            A.Flip(p=0.75),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.02, p=0.8),  # 注意这里是怎么变化的呢？
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
            additional_targets={'L': 'image', 'H': 'image', 'LH_': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image',
                                'image_strong': 'image', 'L_strong': 'image', 'H_strong': 'image', 
                                # 'LL_strong': 'image', 'LH_strong': 'image', 'HL_strong': 'image', 'HH_strong': 'image',
                                }
        ),
        'val': A.Compose([
            A.Resize(size[0], size[1], p=1),
        ],
            additional_targets={'L': 'image', 'H': 'image', 'LH_': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image'
                                }
        ),
        'test': A.Compose([
            A.Resize(size[0], size[1], p=1),
        ],
            additional_targets={'L': 'image', 'H': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image'
                                }
        )
    }
    return data_transforms

def data_strong_resize_2d(size):
    data_transforms = {'train': A.Compose([
            A.Resize(size[0], size[1], p=1),
            # A.Flip(p=0.75),
            # A.Transpose(p=0.5),
            # A.RandomRotate90(p=1),
        ],
            additional_targets={'L': 'image', 'H': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image'
                                }
        ),
        }
    return data_transforms

def data_transform_strong_2d(): # size
    data_transforms = {
        'train': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.GaussNoise(var_limit=(0.1, 1.0), p=0.2),
        ],
            additional_targets={'L': 'image', 'H': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image', 
                                }
        ),
        # 'val': A.Compose([
        #     A.Resize(size[0], size[1], p=1),
        # ],
        #     additional_targets={'L': 'image', 'H': 'image'}
        # ),
        # 'test': A.Compose([
        #     A.Resize(size[0], size[1], p=1),
        # ],
        #     additional_targets={'L': 'image', 'H': 'image'}
        # )
    }
    return data_transforms

def data_normalize_2d(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'L': 'image', 'H': 'image', 'LH_': 'image', 
                                # 'LL': 'image', 'LH': 'image', 'HL': 'image', 'HH': 'image',
                                'image_strong': 'image',
                            #    'L_strong': 'image', 'H_strong': 'image', 
                            #    'LL_strong': 'image', 'LH_strong': 'image', 'HL_strong': 'image', 'HH_strong': 'image',
                               }
    )
    return data_normalize


def data_transform_3d(normalization):
    data_transform = {
        'train': T.Compose([
            T.RandomFlip(),
            T.RandomBiasField(coefficients=(0.12, 0.15), order=2, p=0.2),
            T.OneOf({
               T.RandomNoise(): 0.5,
               T.RandomBlur(std=1): 0.5,
            }, p=0.2),
            T.ZNormalization(masking_method=normalization),
        ]),
        'val': T.Compose([
            T.ZNormalization(masking_method=normalization),
        ]),
        'test': T.Compose([
            T.ZNormalization(masking_method=normalization),
        ])
    }

    return data_transform
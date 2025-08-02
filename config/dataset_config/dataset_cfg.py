import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'EPFL':
            {
                'PATH_DATASET': '/ldap_shared/home/xxx/wavelet/dataset/EPFL',
                'PATH_TRAINED_MODEL': '/ldap_shared/home/xxx/IYNet/checkpoints/EPFL',
                'PATH_SEG_RESULT': '/ldap_shared/home/xxx/wavelet/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                # 'MEAN': [0.53475095],
                'MEAN': [0.55090925],
                # 'STD': [0.12427514],
                'STD': [0.11919559],
                'INPUT_SIZE': (256, 256),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())

            },
        'GlaS':
            {
                'PATH_DATASET': './dataset/GlaS',
                'PATH_TRAINED_MODEL': './checkpoints',
                'PATH_SEG_RESULT': './seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.428206, 0.507778, 0.426366],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'kvasir_seg':
            {
                'PATH_DATASET': './dataset/kvasir_seg',
                'PATH_TRAINED_MODEL': './checkpoints',
                'PATH_SEG_RESULT': './seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.55460666,0.32034614,0.23538549],
                'STD': [0.30688183,0.21513945,0.17803254],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'ISIC-2017':
            {
                'PATH_DATASET': '/ldap_shared/home/xxx/wavelet/dataset/ISIC-2017',
                'PATH_TRAINED_MODEL': '/ldap_shared/home/xxx/wavelet/checkpoints/ISIC-2017',
                'PATH_SEG_RESULT': '/ldap_shared/home/xxx/wavelet/seg_pred/semi/ISIC-2017',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.699002, 0.556046, 0.512134],
                'STD': [0.365650, 0.317347, 0.339400],
                'INPUT_SIZE': (128, 128),
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'P-CT':
            {
                'PATH_DATASET': '/ldap_shared/home/xxx/wavelet/dataset/P-CT',
                'PATH_TRAINED_MODEL': '/mnt/data1/fsw_data/P-CT',
                # 'PATH_TRAINED_MODEL': '/mnt/data1/fsw_code/checkpoints',
                'PATH_SEG_RESULT': '/ldap_shared/home/xxx/wavelet/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (96, 96, 96),
                'PATCH_OVERLAP': (80, 80, 80),
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8,
                'QUEUE_LENGTH': 48
            },
        'LA':
            {
                'PATH_DATASET': '/ldap_shared/home/xxx/wavelet/dataset/LA',
                'PATH_TRAINED_MODEL': '/ldap_shared/home/xxx/wavelet/checkpoints',
                'PATH_SEG_RESULT': '/ldap_shared/home/xxx/wavelet/seg_pred',
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (96, 96, 80),
                'PATCH_OVERLAP': (80, 80, 64),
                # 'FORMAT': '.nrrd',
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8,
                'QUEUE_LENGTH': 48
            },
    }

    return config[dataet_name]

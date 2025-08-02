from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
from torch.backends import cudnn
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchio as tio
import pywt
from torchio import transforms as T

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
from config.train_test_config.train_test_config import save_test_3d
from warnings import simplefilter
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_model', default='/ldap_shared/home/xxx/wavelet/checkpoints/LA/MFwaveNet_3D-l=0.05-e=400-s=50-g=0.5-b=1-cw=0.2-w=20-100-db2-0.0-0.4-0.0-0.4/best_MF_Jc_0.8877.pth')
    parser.add_argument('--dataset_name', default='LA', help='P-CT, LiTS')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--val_alpha', default=[0.2, 0.2])
    parser.add_argument('--val_beta', default=[0.2, 0.2])
    parser.add_argument('--val_gamma', default=[0.7, 0.7])
    parser.add_argument('-n', '--network', default='MFwaveNet_3D')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Results Save
    path_seg_results = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)
    path_seg_results = path_seg_results + '/' + str(os.path.splitext(os.path.split(args.path_model)[1])[0])
    if not os.path.exists(path_seg_results) and rank == args.rank_index:
        os.mkdir(path_seg_results)

    path_seg_results2 = cfg['PATH_SEG_RESULT'] + '/' + str(dataset_name)
    if not os.path.exists(path_seg_results2) and rank == args.rank_index:
        os.mkdir(path_seg_results2)
    path_seg_results2 = path_seg_results2 + '/' + str(os.path.splitext(os.path.split(args.path_model)[1])[0])
    if not os.path.exists(path_seg_results2) and rank == args.rank_index:
        os.mkdir(path_seg_results2)

    data_transform = data_transform_3d(cfg['NORMALIZE'])
    dataset_val = dataset_it(
        data_dir=cfg['PATH_DATASET'] + '/val',
        transform_1=data_transform['test'],
    )

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

    # if rank == args.rank_index:
    #     state_dict = torch.load(args.path_model, map_location=torch.device(args.local_rank))
    #     model.load_state_dict(state_dict=state_dict)
    # model = DistributedDataParallel(model, device_ids=[args.local_rank])

    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    state_dict = torch.load(args.path_model)
    model.load_state_dict(state_dict=state_dict)
    dist.barrier()

    # L H augmentation
    L_H_aug = T.Compose([T.Resize(cfg['PATCH_SIZE']), 
                        #  T.ZNormalization(masking_method=cfg['NORMALIZE'])
                        ]
                         )

    # Test
    since = time.time()

    for i, subject in enumerate(dataset_val.dataset_1):

        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=cfg['PATCH_SIZE'],
            patch_overlap=cfg['PATCH_OVERLAP']
        )

        # val_sampler = torch.utils.data.distributed.DistributedSampler(grid_sampler, shuffle=False)

        dataloaders = dict()
        dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
        # dataloaders['test'] = DataLoader(grid_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
        # aggregator2 = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

        with torch.no_grad():
            model.eval()

            for data in dataloaders['test']:

                inputs_test1 = Variable(data['image'][tio.DATA].cuda())
                inputs_test2 = torch.zeros_like(inputs_test1, device='cpu')
                inputs_test3 = torch.zeros_like(inputs_test1, device='cpu')
                inputs_test4 = torch.zeros_like(inputs_test1, device='cpu')
                inputs_test_numpy = inputs_test1.cpu().detach().numpy()
                for j in range(inputs_test_numpy.shape[0]):
                    img = inputs_test_numpy[j, 0]
                    img_wavelet = pywt.dwtn(img, args.wavelet_type)
                    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = img_wavelet['aaa'], img_wavelet['aad'], img_wavelet['ada'], img_wavelet['add'], img_wavelet['daa'], img_wavelet['dad'], img_wavelet['dda'], img_wavelet['ddd']

                    L_2 = LLL ** 2
                    H_2 = LLH ** 2 + LHL ** 2 + LHH ** 2 + HLL ** 2 + HLH ** 2 + HHL ** 2 + HHH ** 2

                    LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) 
                    LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) 
                    LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) 
                    LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) 
                    HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) 
                    HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) 
                    HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) 
                    HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) 

                    L_2 = (L_2 - L_2.min()) / (L_2.max() - L_2.min()) * 0.5
                    H_2 = (H_2 - H_2.min()) / (H_2.max() - H_2.min()) * 0.5
                    
                    H_ = LLH + LHL + LHH + HLL + HLH + HHL + HHH
                    H_ = (H_ - H_.min()) / (H_.max() - H_.min()) 
                    
                    L_alpha = random.uniform(args.val_alpha[0], args.val_alpha[1])
                    H_beta = random.uniform(args.val_beta[0], args.val_beta[1])

                    L = LLL + L_alpha * H_ + L_alpha * H_2
                    L = (L - L.min()) / (L.max() - L.min()) 
                    
                    H = H_ + H_beta * LLL + H_beta * L_2
                    H = (H - H.min()) / (H.max() - H.min()) 

                    LH_gamma = random.uniform(args.val_gamma[0], args.val_gamma[1])
                    LH_ = LH_gamma * LLL + (1 - LH_gamma) * H_
                    LH_ = (LH_ - LH_.min()) / (LH_.max() - LH_.min()) 

                    L = (L_alpha  + 0.6) * L 
                    H = (H_beta  + 0.6) * H 
                    LH_ = (LH_gamma + 0.0) * LH_ 

                    L = torch.tensor(L).unsqueeze(0)
                    H = torch.tensor(H).unsqueeze(0)
                    LH_ = torch.tensor(LH_).unsqueeze(0)

                    L = L_H_aug(L)
                    H = L_H_aug(H)
                    LH_ = L_H_aug(LH_)
                    inputs_test2[j] = L
                    inputs_test3[j] = H
                    inputs_test4[j] = LH_
                inputs_test2 = Variable(inputs_test2.cuda())
                inputs_test3 = Variable(inputs_test3.cuda())
                inputs_test4 = Variable(inputs_test4.cuda())
                location_test = data[tio.LOCATION]
                inputs_test1 = torch.cat([inputs_test1, inputs_test4], dim=1)
                outputs_test_1, outputs_test_2 = model(inputs_test2, inputs_test3, inputs_test1)
                aggregator.add_batch(outputs_test_1, location_test)
                # aggregator2.add_batch(outputs_test_2, location_test)

        outputs_tensor = aggregator.get_output_tensor()
        # outputs_tensor2 = aggregator2.get_output_tensor()
        save_test_3d(cfg['NUM_CLASSES'], outputs_tensor, subject['ID'], args.threshold, path_seg_results, subject['image']['affine'])
        # save_test_3d(cfg['NUM_CLASSES'], outputs_tensor2, subject['ID'], args.threshold, path_seg_results2, subject['image']['affine'])


    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)

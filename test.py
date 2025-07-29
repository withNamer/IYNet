import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
from torch.backends import cudnn
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_2d, data_normalize_2d
from models.getnetwork import get_network
from dataload.dataset_2d import imagefloder_XNetv2
from config.train_test_config.train_test_config import print_test_eval, save_test_2d


from warnings import simplefilter
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
    parser.add_argument('-p', '--path_model', default='/ldap_shared/home/s_fsw/XNetv2/checkpoints/EPFL/EPFL/MFwavelet-l=0.5-e=300-s=50-g=0.5-b=4-uw=0.5-w=20-20-80/best_XNetv2_Jc_0.8171.pth')
    parser.add_argument('--dataset_name', default='EPFL', help='GlaS, CREMI')
    parser.add_argument('--if_mask', default=True)
    parser.add_argument('--threshold', default=0.50)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, coif1, dmey')
    parser.add_argument('--alpha', default=[0.2, 0.2])
    parser.add_argument('--beta', default=[0.2, 0.2])
    parser.add_argument('--gamma', default=[0.7, 0.7])
    parser.add_argument('--scale', default=[0.7, 0.7])
    parser.add_argument('-n', '--network', default='MFwavelet')
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
    # print(path_seg_results)

    # Dataset
    data_transforms = data_transform_2d(cfg['INPUT_SIZE'])
    data_normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])

    dataset_val = imagefloder_XNetv2(
        data_dir=cfg['PATH_DATASET'] + '/val',
        data_transform_1=data_transforms['val'],
        data_normalize_1=data_normalize,
        wavelet_type=args.wavelet_type,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        scale=args.scale,
        sup=True,
        num_images=None
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    dataloaders = dict()
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16, sampler=val_sampler)

    num_batches = {'val': len(dataloaders['val'])}

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

    # Test
    since = time.time()

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(dataloaders['val']):
            inputs_test_1 = Variable(data['L'].cuda())
            inputs_test_2 = Variable(data['H'].cuda())
            # inputs_test_1 = torch.cat((Variable(data['L'].cuda()), Variable(data['H'].cuda())))
            # inputs_train_3 = Variable(data['H'].cuda())
            inputs_test_3 = torch.cat((Variable(data['image'].cuda()), Variable(data['image'].cuda()), 
            #                         #    Variable(data['LH'].cuda()), Variable(data['HL'].cuda()),
            #                             # Variable(data['H'].cuda()),
            ), dim=1)
            # inputs_test_3 = Variable(data['image'].cuda())
            # inputs_test_4 = Variable(data['LH_'].cuda())
            name_test = data['ID']


            if args.if_mask:
                mask_test = data['mask']
                # from PIL import Image
                # print(name_test)
                # mask_path = os.path.join("/ldap_shared/home/s_fsw/wavelet/dataset/EPFL/val/mask", name_test[0])
                # a = np.array(Image.open(mask_path))
                # a = (a > 128).astype(np.uint8)
                # import cv2
                # a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_NEAREST)
                # print(mask_test)
                # print(mask_test.shape)
                # print(a)
                # print(a.shape)
                # print(np.sum(a==mask_test[0].numpy()))

                mask_test = Variable(mask_test.cuda())

            outputs_test1, outputs_test2, = model(inputs_test_2, inputs_test_1, inputs_test_3)

            if args.if_mask:
                if i == 0:
                    score_list_test = (outputs_test2 + outputs_test2) / 2
                    name_list_test = name_test
                    mask_list_test = mask_test
                else:
                # elif 0 < i <= num_batches['val'] / 16:
                    score_list_test = torch.cat((score_list_test, (outputs_test2 + outputs_test2) / 2), dim=0)
                    name_list_test = np.append(name_list_test, name_test, axis=0)
                    mask_list_test = torch.cat((mask_list_test, mask_test), dim=0)
                torch.cuda.empty_cache()
            else:
                save_test_2d(cfg['NUM_CLASSES'], (outputs_test2 + outputs_test2) / 2, name_test, args.threshold, path_seg_results, cfg['PALETTE'])
                torch.cuda.empty_cache()

        if args.if_mask:
            score_gather_list_test = [torch.zeros_like(score_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_test, score_list_test)
            score_list_test = torch.cat(score_gather_list_test, dim=0)

            mask_gather_list_test = [torch.zeros_like(mask_list_test) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_test, mask_list_test)
            mask_list_test = torch.cat(mask_gather_list_test, dim=0)

            name_gather_list_test = [None for _ in range(ngpus_per_node)]
            torch.distributed.all_gather_object(name_gather_list_test, name_list_test)
            name_list_test = np.concatenate(name_gather_list_test, axis=0)

        if args.if_mask and rank == args.rank_index:
            print('=' * print_num)
            # print(score_list_test)
            # print("mask", mask_list_test)
            test_eval_list = print_test_eval(cfg['NUM_CLASSES'], score_list_test, mask_list_test, print_num_minus)
            # print(cfg['PALETTE'])

            # score_list_test_ = torch.softmax(score_list_test, dim=1)
            # pred_results = score_list_test_[:, 1, ...].cpu().numpy()
            # pred_results[pred_results > test_eval_list[0]] = 1
            # pred_results[pred_results <= test_eval_list[0]] = 0
            # mask_list_test_ = mask_list_test.cpu().numpy()#.flatten()
            # print(np.unique(mask_list_test_))
            # print(np.unique(pred_results))
            # pred_results_ = pred_results.flatten()
            # sum_area = (pred_results_ + mask_list_test_)
            # tp = float(np.sum(sum_area == 2))
            # union = np.sum(sum_area == 1)
            # print(tp / float(union + tp))
            # print(2 * tp / float(union + 2 * tp))
            # from medpy.metric.binary import dc, jc, hd95, asd
            # print(dc(pred_results, mask_list_test_))
            # print(jc(pred_results, mask_list_test_))
            # print(hd95(pred_results, mask_list_test_))
            # # print(hd95(mask_list_test_, mask_list_test_))
            # print(asd(pred_results, mask_list_test_))

            # from PIL import Image
            # print(dc(pred_results, mask_list_test_))
            # img_volume = np.zeros((mask_list_test_.shape), dtype=np.uint8)
            # mask_volume = np.zeros((mask_list_test_.shape), dtype=np.uint8)
            # for idx, i in enumerate(name_list_test):
            #     mask_path = os.path.join("/ldap_shared/home/s_fsw/wavelet/dataset/EPFL/val/mask", i)
            #     a = np.array(Image.open(mask_path).resize((256, 256), Image.NEAREST))
            #     mask_volume[idx] = (a > 128)
            # for idx, i in enumerate(name_list_test):
            #     pred_path = os.path.join("/ldap_shared/home/s_fsw/wavelet/seg_pred/EPFL/best_XNetv2_Jc_0.8335", i)
            #     img_volume[idx] = np.array(Image.open(pred_path))
            
            # print(jc(img_volume, mask_volume))
            # print(dc(img_volume, mask_volume))
            
            save_test_2d(cfg['NUM_CLASSES'], score_list_test, name_list_test, test_eval_list[0], path_seg_results, cfg['PALETTE'])

            torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)
        print('-' * print_num)
        print('| Testing Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('=' * print_num)
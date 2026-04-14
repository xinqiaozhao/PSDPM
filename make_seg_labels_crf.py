import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import argparse
import numpy as np
import importlib
import os
import imageio
from PIL import Image

from data import data_voc
from tool import torchutils, indexing, imutils

cudnn.enabled = True

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def _work(process_id, dataset, args):

    # n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=4, pin_memory=False)

    cam_out_dir = os.path.join(args.session_name, 'npy')

    densecrf = DenseCRF(iter_max=8,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,)

    mean_bgr = (104.008, 116.669, 122.675)

    with torch.no_grad():

        # model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            orig_img_size = np.array(pack['size'])

            # edge, _ = model(pack['img'][0][0].cuda(non_blocking=True))

            cam_dict = np.load(cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
            
            cams =(cam_dict['IS_CAM1'])
            cams = torch.tensor(cams)


            cams = F.interpolate(torch.tensor(cams).unsqueeze(1), (orig_img_size[0],orig_img_size[1]), mode='bilinear')[:,0]
            # cams = np.power(cams, 1.3)
            # print(cams.size())
            
            keys = cam_dict['keys']
            # print(edge.size())
            # print(torch.tensor(cams).unsqueeze(1).size())
            # cams = F.interpolate(torch.tensor(cams).unsqueeze(1), edge.shape[1:], mode='bilinear', align_corners=False)[:,0]
            # cams = np.power(cams, 1.3)
            # cam_downsized_values = torch.tensor(cams.unsqueeze(1))[1:, ...].cuda()
            # # rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            # rw_up = F.interpolate(cam_downsized_values, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            # rw_up = rw_up / torch.max(rw_up)
            # rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=0.25)

            # cams = F.interpolate(torch.tensor(cams).unsqueeze(1), edge.shape[1:], mode='bilinear', align_corners=False)[:,0]
            # cams = np.power(cams, 1.3)
            # cam_downsized_values = torch.tensor(cams.unsqueeze(1)).cuda()
            # rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            # rw_up = F.interpolate(cam_downsized_values, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            # rw_up = rw_up / torch.max(rw_up)

            # print((pack['img'])[1].shape)
            bgfgsum = cams
            bgfgsum = ((bgfgsum)).cpu().numpy()
            # bgfgsum = (F.softmax(bgfgsum,dim=0)).cpu().numpy()

            imgdensecrf = pack['img'][1][0].cpu().numpy().astype(np.float32)
            imgdensecrf -= mean_bgr
            rw_up_bgsum = torch.tensor(densecrf(imgdensecrf.astype(np.uint8), (bgfgsum)))
            # rw_up = torch.tensor(densecrf(pack['img'][1][0], rw_up.cpu().numpy()))
            # rw_up_bgsum = (torch.tensor(bgfgsum) + rw_up_bgsum)/2
            # rw_up_bgsum = torch.tensor(bgfgsum)

            rw_pred = torch.argmax(rw_up_bgsum, dim=0)
            rw_pred=rw_pred.cpu().numpy()
            rw_pred = keys[rw_pred]
            rw_pred = torch.tensor(rw_pred)
            rw_pred[torch.max((rw_up_bgsum), dim=0)[0] < 0.5] = 255
            rw_pred = rw_pred.cpu().numpy()

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            # out = Image.fromarray(rw_pred.astype(np.uint8), mode='P')
            # out.putpalette(palette)
            # out.save(os.path.join(os.path.join(args.sem_seg_out_dir, img_name + '_palette.png')))

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--infer_list", default="data/trainaug_voc.txt", type=str)
    parser.add_argument("--voc12_root", default="/data/nips/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--irn_network", default="network.resnet50_irn", type=str)
    parser.add_argument("--session_name", default="exp", type=str)
    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_out_dir", default="", type=str)
    args = parser.parse_args()

    # model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    # irn_weights_path = os.path.join(args.session_name, 'ckpt', 'irn.pth')
    # model.load_state_dict(torch.load(irn_weights_path), strict=False)
    # model.eval()
    # n_gpus = torch.cuda.device_count()
    prosss = 30
    dataset = data_voc.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root, scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, prosss)

    args.sem_seg_out_dir = os.path.join(args.session_name, 'pseudo_label')
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=prosss, args=( dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()

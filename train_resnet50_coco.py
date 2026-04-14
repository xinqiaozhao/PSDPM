import torch, os
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import importlib
import numpy as np
from tensorboardX import SummaryWriter
from data import data_coco, data_voc
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tool import pyutils, torchutils, visualization, imutils
import random
import torchvision
from libs.models import DeepLabV2_ResNet101_MSC,DeepLabV3_ResNet101_MSC,DeepLabV3Plus_ResNet101_MSC
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torchvision.transforms as T
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # print(img.size())
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        mask = mask.to(img.device)
        img = img * mask

        return img

# normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
RandomHorizontalFlip = T.RandomHorizontalFlip()
RandomSolarize = T.RandomSolarize(192)
ColorJitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
nclaug4_2 = T.RandomGrayscale(p=0.2)
GaussianBlur = T.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
# aug1 = T.RandAugment(num_ops=2, magnitude=20)
augtopil = T.ToPILImage()
augtotensor = T.PILToTensor()
nclaug4 = Cutout(n_holes=5, length=8)

class TorchvisionNormalize1val():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = (img).float()
        proc_img = torch.empty_like(imgarr)

        proc_img[:, :, :, 0] = (imgarr[:, :, :, 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[:, :, :, 1] = (imgarr[:, :, :, 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[:, :, :, 2] = (imgarr[:, :, :, 2] / 255. - self.mean[2]) / self.std[2]

        proc_img = torch.permute(proc_img,[0,3,1,2]).contiguous()

        # proc_img2 = torch.cat([torch.where(torch.isclose(torch.tensor(proc_img[:,0,:,:]),torch.tensor(-self.mean[0] / self.std[0])),0.,proc_img[:,0,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,1,:,:]),torch.tensor(-self.mean[1] / self.std[1])),0.,proc_img[:,1,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,2,:,:]),torch.tensor(-self.mean[2] / self.std[2])),0.,proc_img[:,2,:,:]).unsqueeze(1)],dim=1)

        return proc_img

class TorchvisionNormalize1():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = torch.permute(img,[0,3,1,2])
        rdi = random.randint(0,2)
        if rdi == 0:
            # img = RandomHorizontalFlip(img)
            # img = ColorJitter(img)
            # b = img.clone()
            # for i in range(len(b)):
            #     a = augtopil(b[i])
            #     a = GaussianBlur(a)
            #     a = augtotensor(a)
            #     b[i] = a
            # img = b.clone().detach()
            img = nclaug4_2(img)
            # img = RandomSolarize(img)

        elif rdi == 1:
            img = RandomHorizontalFlip(img)
            img = ColorJitter(img)
            b = img.clone()
            for i in range(len(b)):
                a = augtopil(b[i])
                a = GaussianBlur(a)
                a = augtotensor(a)
                b[i] = a
            img = b.clone().detach()
        else:
            pass
        
        img = torch.permute(img,[0,2,3,1])
        imgarr = (img).float()
        proc_img = torch.empty_like(imgarr)

        proc_img[:, :, :, 0] = (imgarr[:, :, :, 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[:, :, :, 1] = (imgarr[:, :, :, 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[:, :, :, 2] = (imgarr[:, :, :, 2] / 255. - self.mean[2]) / self.std[2]

        proc_img = torch.permute(proc_img,[0,3,1,2]).contiguous()
        # rdi = random.randint(0, 2)
        # if rdi ==1:
        #     for i in range(len(proc_img)):
        #         proc_img[i] = nclaug4(proc_img[i])

        # proc_img2 = torch.cat([torch.where(torch.isclose(torch.tensor(proc_img[:,0,:,:]),torch.tensor(-self.mean[0] / self.std[0])),0.,proc_img[:,0,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,1,:,:]),torch.tensor(-self.mean[1] / self.std[1])),0.,proc_img[:,1,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,2,:,:]),torch.tensor(-self.mean[2] / self.std[2])),0.,proc_img[:,2,:,:]).unsqueeze(1)],dim=1)

        return proc_img

class TorchvisionNormalizedeeplab1():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        proc_img = (img).float()
        proc_img = torch.permute(proc_img, [0, 3, 1, 2]).contiguous()
        # proc_img = torch.empty_like(imgarr)
        #
        # proc_img[:, 0, :, :] = (imgarr[:, 0, :, :] / 255. - self.mean[0]) / self.std[0]
        # proc_img[:, 1, :, :] = (imgarr[:, 1, :, :] / 255. - self.mean[1]) / self.std[1]
        # proc_img[:, 2, :, :] = (imgarr[:, 2, :, :] / 255. - self.mean[2]) / self.std[2]
        #
        # proc_img2 = torch.cat([torch.where(torch.isclose(torch.tensor(proc_img[:,0,:,:]),torch.tensor(-self.mean[0] / self.std[0])),0.,proc_img[:,0,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,1,:,:]),torch.tensor(-self.mean[1] / self.std[1])),0.,proc_img[:,1,:,:]).unsqueeze(1),\
        #                        torch.where(torch.isclose(torch.tensor(proc_img[:,2,:,:]),torch.tensor(-self.mean[2] / self.std[2])),0.,proc_img[:,2,:,:]).unsqueeze(1)],dim=1)

        return proc_img

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

kllossc = torch.nn.KLDivLoss(reduction='batchmean')


TorchvisionNormalize = TorchvisionNormalize1()
TorchvisionNormalizeval = TorchvisionNormalize1val()
TorchvisionNormalizedeeplab = TorchvisionNormalizedeeplab1()

def validate(model, data_loader, global_step,tblogger):
    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir="/data/tfl/VOCdevkit/VOC2012")
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]
    print('validating ... ', flush=True, end='')
    model.eval()
    with torch.no_grad():
        preds = []
        preds1 = []
        preds2 = []
        for iter, pack in enumerate(data_loader):       
            img = pack['img'].cuda(3).float()
            img = TorchvisionNormalizeval(img)
            label = pack['label'].cuda(3).unsqueeze(-1).unsqueeze(-1)
            label = F.pad(label, (0, 0, 0, 0, 1, 0), 'constant', 1.0)
            outputs = model.forward(img, label, pack['label'].cuda(3))
            IS_cam1 = outputs['cam1']
            IS_cam1 = F.interpolate(IS_cam1, img.shape[2:], mode='bilinear')
            IS_cam1 = IS_cam1/(F.adaptive_max_pool2d(IS_cam1, (1, 1)) + 1e-5)
            cls_labels_bkg1 = torch.argmax(IS_cam1, 1)
            preds1.append(cls_labels_bkg1[0].cpu().numpy().copy())

            IS_cam = outputs['cam2']
            IS_cam = F.interpolate(IS_cam, img.shape[2:], mode='bilinear')
            IS_cam = IS_cam/(F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            cls_labels_bkg = torch.argmax(IS_cam, 1)
            preds.append(cls_labels_bkg[0].cpu().numpy().copy())


        confusion = calc_semantic_segmentation_confusion(preds1, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        confusion = calc_semantic_segmentation_confusion(preds, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou1 = gtjresj / denominator

        print('\n')
        print({'iou': iou, 'miou': np.nanmean(iou)})
        print('\n')
        print({'iou1': iou1, 'miou': np.nanmean(iou1)})
    model.train()

    return np.nanmean(iou) + np.nanmean(iou1)

def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def shuffle_batch(x, y,z):
    index = torch.randperm(x.size(0))
    x = x[index]
    y = y[index]
    z= z[index]
    return x, y, z



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=6, type=int)
    parser.add_argument("--network", default="network.resnet50_SFC", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp4", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--tf_freq", default=100, type=int)
    parser.add_argument("--val_freq", default=300, type=int)
    parser.add_argument("--dataset", default="coco", type=str)
    parser.add_argument("--dataset_root", default="/data/tfl/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    setup_seed(args.seed)
    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'ckpt'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '/data/tfl/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=21)
        train_dataset = data_voc.VOC12ClsDataset('data/trainaug_' + args.dataset + '.txt', voc12_root=dataset_root,
                                                                    resize_long=(480, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_voc.VOC12ClsDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    elif args.dataset == 'coco':
        args.tf_freq = 99999
        args.val_freq = 99999
        dataset_root = '/data/nips/COCO'
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=81)
        train_dataset = data_coco.COCOClsDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_coco.COCOClsDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    state_dict = torch.load('exp/ckpt/final.pth')
    # model.load_state_dict(state_dict['net'], strict=True)
    model = torch.nn.DataParallel(model,device_ids=[3])
    model.train()


    bestiou = 0
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()
    # slot = torch.zeros([20,3,512,512])
    # vm = torch.zeros([20,21,512,512])
    # sltarget = torch.zeros([20,20])
    # target2 = torch.linspace(0, 19, 20, dtype=torch.long).cuda()
    # deepv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).cuda(1)

    deepv3 = DeepLabV2_ResNet101_MSC(n_classes=21)
    state_dict = torch.load('checkpoint_final.pth')
    for m in deepv3.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    deepv3.load_state_dict(state_dict, strict=True)  # to skip ASPP
    deepv3 = torch.nn.DataParallel(deepv3, device_ids=[0,1,2]).cuda()
    deepv3.eval()

    densecrf = DenseCRF(iter_max=10,
                        pos_xy_std=1,
                        pos_w=3,
                        bi_xy_std=67,
                        bi_rgb_std=3,
                        bi_w=4, )

    # mean_bgr = (104.008, 116.669, 122.675)

    for ep in range(args.max_epoches):
        
        print('Epoch %d/%d' % (ep + 1, args.max_epoches))
        index = 0
        for step, pack in enumerate(train_data_loader):

            # print(pack['name'])

            scale_factor = 0.5
            img1 = pack['img']
            label = pack['label'].cuda(3)
            valid_mask = pack['valid_mask'].cuda(3)
            img1, label, valid_mask = shuffle_batch(img1, label, valid_mask)

            # ba = len(label)
            # for i in range(ba):
            #     argl = (label[i, :].squeeze() == 1).nonzero(as_tuple=False).squeeze()
            #     slot[argl] = img1[i].cpu()
            #     vm[argl] = valid_mask[i].cpu()
            #     sltarget[argl] = label[i].cpu()

            label = label.cuda(3)
            img1gt = TorchvisionNormalizedeeplab(img1.float()).clone()
            # print(img1gt.shape) 16,3,512,512
            img1 = img1.cuda(3)
            # print(img1gt.shape) 16,3,512,512
            # slot_c, target2_c, vm_c = shuffle_batch(slot, target2, vm)
            # first = random.randint(0,19)
            # second = random.randint(0,19)
            # third = random.randint(0,19)
            # four = random.randint(0,19)
            # label_bank = label
            # valid_mask_bank = valid_mask
            # img1_bank = img1

            # slot_c = slot.cuda(0)
            # target2_c = sltarget.cuda(0)
            # vm_c = vm.cuda(0)
            # img1 = torch.cat([img1, slot_c[first].unsqueeze(0), slot_c[second].unsqueeze(0), slot_c[third].unsqueeze(0), slot_c[four].unsqueeze(0)],dim=0)
            # label = torch.cat([label, target2_c[first].unsqueeze(0), target2_c[second].unsqueeze(0), target2_c[third].unsqueeze(0), target2_c[four].unsqueeze(0)],dim=0)
            # label_bank = label
            # valid_mask = torch.cat([valid_mask, vm_c[first].unsqueeze(0), vm_c[second].unsqueeze(0), vm_c[third].unsqueeze(0), vm_c[four].unsqueeze(0)],dim=0)
            img2gt = F.interpolate((img1gt),scale_factor=scale_factor,mode='bilinear',align_corners=True)

            N, c, h, w = img1gt.shape
            # print(img1gt.shape)
            my_label = label
            label = label.unsqueeze(2).unsqueeze(3)
            valid_mask[:,1:] = valid_mask[:,1:] * label
            # print(valid_mask.shape)
            valid_mask_lowres0 = F.interpolate(valid_mask, size=(h//16, w//16), mode='nearest')
            valid_mask_lowres1 = F.interpolate(valid_mask, size=(h//32, w//32), mode='nearest')
            
            bg_score = torch.ones((N,1)).cuda(3)
            label = torch.cat((bg_score.unsqueeze(-1).unsqueeze(-1), label), dim=1)

            # gtimg1 = deepv3((img1).clone().cuda(1))['out'].detach().cuda(0)
            # gtimg2 = deepv3((img2).clone().cuda(1))['out'].detach().cuda(0)

            # gtimg1_1 = deepv3((img1gt[:4]).clone().cuda(1)).detach().cuda(0)
            # gtimg2_1 = deepv3((img2gt[:4]).clone().cuda(1)).detach().cuda(0)
            #
            # gtimg1_2 = deepv3((img1gt[4:8]).clone().cuda(1)).detach().cuda(0)
            # gtimg2_2 = deepv3((img2gt[4:8]).clone().cuda(1)).detach().cuda(0)
            #
            # gtimg1 = torch.cat([gtimg1_1,gtimg1_2],dim=0)
            # gtimg2 = torch.cat([gtimg2_1, gtimg2_2], dim=0)
            #
            # gtimg1_2 = deepv3((img1gt[8:12]).clone().cuda(1)).detach().cuda(0)
            # gtimg2_2 = deepv3((img2gt[8:12]).clone().cuda(1)).detach().cuda(0)
            #
            # gtimg1 = torch.cat([gtimg1, gtimg1_2], dim=0)
            # gtimg2 = torch.cat([gtimg2, gtimg2_2], dim=0)
            #
            # gtimg1_2 = deepv3((img1gt[12:16]).clone().cuda(1)).detach().cuda(0)
            # gtimg2_2 = deepv3((img2gt[12:16]).clone().cuda(1)).detach().cuda(0)
            #
            # gtimg1 = torch.cat([gtimg1, gtimg1_2], dim=0)
            # gtimg2 = torch.cat([gtimg2, gtimg2_2], dim=0)
            # print(img1gt.shape)
            gtimg1 = deepv3((img1gt).cuda()).detach().cuda(3)
            # print(gtimg1.shape)
            gtimg2 = deepv3((img2gt).cuda()).detach().cuda(3)

            # imgdensecrf = img1gt.cpu().numpy()
            # # imgdensecrf -= mean_bgr
            # for gi in range(len(gtimg1)):
            #     gtimg1[gi] = torch.tensor(densecrf(imgdensecrf[gi].astype(np.uint8).transpose([1,2,0]), gtimg1[gi].numpy()))
            # gtimg1 = torch.tensor(gtimg1).cuda(3)
            #
            # imgdensecrf = img2gt.cpu().numpy()
            # # imgdensecrf -= mean_bgr
            # for gi in range(len(gtimg1)):
            #     gtimg2[gi] = torch.tensor(densecrf(imgdensecrf[gi].astype(np.uint8).transpose([1,2,0]), gtimg2[gi].numpy()))
            # gtimg2 = torch.tensor(gtimg2).cuda(3)

            gtimg1 = F.interpolate(gtimg1, size=(h // 16, w // 16), mode='bilinear')
            # belonging = gtimg1.argmax(1)
            # seeds = torch.zeros((len(img1), h // 16, w // 16, 21)).cuda()
            # gtimg1 = seeds.scatter_(-1, belonging.view(len(img1), h // 16, w // 16, 1), 1).permute(0, 3, 1, 2).contiguous()
            # print(gtimg1.shape) 16,21,32,32
            # gtimg1 = gtimg1 * valid_mask_lowres0

            gtimg2 = F.interpolate(gtimg2, size=(h // 32, w // 32), mode='bilinear')
            # belonging = gtimg2.argmax(1)
            # seeds = torch.zeros((len(img1), h // 32, w // 32, 21)).cuda()
            # gtimg2 = seeds.scatter_(-1, belonging.view(len(img1), h // 32, w // 32, 1), 1).permute(0, 3, 1, 2).contiguous()
            # gtimg2 = gtimg2 * valid_mask_lowres1

            # print(img1.shape) 16,512,512,3
            # print(valid_mask_lowres0.shape) 16,21,32,32
            # outputs1 = model.forward(img1, valid_mask_lowres0, my_label)
            # outputs2 = model.forward(img2, valid_mask_lowres1, my_label)
            img1 = TorchvisionNormalize(img1)
            # print(img1.shape)
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            # print(valid_mask_lowres0.shape)16,21,32,32
            # print(img1.shape)
            outputs1 = model.forward((img1), valid_mask_lowres0, my_label)
            outputs2 = model.forward((img2), valid_mask_lowres1, my_label)

            ####### update the bank
            # for i in range(ba):
            #     argl = (label_bank[i,:].squeeze()==1).nonzero(as_tuple =False).squeeze()
            #     valid_cat = torch.nonzero(label_bank[i])[:, 0].cpu().numpy()
            #     top_cam = F.interpolate(outputs1['cam1'][i][valid_cat].unsqueeze(0),
            #                             size=(h, w), mode='nearest').squeeze(0)
            #     for j in range(len(valid_cat)):
            #         print(j,valid_cat,top_cam.shape)
            #         slot[argl[j]] = torch.where(top_cam[j]>top_cam[j].mean(),img1[i],torch.tensor([0]).cuda()).cpu()
            #         vm[argl[j]] = valid_mask[i].cpu()
            #         sltarget[argl[j]] = label_bank[i].cpu()
            #######


            index += args.batch_size
            label1, cam1, cam_rv1, orignal_cam1 = outputs1['score'], outputs1['cam1'], outputs1['cam2'], outputs1['orignal_cam']
            loss_cls1 = F.multilabel_soft_margin_loss(label1, label[:,1:,:,:])
            cam_rv1 = cam_rv1 / (F.adaptive_max_pool2d(cam_rv1, (1, 1)) + 1e-5)
            
            lo = torch.abs(cam1[:,1:,:,:]-cam_rv1[:,1:,:,:])
            # cons = lo[:,1:,:,:]
            
            # ben = [66642, 13956, 13316, 12784, 11044,  8134,  8062,  7726,  7582,
            #     7536,  7536,  7626,  7698,  7704,  7792,  8032,  8308,  9036,
            #     10524, 11118]
            #
            # Images_num = [4155, 1228, 1188, 1150, 1005, 714, 705, 649, 613, 590,
            #         567, 522, 504, 503, 492, 468, 445, 393,
            #         300, 267]
            # Images_num = torch.Tensor(Images_num)
            # Images_num = Images_num+660*4*3/20
            # a = torch.Tensor(ben)
            # a = a /Images_num
            # a/=a.mean()

            # cons[:,14,:,:] = cons[:,14,:,:]*a[0] # 4.0486026 person
            # cons[:,8,:,:] = cons[:,8,:,:]*a[1] # 0.4921020 chair
            # cons[:,11,:,:] = cons[:,11,:,:]*a[2] # 0.443499 dog
            # cons[:,6,:,:] = cons[:,6,:,:]*a[3] # 0.28434 car
            # cons[:,7,:,:] = cons[:,7,:,:]*a[4] # 0.39732 cat
            # cons[:,4,:,:] = cons[:,4,:,:]*a[5] #  0.13244 bottle
            # cons[:,2,:,:] = cons[:,2,:,:]*a[6] # 0.143377 bird
            # cons[:,17,:,:] = cons[:,17,:,:]*a[7] # 0.211421 sofa
            # cons[:,10,:,:] = cons[:,10,:,:]*a[8] # 0.25516 diningtable
            # cons[:,0,:,:] = cons[:,0,:,:]*a[9] # 0.28311 aeroplane
            # cons[:,19,:,:] = cons[:,19,:,:]*a[10] # 0.311057 tvmonitor
            # cons[:,15,:,:] = cons[:,15,:,:]*a[11] # 0.365735 pottedplant
            # cons[:,1,:,:] = cons[:,1,:,:]*a[12] # 0.387606 bicycle
            # cons[:,18,:,:] = cons[:,18,:,:]*a[13] # 0.636182 train
            # cons[:,13,:,:] = cons[:,13,:,:]*a[14] # 0.672764 motorbike
            # cons[:,3,:,:] = cons[:,3,:,:]*a[15] # 0.388821 boat
            # cons[:,12,:,:] = cons[:,12,:,:]*a[16] # 0.402187 horse
            # cons[:,5,:,:] = cons[:,5,:,:]*a[17] # 0.431348  bus
            # cons[:,16,:,:] = cons[:,16,:,:]*a[18] # 1.743333 sheet
            # cons[:,9,:,:] = cons[:,9,:,:]*a[19] #  2.0823970 cow

            # lo[:,1:,:,:] = cons
            lossGSC = torch.mean(torch.abs(lo))*0.8

            # cam1 = F.interpolate(cam1,scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            label2, cam2, cam_rv2, orignal_cam2 = outputs2['score'], outputs2['cam1'], outputs2['cam2'], outputs2['orignal_cam']
            loss_cls2 = F.multilabel_soft_margin_loss(label2, label[:,1:,:,:])
            # print(gtimg1.view(-1,21,gtimg1.size(-1)*gtimg1.size(-2)).mean(-1).size())
            # print(gtimg2.size())
            # print(label2.size())
            # klloss = -kllossc(F.log_softmax(cam1.view(-1,21,gtimg1.size(-1)*gtimg1.size(-2)).mean(-1)),F.softmax(gtimg1.view(-1,21,gtimg1.size(-1)*gtimg1.size(-2)).mean(-1)).detach())
            # klloss += -kllossc(F.log_softmax(cam2.view(-1,21,gtimg2.size(-1)*gtimg2.size(-2)).mean(-1)),F.softmax(gtimg2.view(-1, 21, gtimg2.size(-1) * gtimg2.size(-2)).mean(-1)).detach())
            # outputs1.view(-1,21)

            cam2 = cam2*label
            lossCLS = (loss_cls1 + loss_cls2)/2
            cons = torch.abs(cam2[:,1:,:,:]-cam_rv2[:,1:,:,:])
            # cons[:,14,:,:] = cons[:,14,:,:]*a[0] # 4.0486026 person
            # cons[:,8,:,:] = cons[:,8,:,:]*a[1] # 0.4921020 chair
            # cons[:,11,:,:] = cons[:,11,:,:]*a[2] # 0.443499 dog
            # cons[:,6,:,:] = cons[:,6,:,:]*a[3] # 0.28434 car
            # cons[:,7,:,:] = cons[:,7,:,:]*a[4] # 0.39732 cat
            # cons[:,4,:,:] = cons[:,4,:,:]*a[5] #  0.13244 bottle
            # cons[:,2,:,:] = cons[:,2,:,:]*a[6] # 0.143377 bird
            # cons[:,17,:,:] = cons[:,17,:,:]*a[7] # 0.211421 sofa
            # cons[:,10,:,:] = cons[:,10,:,:]*a[8] # 0.25516 diningtable
            # cons[:,0,:,:] = cons[:,0,:,:]*a[9] # 0.28311 aeroplane
            # cons[:,19,:,:] = cons[:,19,:,:]*a[10] # 0.311057 tvmonitor
            # cons[:,15,:,:] = cons[:,15,:,:]*a[11] # 0.365735 pottedplant
            # cons[:,1,:,:] = cons[:,1,:,:]*a[12] # 0.387606 bicycle
            # cons[:,18,:,:] = cons[:,18,:,:]*a[13] # 0.636182 train
            # cons[:,13,:,:] = cons[:,13,:,:]*a[14] # 0.672764 motorbike
            # cons[:,3,:,:] = cons[:,3,:,:]*a[15] # 0.388821 boat
            # cons[:,12,:,:] = cons[:,12,:,:]*a[16] # 0.402187 horse
            # cons[:,5,:,:] = cons[:,5,:,:]*a[17] # 0.431348  bus
            # cons[:,16,:,:] = cons[:,16,:,:]*a[18] # 1.743333 sheet
            # cons[:,9,:,:] = cons[:,9,:,:]*a[19] #  2.0823970 cow

            loss_consistency = torch.mean(cons)*0.8

            cam1 = F.interpolate(cam1, scale_factor=scale_factor, mode='bilinear', align_corners=True) * label
            # losscon12 = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :])) * 0.8

            losses =  lossCLS*2 + (lossGSC + loss_consistency) *8


            avg_meter.add(
                {'lossCLS': lossCLS.item(), 'lossGSC': lossGSC.item(), 'loss_consistency': loss_consistency.item(),
                 })
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            if (optimizer.global_step - 1) % args.print_freq == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossGSC:%.4f' % (avg_meter.pop('lossGSC')),
                      'loss_consistency:%.4f' % (avg_meter.pop('loss_consistency')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)

                # tf record
                tblogger.add_scalar('lossCLS', lossCLS, optimizer.global_step)
                tblogger.add_scalar('lossGSC', lossGSC, optimizer.global_step)
                tblogger.add_scalar('loss_er', loss_consistency, optimizer.global_step)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], optimizer.global_step)
            
            if (optimizer.global_step - 1) % args.tf_freq == 0:
                # visualization
                img_1 = visualization.convert_to_tf(img1[0])
                norm_cam = F.interpolate(cam1,img_1.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                cam_rv1 = F.interpolate(cam_rv1,img_1.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                CAM1 = visualization.generate_vis(norm_cam, None, img_1, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)
                prototype_CAM1 = visualization.generate_vis(cam_rv1, None, img_1, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)

                gtimg1 = F.interpolate(gtimg1,img_1.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                gtimg1 = visualization.generate_vis(gtimg1, None, img_1, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)

                img_1 = visualization.convert_to_tf(img1[0])
                orignal_cam1 = F.interpolate(orignal_cam1, img_1.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                orignal_cam1 = visualization.generate_vis(orignal_cam1, None, img_1,
                                                  func_label2color=visualization.VOClabel2colormap, threshold=None,
                                                  norm=True)
                
                img_2 = visualization.convert_to_tf(img2[0])
                norm_cam2 = F.interpolate(cam2, img_2.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                cam_rv2 = F.interpolate(cam_rv2, img_2.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                CAM2 = visualization.generate_vis(norm_cam2, None, img_2, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)
                prototype_CAM2 = visualization.generate_vis(cam_rv2, None, img_2, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)
                
                tblogger.add_images('gt', gtimg1, optimizer.global_step)
                tblogger.add_images('CAM', CAM1, optimizer.global_step)
                tblogger.add_images('prototype_CAM1', prototype_CAM1, optimizer.global_step)
                tblogger.add_images('orignal_cam1', orignal_cam1, optimizer.global_step)
                tblogger.add_images('CAM2', CAM2, optimizer.global_step)
                tblogger.add_images('prototype_CAM2', prototype_CAM2, optimizer.global_step)

            if (optimizer.global_step-1) % args.val_freq == 0 and optimizer.global_step > 10:
                miou = validate(model, val_data_loader, optimizer.global_step, tblogger)
#                torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'iter_' + str(optimizer.global_step) + '.pth'))
                if miou > bestiou:
                    bestiou = miou
                    torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'best.pth'))
        else:
            timer.reset_stage()
    
    torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'final.pth'))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
    
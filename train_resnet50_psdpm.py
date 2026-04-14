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

RandomHorizontalFlip = T.RandomHorizontalFlip()
RandomSolarize = T.RandomSolarize(192)
ColorJitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
nclaug4_2 = T.RandomGrayscale(p=0.2)
GaussianBlur = T.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
augtopil = T.ToPILImage()
augtotensor = T.PILToTensor()
nclaug4 = Cutout(n_holes=5, length=16)

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


        return proc_img

class TorchvisionNormalize1():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = torch.permute(img,[0,3,1,2])

        img = torch.permute(img,[0,2,3,1])
        imgarr = (img).float()
        proc_img = torch.empty_like(imgarr)

        proc_img[:, :, :, 0] = (imgarr[:, :, :, 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[:, :, :, 1] = (imgarr[:, :, :, 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[:, :, :, 2] = (imgarr[:, :, :, 2] / 255. - self.mean[2]) / self.std[2]

        proc_img = torch.permute(proc_img,[0,3,1,2]).contiguous()

        return proc_img

class TorchvisionNormalizedeeplab1():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        proc_img = (img).float()
        proc_img = torch.permute(proc_img, [0, 3, 1, 2]).contiguous()

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
    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir="/data/VOCdevkit/VOC2012")
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]
    print('validating ... ', flush=True, end='')
    model.eval()
    with torch.no_grad():
        preds = []
        preds1 = []
        preds3 = []
        for iter, pack in enumerate(data_loader):       
            img = pack['img'].cuda().float()
            img = TorchvisionNormalizeval(img)
            label = pack['label'].cuda().unsqueeze(-1).unsqueeze(-1)
            label = F.pad(label, (0, 0, 0, 0, 1, 0), 'constant', 1.0)
            outputs = model.forward(img, label, pack['label'].cuda())
            IS_cam1 = outputs['affsum']
            IS_cam1 = F.interpolate(IS_cam1, img.shape[2:], mode='bilinear')
            IS_cam1 = IS_cam1/(F.adaptive_max_pool2d(IS_cam1, (1, 1)) + 1e-5)
            cls_labels_bkg1 = torch.argmax(IS_cam1, 1)
            preds1.append(cls_labels_bkg1[0].cpu().numpy().copy())

            IS_cam = outputs['cam2']
            IS_cam = F.interpolate(IS_cam, img.shape[2:], mode='bilinear')
            IS_cam = IS_cam/(F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            cls_labels_bkg = torch.argmax(IS_cam, 1)
            preds.append(cls_labels_bkg[0].cpu().numpy().copy())

            IS_cam = outputs['orignal_cam']
            IS_cam = F.interpolate(IS_cam, img.shape[2:], mode='bilinear')
            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            cls_labels_bkg = torch.argmax(IS_cam, 1)
            preds3.append(cls_labels_bkg[0].cpu().numpy().copy())




        confusion = calc_semantic_segmentation_confusion(preds, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou1 = gtjresj / denominator

        confusion = calc_semantic_segmentation_confusion(preds3, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou3 = gtjresj / denominator

        confusion = calc_semantic_segmentation_confusion(preds1, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou4 = gtjresj / denominator

        print('\n')
        print({'iou1': iou1, 'miou': np.nanmean(iou1)})
        print('\n')
        print({'iou3': iou3, 'miou': np.nanmean(iou3)})
        print('\n')
        print({'iou3': iou4, 'miou': np.nanmean(iou4)})
    model.train()

    return np.nanmean(iou3)

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
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=32, type=int)
    parser.add_argument("--network", default="network.resnet50_psdpm", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp4", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--tf_freq", default=20, type=int)
    parser.add_argument("--val_freq", default=300, type=int)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--dataset_root", default="/data/VOCdevkit/VOC2012", type=str)
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
        dataset_root = '/data/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=21)
        train_dataset = data_voc.VOC12ClsDataset('data/trainaug_' + args.dataset + '.txt', voc12_root=dataset_root,
                                                                    resize=(512, 512), hor_flip=False)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_voc.VOC12ClsDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    elif args.dataset == 'coco':
        args.tf_freq = 99999
        args.val_freq = 99999
        dataset_root = '/data/COCO'
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
    param_groups2 = model.trainable_parameters2()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()


    bestiou = 0
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()


    densecrf = DenseCRF(iter_max=10,
                        pos_xy_std=1,
                        pos_w=3,
                        bi_xy_std=67,
                        bi_rgb_std=3,
                        bi_w=4, )

    slot = torch.zeros([21, 3, 512, 512])
    slot_vm = torch.zeros([21,21, 32, 32])

    sltarget = torch.ones([21, 21])
    sltarget[:,0]=1
    gts1 = torch.zeros([21, 21, 32, 32])
    gts2 = torch.zeros([21, 21, 16, 16])


    for ep in range(args.max_epoches):
        
        print('Epoch %d/%d' % (ep + 1, args.max_epoches))
        index = 0
        for step, pack in enumerate(train_data_loader):


            scale_factor = 1
            img1 = pack['img']
            label = pack['label'].cuda()
            valid_mask = pack['valid_mask'].cuda()


            label = label
            img1gt = TorchvisionNormalizedeeplab(img1.float()).clone()
            img1 = img1.cuda()



            N, h, w,c = img1.shape
            my_label = label
            label = label.unsqueeze(2).unsqueeze(3)
            valid_mask[:,1:] = valid_mask[:,1:] * label

            valid_mask_lowres0 = F.interpolate(valid_mask, size=(h//16, w//16), mode='nearest')
            valid_mask_lowres1 = F.interpolate(valid_mask, size=(h//32, w//32), mode='nearest')

            bg_score = torch.ones((N, 1)).cuda()
            label = torch.cat((bg_score.unsqueeze(-1).unsqueeze(-1), label), dim=1)


            gtimg1 = []
            gtimg2 = []
            for j in range(len(pack['name'])):

                gtimg_new1 = torch.tensor(np.load('/gtimg1/' + pack['name'][j] + '.npy',
                                                  allow_pickle=True).item()['msc_seg'].copy()).squeeze()


                gtimg2_new = F.interpolate(gtimg_new1.clone().unsqueeze(0), size=(h // 32, w // 32), mode='bilinear').numpy()[0]

                gtimg1_new = F.interpolate(gtimg_new1.unsqueeze(0), size=(h // 16, w // 16), mode='bilinear').numpy()[0]

                gtimg1.append(gtimg1_new)
                gtimg2.append(gtimg2_new)


            gtimg1 = torch.tensor(np.array(gtimg1)).cuda()
            gtimg2 = torch.tensor(np.array(gtimg2)).cuda()

            img1 = TorchvisionNormalize(img1)


            outputs1 = model.forward((torch.cat([img1,slot.cuda()],0)), torch.cat([valid_mask_lowres0,slot_vm.cuda()],0), torch.cat([my_label,sltarget.cuda()[:,1:]],0), torch.cat([gtimg1,gts1.cuda()],0))
            outputs2 = model.forward((torch.cat([F.interpolate(img1.cuda(),scale_factor=0.5,mode='bilinear',align_corners=True),F.interpolate(slot.cuda(),scale_factor=0.5,mode='bilinear',align_corners=True)],0)), torch.cat([valid_mask_lowres1,F.interpolate(slot_vm.cuda(), size=(h // 32, w // 32), mode='bilinear')],0), torch.cat([my_label,sltarget.cuda()[:,1:]],0), torch.cat([gtimg2,gts2.cuda()],0))





            index += args.batch_size
            label1, cam1, cam_rv1, orignal_cam1,affsum1 = outputs1['score'], outputs1['cam1'], outputs1['cam2'], outputs1['orignal_cam'], outputs1['affsum']
            loss_cls1 = F.multilabel_soft_margin_loss(label1, torch.cat([label[:,1:,:,:],sltarget.cuda()[:,1:].unsqueeze(-1).unsqueeze(-1)],0))


            for i in range(len(label)):
                argl = (label[i, :].squeeze() == 1).nonzero(as_tuple=False).squeeze()
                ri = random.randint(0, len(argl) - 1)
                ri = argl[ri].cuda()

                imgsssss = img1[i]
                imgsssss[0, :, :][imgsssss[0, :, :] == 0] = -0.485 / 0.229
                imgsssss[1, :, :][imgsssss[1, :, :] == 0] = -0.456 / 0.224
                imgsssss[2, :, :][imgsssss[2, :, :] == 0] = -0.406 / 0.225
                slot[ri.cpu()] = (imgsssss).cpu()
                gts1[ri.cpu()] = gtimg1[i]
                gts2[ri.cpu()] = gtimg2[i]
                slot_vm[ri.cpu()] = valid_mask_lowres0[i]
                sltarget[ri.cpu(),1:] = my_label[i]

            lo = torch.abs(cam1[:, :, :, :] - cam_rv1[:, :, :, :])
            lo += torch.abs(orignal_cam1[:, :, :, :] - cam_rv1[:, :, :, :])*0.05

            lossGSC = torch.mean(torch.abs(lo))*2

            label2, cam2, cam_rv2, orignal_cam2,affsum2 = outputs2['score'], outputs2['cam1'], outputs2['cam2'], outputs2['orignal_cam'], outputs2['affsum']
            loss_cls2 = F.multilabel_soft_margin_loss(label2, torch.cat([label[:,1:,:,:],sltarget.cuda()[:,1:].unsqueeze(-1).unsqueeze(-1)],0))

            lossCLS = (loss_cls1 + loss_cls2)/2

            cons = torch.abs(cam2[:, :, :, :] - cam_rv2[:, :, :, :])
            cons += torch.abs(orignal_cam2[:, :, :, :] - cam_rv2[:, :, :, :])*0.05

            loss_consistency = (cons).mean()  * 0.4

            cam1 = F.interpolate(cam1, scale_factor=scale_factor, mode='bilinear', align_corners=True) * torch.cat([label,sltarget.cuda().unsqueeze(-1).unsqueeze(-1)],0)


            losses =  lossCLS + (lossGSC +loss_consistency)


            avg_meter.add(
                {'lossCLS': lossCLS.item(), 'lossGSC': lossGSC.item(), 'loss_consistency': loss_consistency.item()
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
                # tblogger.add_scalar('loss_er', loss_consistency, optimizer.global_step)
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

                affsum1 = F.interpolate(affsum1, img_1.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                affsum1 = visualization.generate_vis(affsum1, None, img_1,
                                                          func_label2color=visualization.VOClabel2colormap,
                                                          threshold=None,
                                                          norm=True)
                
                img_2 = visualization.convert_to_tf(img1[0])
                norm_cam2 = F.interpolate(cam2, img_2.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                cam_rv2 = F.interpolate(cam_rv2, img_2.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                CAM2 = visualization.generate_vis(norm_cam2, None, img_2, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)
                prototype_CAM2 = visualization.generate_vis(cam_rv2, None, img_2, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=True)
                
                tblogger.add_images('gt', gtimg1, optimizer.global_step)

                tblogger.add_images('slot', slot, optimizer.global_step)
                tblogger.add_images('CAM', CAM1, optimizer.global_step)
                tblogger.add_images('prototype_CAM1', prototype_CAM1, optimizer.global_step)
                tblogger.add_images('orignal_cam1', orignal_cam1, optimizer.global_step)
                tblogger.add_images('affsum1', affsum1, optimizer.global_step)
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
    
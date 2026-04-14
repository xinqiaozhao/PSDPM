import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
import torchvision
from libs.models import DeepLabV2_ResNet101_MSC

class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.f9 = torch.nn.Conv2d(2053, 2048, 1, bias=False)
        # self.f9_2 = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        self.f10 = torch.nn.Conv2d(517, 512, 1, bias=False)
        # self.f10_2 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.f11 = torch.nn.Conv2d(1029, 1024, 1, bias=False)
        # self.f11_2 = torch.nn.Conv2d(1024, 1024, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList(
            [self.classifier, self.f9, self.f10,self.f11, self.side1,self.side2, self.side3, self.side4])

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        seri1 = torch.range(1, w, 1).repeat(n, h, 1).cuda(f.device).view(n,1,h,w)
        seri2 = torch.range(1, h, 1).repeat(n, w, 1).permute(0,2,1).cuda(f.device).view(n,1,h,w)
        seri = F.normalize(torch.cat([seri1,seri2],dim=1),dim=1)
        f = torch.cat([seri, f], dim=1)
        # print(cam.shape)
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        # print(cam.shape)
        f = self.f9(f)

        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        # cam[:, 0] = cam[:, 0] * 0.5

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        # affeye = 1-torch.eye(h*w,device=aff.device).float()
        aff = aff  / (h*w)

        # aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)

        return aff, cam
        # cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        # cam_rv = torch.matmul(cam_rv.view(n,-1,h*w), aff).view(n,-1,h,w)
        # return cam_rv

    def PCM2(self, cam, f):
        n, c, h, w = f.size()
        seri1 = torch.range(1, w, 1).repeat(n, h, 1).cuda(f.device).view(n, 1, h, w)
        seri2 = torch.range(1, h, 1).repeat(n, w, 1).permute(0, 2, 1).cuda(f.device).view(n, 1, h, w)
        seri = F.normalize(torch.cat([seri1, seri2], dim=1), dim=1)
        f = torch.cat([seri, f], dim=1)

        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f10(f)

        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        # affeye = 1 - torch.eye(h * w, device=aff.device).float()
        aff = aff  / (h * w)
        # aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        # cam[:, 0] = cam[:, 0] * 0
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        cam_rv = torch.matmul(cam_rv.view(n,-1,h*w), aff).view(n,-1,h,w)+cam.view(n, -1, h, w)
        # cam_rv = torch.matmul(cam_rv.view(n, -1, h * w), aff).view(n, -1, h, w)
        return cam_rv

    def PCM3(self, cam, f):
        n, c, h, w = f.size()
        seri1 = torch.range(1, w, 1).repeat(n, h, 1).cuda(f.device).view(n, 1, h, w)
        seri2 = torch.range(1, h, 1).repeat(n, w, 1).permute(0, 2, 1).cuda(f.device).view(n, 1, h, w)
        seri = F.normalize(torch.cat([seri1, seri2], dim=1), dim=1)
        f = torch.cat([seri, f], dim=1)

        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f11(f)

        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        # aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        # affeye = 1 - torch.eye(h * w, device=aff.device).float()
        aff = aff  / (h * w)
        # cam[:, 0] = cam[:, 0]*0
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        cam_rv = torch.matmul(cam_rv.view(n,-1,h*w), aff).view(n,-1,h,w)+cam.view(n, -1, h, w)
        return cam_rv

    def prototype(self, norm_cam, feature, valid_mask):
        # n, c, h, w = norm_cam.shape
        # norm_cam[:, 0] = norm_cam[:, 0]*0.3
        # seeds = torch.zeros((n, h, w, c)).to(norm_cam.device)
        # belonging = norm_cam.argmax(1)
        # seeds = seeds.scatter_(-1, belonging.view(n, h, w, 1), 1).permute(0, 3, 1, 2).contiguous()
        seeds = norm_cam  # 4, 21, 32, 32

        n, c, h, w = feature.shape  # hie
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(
            1)  # .clone().detach()  # seed:[n,21,1,h,w], feature:[n,1,4c,h,w], crop_feature:[n,21,4c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1, c, h, w), (1, 1)).view(n, self.num_cls, c, 1,
                                                                                       1)  # prototypes:[n,21,c,1,1]

        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype,
                                                dim=2))  # feature:[n,1,4c,h,w], prototypes:[n,21,4c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam, prototype

    def forward(self, x, valid_mask, my_label=None, gt=None, epoch=None, index=None, train=None):
        # print(x.shape)
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        side2 = self.side2(x2)
        side3 = self.side3(x3)
        side4 = self.side4(x4)
        cam = self.classifier(x4)
        score1 = F.adaptive_avg_pool2d(cam, 1)
        # print(score1.shape)
        # x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)

        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        # print(x4.shape[2:])
        # print(norm_cam.shape)
        # print(valid_mask.shape)
        norm_cam = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask
        # norm_cam = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * label.unsqueeze(0).clone()

        orignal_cam = norm_cam

        hie_fea = torch.cat(
            [
                F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear')],
            dim=1)


        protocam,prototype = self.prototype((orignal_cam), hie_fea.clone(), valid_mask.clone())
        score = (score1 + 0.5*F.adaptive_avg_pool2d(protocam, 1)[:,1:,:,:])/1.5
        protocam = protocam*valid_mask
        if gt is not None:
            aff, norm_cam1 = self.PCM((protocam+4*gt)/4,
                                  torch.cat([F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4],
                                            dim=1))
        else:
            aff, norm_cam1 = self.PCM((protocam ) ,
                                      torch.cat(
                                          [F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4],
                                          dim=1))
        n, c, h, w = x4.size()
        norm_camx = norm_cam1.view(n,-1,h,w)

        norm_cam2 = torch.matmul(norm_cam1, aff)
        norm_cam2new = norm_cam2.view(n,-1,h,w)
        # print(norm_cam2.shape)
        norm_cam2 = torch.matmul(norm_cam2, aff).view(n, -1, h, w)+norm_cam1.view(n, -1, h, w)

        norm_cam3 = self.PCM3((F.interpolate(norm_cam2, x3.shape[2:], mode='bilinear', align_corners=True)),
                             torch.cat([F.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=True), x3],
                                       dim=1))
        norm_cam3 = F.interpolate(norm_cam3, orignal_cam.shape[2:], mode='bilinear', align_corners=True)


        norm_cam = self.PCM2((F.interpolate(norm_cam3, x2.shape[2:], mode='bilinear', align_corners=True)),
                             torch.cat([F.interpolate(x, x2.shape[2:], mode='bilinear', align_corners=True), x2],
                                       dim=1))
        norm_cam = F.interpolate(norm_cam, orignal_cam.shape[2:], mode='bilinear', align_corners=True)

        norm_cam += norm_cam2*3
        norm_cam += norm_cam3*2
        norm_cam /= 5

        if gt is not None:
            IS_cam = protocam
            # norm_cam += 2*gt
            # norm_cam /= 3
        else:
            IS_cam = protocam

        return {"score": score, "cam1": norm_cam, "cam2": IS_cam, "orignal_cam": orignal_cam, "feat": hie_fea, "protocam":protocam, "prototype":prototype, "cam_gt":norm_cam2new}
        # return {"score": score, "cam1": norm_cam, "cam2": IS_cam, "orignal_cam": orignal_cam}

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

        # self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).cuda()

        # deepv3 = DeepLabV2_ResNet101_MSC(n_classes=21)
        # state_dict = torch.load('./checkpoint_final.pth')
        # for m in deepv3.state_dict().keys():
        #     if m not in state_dict.keys():
        #         print("    Skip init:", m)
        # deepv3.load_state_dict(state_dict, strict=True)  # to skip ASPP
        # deepv3 = torch.nn.DataParallel(deepv3, device_ids=[0]).cuda(0)
        # deepv3.eval()
        # self.deeplabv3 = deepv3

    def forward(self, x, label):
        x0 = self.stage0(x)
        x1 = self.stage1(x0).detach()
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        side2 = self.side2(x2)
        side3 = self.side3(x3)
        side4 = self.side4(x4)
        cam = self.classifier(x4)

        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        x = (x[0] + x[1].flip(-1)).unsqueeze(0)
        x2 = (x2[0] + x2[1].flip(-1)).unsqueeze(0)
        x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)
        x3 = (x3[0] + x3[1].flip(-1)).unsqueeze(0)

        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * label.unsqueeze(0).clone()
        orignal_cam = norm_cam

        hie_fea = torch.cat(
            [
                F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear')],
            dim=1)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)

        protocam,prototype = self.prototype((orignal_cam), hie_fea.clone(), label.unsqueeze(0).clone())
        protocam = protocam*label.unsqueeze(0)

        aff, norm_cam1 = self.PCM(protocam.detach(),
                                  torch.cat([F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4],
                                            dim=1))
        n, c, h, w = x4.size()
        norm_camx = norm_cam1.view(n,-1,h,w)
        norm_cam2 = torch.matmul(norm_cam1, aff).view(n, -1, h, w) + norm_cam1.view(n, -1, h, w)

        norm_cam3 = self.PCM3((F.interpolate(norm_cam2, x3.shape[2:], mode='bilinear', align_corners=True)),
                              torch.cat([F.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=True), x3],
                                        dim=1))
        norm_cam3 = F.interpolate(norm_cam3, orignal_cam.shape[2:], mode='bilinear', align_corners=True)

        norm_cam = self.PCM2((F.interpolate(norm_cam3, x2.shape[2:], mode='bilinear', align_corners=True)),
                             torch.cat([F.interpolate(x, x2.shape[2:], mode='bilinear', align_corners=True), x2],
                                       dim=1))
        norm_cam = F.interpolate(norm_cam, orignal_cam.shape[2:], mode='bilinear', align_corners=True)

        norm_cam += norm_cam2*3
        norm_cam += norm_cam3*2
        norm_cam /= 5


        return protocam[0], norm_cam[0], orignal_cam[0], norm_camx[0],norm_cam2[0],norm_cam3[0]

        # gt = self.deeplabv3(x)
        # return gt['out'][0], gt['out'][0], gt['out'][0]

        # gt = self.deeplabv3(x)
        # return gt[0], gt[0], gt[0]
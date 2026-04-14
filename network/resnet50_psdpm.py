import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
import torchvision
from libs.models import DeepLabV2_ResNet101_MSC

class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(self, input,input2,input3):
        batch_size, seq_length, embed_dim = input.size()
        query = self.query(input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = (input).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        return attn_output, attn_weights.mean(1)

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

        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)
        self.classifier2 = nn.Conv2d(1024, self.num_cls - 1, 1, bias=False)
        self.classifier3 = nn.Conv2d(512, self.num_cls - 1, 1, bias=False)

        self.f9 = torch.nn.Conv2d(2051, 2048, 1, bias=False)
        self.f9_2 = torch.nn.Conv2d(2048, 2048, 1, bias=False)
        self.f10 = torch.nn.Conv2d(1027, 1024, 1, bias=False)
        self.f10_2 = torch.nn.Conv2d(1024, 1024, 1, bias=False)
        self.f11 = torch.nn.Conv2d(515, 512, 1, bias=False)
        self.f11_2 = torch.nn.Conv2d(512, 512, 1, bias=False)


        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList(
            [self.classifier,self.classifier2,self.classifier3,self.f9,self.f9_2,self.f10,self.f10_2,self.f11,self.f11_2,self.side2,self.side3,self.side4])
        self.transformer =  nn.ModuleList([self.f9,self.f9_2,self.f10,self.f10_2,self.f11,self.f11_2])

    def PCM(self, cam, rgb,f):
        rawf = f.clone()
        n, c, h, w = f.size()

        f = torch.cat([rgb,f], dim=1)
        f = self.f9(f)
        f= F.relu(f)
        f = self.f9_2(f).view(-1,h*w,2048)
        f = F.relu(f)

        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)


        f = f.view(n, -1, h * w)

        aff = F.softmax(torch.matmul(f.transpose(1, 2), f), dim=-1)

        attn_output = torch.matmul(aff, rawf.view(n, -1, h * w).transpose(1, 2))

        return aff, cam,attn_output


    def PCM2(self, cam, rgb,f):
        rawf = f.clone()
        n, c, h, w = f.size()

        f = torch.cat([ rgb, f], dim=1)

        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f10(f)
        f = F.relu(f)
        f = self.f10_2(f).view(-1, h * w,1024)
        f = F.relu(f)



        f = f.view(n, -1, h * w)

        aff = F.softmax(torch.matmul(f.transpose(1, 2), f), dim=-1)

        attn_output = torch.matmul(aff, rawf.view(n, -1, h * w).transpose(1, 2))

        return aff, cam,attn_output


    def PCM3(self, cam, rgb,f):
        rawf = f.clone()
        n, c, h, w = f.size()

        f = torch.cat([rgb, f], dim=1)

        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f11(f)
        f = F.relu(f)
        f = self.f11_2(f).view(-1, h * w, 512)
        f = F.relu(f)


        f = f.view(n, -1, h * w)

        aff = F.softmax(torch.matmul(f.transpose(1, 2), f),dim=-1)

        attn_output = torch.matmul(aff, rawf.view(n, -1, h * w).transpose(1, 2))

        return aff, cam,attn_output

    def prototype(self, norm_cam, feature,feature2, valid_mask):

        seeds = norm_cam

        n, c, h, w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(
            1)
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1, c, h, w), (1, 1)).view(n, self.num_cls, c, 1,
                                                                                       1)

        IS_cam = F.relu(torch.cosine_similarity(feature2.unsqueeze(1), prototype,
                                                dim=2))
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def get_cos_sin(self, x, y):
        normx = torch.norm(x, 2, 1, keepdim=True)
        normx[normx == 0] = 1
        normy = torch.norm(y, 2, 1, keepdim=True)
        normy[normy == 0] = 1
        cos_val = (x * y).sum(-1, keepdim=True) / normx / normy
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def forward(self, x, valid_mask, my_label=None, gt=None, slot=None, epoch=None, index=None, train=None):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)

        norm_cam = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask
        orignal_cam = norm_cam
        cam = self.classifier2(x3)
        score += F.adaptive_avg_pool2d(cam, 1)
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)

        orignal_cam2 = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask
        orignal_cam = (orignal_cam+orignal_cam2*0.5)
        cam = self.classifier3(x2)
        score += F.adaptive_avg_pool2d(cam, 1)
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        orignal_cam3 = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask
        orignal_cam = (orignal_cam + orignal_cam3*0.25 ) / 1.75


        aff, norm_cam1,attn_output = self.PCM((orignal_cam ) ,
                                  F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4)
        n, c, h, w = x4.size()
        affsum = aff.sum()
        att1= (attn_output.transpose(1,2).reshape(n,c,h,w)+x4*5)
        aff_v = self.classifier(att1)
        score += F.adaptive_avg_pool2d(aff_v, 1)
        aff_v = F.relu(aff_v)
        aff_v = aff_v / (F.adaptive_max_pool2d(aff_v, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(aff_v, dim=1)[0].unsqueeze(1)
        aff_v = torch.cat([cam_bkg, aff_v], dim=1)
        aff_v = F.interpolate(aff_v, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask

        aff2, norm_cam12, attn_output2 = self.PCM2((orignal_cam2),
                                               F.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=True), x3)
        n, c, h, w = x3.size()
        affsum2 = aff2.sum()
        att2=(attn_output2.transpose(1,2).reshape(n,c,h,w)+x3*5)
        aff_v2 = self.classifier2(att2)
        score += F.adaptive_avg_pool2d(aff_v2, 1)
        aff_v2 = F.relu(aff_v2)
        aff_v2 = aff_v2 / (F.adaptive_max_pool2d(aff_v2, (1, 1)) + 1e-5)
        cam_bkg2 = 1 - torch.max(aff_v2, dim=1)[0].unsqueeze(1)
        aff_v2 = torch.cat([cam_bkg2, aff_v2], dim=1)
        aff_v2 = F.interpolate(aff_v2, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask

        aff3, norm_cam13, attn_output3 = self.PCM3((orignal_cam3),
                                                   F.interpolate(x, x2.shape[2:], mode='bilinear', align_corners=True),
                                                   x2)
        n, c, h, w = x2.size()
        affsum3 = aff3.sum()
        att3=(attn_output3.transpose(1,2).reshape(n,c,h,w)+x2*5)
        aff_v3 = self.classifier3(att3)
        score += F.adaptive_avg_pool2d(aff_v3, 1)
        aff_v3 = F.relu(aff_v3)
        aff_v3 = aff_v3 / (F.adaptive_max_pool2d(aff_v3, (1, 1)) + 1e-5)
        cam_bkg3 = 1 - torch.max(aff_v3, dim=1)[0].unsqueeze(1)
        aff_v3 = torch.cat([cam_bkg3, aff_v3], dim=1)
        aff_v3 = F.interpolate(aff_v3, x4.shape[2:], mode='bilinear', align_corners=True) * valid_mask

        cam22 = (aff_v+aff_v2*0.5+aff_v3*0.25)
        cam22 = cam22 / 1.75

        side2 = self.side2(att3)
        side3 = self.side3(att2)
        side4 = self.side4(att1)
        hie_fea = torch.cat(
            [
                F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear')],
            dim=1)

        side2 = self.side2(x2)
        side3 = self.side3(x3)
        side4 = self.side4(x4)
        hie_fea2 = torch.cat(
            [
                F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear')],
            dim=1)

        pcam = self.prototype((cam22), hie_fea.clone(), hie_fea2.clone(), valid_mask.clone())
        pcam = pcam * valid_mask

        if gt is not None:
            gt = gt*valid_mask



        return {"score": score, "cam1": gt, "cam2": cam22, "orignal_cam": pcam,'affsum':orignal_cam}


    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
    def trainable_parameters2(self):
        return (list(self.transformer.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls



    def forward(self, x, label):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        side2 = self.side2(x2)
        side3 = self.side3(x3)
        side4 = self.side4(x4)
        hie_fea = torch.cat(
            [
                F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear'),
                F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:],
                              mode='bilinear')],
            dim=1)

        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)

        norm_cam = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * label.unsqueeze(0)
        orignal_cam = norm_cam
        cam = self.classifier2(x3)
        score += F.adaptive_avg_pool2d(cam, 1)
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)

        orignal_cam2 = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * label.unsqueeze(0)
        orignal_cam = (orignal_cam + orignal_cam2 * 0.5)
        cam = self.classifier3(x2)
        score += F.adaptive_avg_pool2d(cam, 1)
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        orignal_cam3 = F.interpolate(norm_cam, x4.shape[2:], mode='bilinear', align_corners=True) * label.unsqueeze(0)
        orignal_cam = (orignal_cam + orignal_cam3 * 0.25) / 1.75




        return orignal_cam[0], orignal_cam[0], orignal_cam[0]


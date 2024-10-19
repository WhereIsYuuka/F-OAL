
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.scale = 0.09

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized, weight_normalized.transpose(0, 1))
        scores = cos_dist / self.scale
        return scores


class ModifiedViT(nn.Module):
    def __init__(self, n_classes):
        super(ModifiedViT, self).__init__()
        self.numclass=n_classes
        self.vit = timm.create_model("vit_base_patch16_224",pretrained=True)
        # self.cnn = timm.create_model('resnet34', pretrained=True)
        # self.vit = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.head = nn.Identity()
        # self.vit.heads = nn.Identity()
        self.fc = nn.Linear(768,n_classes)
        self.pcrLinear = cosLinear(768, n_classes)

    def features(self, x):
        x = self.vit(x)
        return x

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x

    def pcrForward(self, x):
        out = self.features(x)
        logits = self.pcrLinear(out)
        return logits, out

class ModifiedViTDVC(nn.Module):
    def __init__(self, n_classes):
        super(ModifiedViTDVC, self).__init__()
        self.numclass=n_classes
        self.vit = timm.create_model("vit_base_patch16_224",pretrained=True)
        # self.vit = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.head =nn.Identity()
        # self.vit.heads = nn.Identity()
        self.fc = nn.Linear(768,n_classes)
        self.pcrLinear = cosLinear(768, n_classes)

    def features(self, x):
        x = self.vit(x)
        return x

    def forward(self, x):
        out = self.vit(x)
        logits = self.fc(out)
        return logits,out

class QNet(nn.Module):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )

    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt


class DVCNet(nn.Module):
    def __init__(self,
                 backbone,
                 n_units,
                 n_classes,
                 has_mi_qnet=True):
        super(DVCNet, self).__init__()

        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz,fea = self.backbone(xx)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]


def VIT_DVC(nclasses):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    backnone = ModifiedViTDVC(nclasses)
    return DVCNet(backbone=backnone,n_units=128,n_classes=nclasses,has_mi_qnet=True)



''' This is encoder for FOAL'''
class Encoder(nn.Module):
    def __init__(self, n_classes,projection_size):
        super(Encoder, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.numclass = n_classes
        self.vit.head=nn.Identity()
        self.linear_projection=nn.Linear(768,projection_size,bias=False)
        self.fc=nn.Linear(projection_size,n_classes,bias=False)
        self.sig=nn.Sigmoid()


    def expansion(self,x):

        B = x.shape[0]  # Batch size
        x = self.vit.patch_embed(x)

        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        cumulative_output = 0
        concatenated_output = []

        for blk in self.vit.blocks:
            x = blk(x)
            cumulative_output += x[:, 0]
            concatenated_output.append(x[:, 0])
        x = cumulative_output / len(self.vit.blocks)
        x = self.vit.norm(x)
        x = self.linear_projection(x)
        x = self.sig(x)
        return x



    def forward(self, x):
        fusion_feature = self.expansion(x)
        x = self.fc(fusion_feature)
        return x






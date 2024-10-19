import timm
import torch
import torch.nn as nn


class CustomViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=1000):
        super(CustomViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        # 需要保证头部能够处理累加后的特征，所以使用原头部是可行的
        self.head = self.vit.head

    def forward(self, x):
        B = x.shape[0]  # Batch size
        x = self.vit.patch_embed(x)  # 应用 patch embedding

        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # 扩展CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # 将CLS token添加到序列的前面
        x = x + self.vit.pos_embed  # 添加位置嵌入
        x = self.vit.pos_drop(x)  # 应用位置dropout

        # 初始化一个变量来累加每个块的输出
        cumulative_output = 0

        for blk in self.vit.blocks:
            x = blk(x)
            cumulative_output += x[:, 0]  # 累加每个块的CLS token输出

        # 计算平均值
        average_output = cumulative_output / len(self.vit.blocks)

        # 应用最后的层次归一化
        x = self.vit.norm(average_output)

        # 通过分类头部


        return x

model = CustomViT(pretrained=True)
input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个假的输入张量
output = model(input_tensor)
print(output.shape)  # 检查输出的形状

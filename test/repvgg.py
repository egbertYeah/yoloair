import torch.nn as nn
import numpy as np
import torch
import copy
from se_block import SEBlock
 
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """带BN的卷积层"""
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
 
class RepVGGBlock(nn.Module):
 
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy  # 推理部署
        self.groups = groups
        self.in_channels = in_channels
 
        assert kernel_size == 3
        assert padding == 1
 
        padding_11 = padding - kernel_size // 2  # //为对商进行向下取整；padding_11应该是1×1卷积的padding数
 
        self.nonlinearity = nn.ReLU()
 
        if use_se:  # 是否将identity分支换成SE
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
 
        if deploy:  # 定义推理模型时，基本block就是一个简单的 conv2d
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
 
        else:  # 训练时
            # identity分支，仅有一个BN层；当输入输出channel不同时，不采用identity分支，即只有3×3卷积分支和1×1卷积分支
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            # 3×3卷积层+BN
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # 1×1卷积+BN，这里padding_11应该为0
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)
 
 
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):  # 如果在推理阶段，会定义self.rbr_reparam这个属性，这里是判断是否在推理阶段
            # 注意：执行return语句后就会退出函数，之后的语句不再执行
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))  # self.se()要看use_se是否为True，决定是否用SE模块
 
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # 训练阶段：3×3卷积、1×1卷积、identity三个分支的结果相加后进行ReLU
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
 
 
    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
 
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle
 
 
 
#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        """获取带BN的3×3卷积、带BN的1×1卷积和带BN的identity分支的等效卷积核、偏置，论文Fig4的第二个箭头"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # 返回三个等效卷积层的叠加，即三个卷积核相加、三个偏置相加
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
 
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """                     [0  0  0]
            [1]  >>>padding>>>  [0  1  0]
                                [0  0  0] """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])
 
    def _fuse_bn_tensor(self, branch):
        """融合BN层，既可以实现卷积层和BN层融合成等效的3×3卷积层，也可以实现单独BN层等效成3×3卷积层，论文Fig4的第一个箭头"""
        if branch is None:  # 若branch不是3x3、1x1、BN，那就返回 W=0, b=0
            return 0, 0
        if isinstance(branch, nn.Sequential):  # 若branch是nn.Sequential()类型，即一个卷积层+BN层
            kernel = branch.conv.weight  # 卷积核权重
            running_mean = branch.bn.running_mean  # BN的均值
            running_var = branch.bn.running_var  # BN的方差
            gamma = branch.bn.weight  # BN的伽马
            beta = branch.bn.bias  # BN的偏置
            eps = branch.bn.eps  # BN的小参数，为了防止分母为0
 
        else:  # 如果传进来的仅仅是一个BN层
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups  # self.group如果不等于1，即为分组卷积
                # 由于BN层不改变特征层维度，所以这里在自定义kernel时：batch=in_channels，那么输出特征层的深度依然是in_channels
                # np.zeros((self.in_channels, input_dim, 3, 3)即为in_channels个(input_dim, 3, 3)大小的卷积核
                # 每个卷积核生成一个channel的feature map，故生成输出feature map的通道数依然是in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)  # [batch,C,H,W]
                for i in range(self.in_channels):
                    # 这个地方口头表达不太好说，总之就是将BN等效成3×3卷积核的过程，可以去看一下论文理解
                    # 值得注意的是，i是从0算起的
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)  # 转成tensor并指定给device
            # 获取参数
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()  # 标准差
        t = (gamma / std).reshape(-1, 1, 1, 1)  # reshape成与[batch,1,1,1]方便在kernel * t时进行广播
        return kernel * t, beta - running_mean * gamma / std  # 返回加权后的kernel和偏置
 
    def switch_to_deploy(self):
        """转换成推理所需的VGG结构"""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
 
 
 
class RepVGG(nn.Module):
 
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None,
                 deploy=False, use_se=False):
        super(RepVGG, self).__init__()
 
        # len(width_multiplier) == 4对应论文3.4节第三段第一句
        assert len(width_multiplier) == 4  # 宽度因子，改变输入/输出通道数
 
 
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()  # 用于分组卷积的字典，在L215处有定义
        self.use_se = use_se
 
        assert 0 not in self.override_groups_map  # 确保0不会成为分组数
 
        self.in_planes = min(64, int(64 * width_multiplier[0]))
 
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1  # 当前layer的索引，用于对特定layer设置分组卷积
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
 
 
    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # 论文中提到：每个stage仅在第一个layer进行步长为2
        blocks = []  # 创建空列表
        for stride in strides:
            # 获取override_groups_map中键self.cur_layer_idx对应的值，若未设置值，则返回默认值1
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)  # 分组数
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes  # RepVGG中实例属性self.in_planes随传入planes而改变
            self.cur_layer_idx += 1  # 每次循环即创建一个layer，那么相应的layer的索引就要+1
        return nn.Sequential(*blocks)
 
    def forward(self, x):  # 前向传播
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
 
# 特定layer的索引分组卷积
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # 论文3.4节第四段
g2_map = {l: 2 for l in optional_groupwise_layers}  # {2: 2, 4: 2, 6: 2,...,20: 2, 22: 2, 24: 2, 26: 2}
g4_map = {l: 4 for l in optional_groupwise_layers}
 
def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)
 
 
def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)
 
 
def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)
 
def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)
 
def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)
 
def create_RepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)
 
 
func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
'RepVGG-D2se': create_RepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
 
 
def get_RepVGG_func_by_name(name):  # 传入所需构建的model名（key），返回构建model的函数
    return func_dict[name]
 
 
 
#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True
 
#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example
 
def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    """模型转换"""
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
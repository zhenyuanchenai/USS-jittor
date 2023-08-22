# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import jittor as jt
import jittor.nn as nn
from src.decoder import UPerHead

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = jt.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = jt.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Mulpixelattn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_mlp = channels
        self.atten = nn.Sequential(
            nn.Conv2d(channels, hidden_mlp, 1),
            nn.BatchNorm2d(hidden_mlp),
            nn.ReLU(),
            nn.Conv2d(hidden_mlp, channels, 1),
            nn.BatchNorm2d(channels, affine=True),
        )
        self.threshold = jt.zeros((1, channels, 1, 1))  #没有梯度
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):
        x = self.atten(x)
        x = x + self.threshold
        att = jt.sigmoid(x)   #64,
        return att


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 widen=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 normalize=False,
                 output_dim=0,
                 hidden_mlp=0,
                 nmb_prototypes=0,
                 eval_mode=False,
                 train_mode='finetune',
                 num_classes=919,
                 shallow=None):
        super().__init__()
        assert train_mode in ['pretrain', 'pixelattn', 'finetune'], train_mode
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.num_classes = num_classes
        self.shallow = shallow
        if isinstance(self.shallow, int):
            self.shallow = [self.shallow]

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code
        # because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(3,
                               num_out_filters,
                               kernel_size=7,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block,
                                       num_out_filters,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block,
                                       num_out_filters,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block,
                                       num_out_filters,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion,
                                             output_dim)                                   #就是为了得到最后的pixel的特征，所以做一个映射。
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = nn.ModuleList()
                if self.shallow is not None:
                    for stage in shallow:
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Linear(num_out_filters * block.expansion,
                                      output_dim))

        else:  #pretrain走这个，output_dim=128,hidden_mlp=512,走三遍三个head：projection_head_shallow， 最后一层投影，外加中间三层的投影
               #pixel-attention也走这个，只是最后一层投影。
            mlps = [
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Linear(hidden_mlp, output_dim),
                nn.BatchNorm1d(output_dim, affine=False)
            ]
            self.projection_head = nn.Sequential(*mlps)
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = nn.ModuleList()
                if self.shallow is not None:
                    for stage in shallow: #3
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Sequential(
                                nn.Linear(
                                    num_out_filters * block.expansion // (2 * (4 - stage)),
                                    hidden_mlp), nn.BatchNorm1d(hidden_mlp),
                                nn.ReLU(),
                                nn.Linear(hidden_mlp, output_dim),
                                nn.BatchNorm1d(output_dim, affine=False)))
        if self.train_mode == 'pretrain': #projection_head_pixel_shallow（每一次两个linear，最后映射到64维度）执行三次。
            self.projection_head_pixel_shallow = nn.ModuleList()
            if self.shallow is not None:
                for stage in shallow:
                    assert stage < 4
                    self.projection_head_pixel_shallow.add_module(
                        f'projection_head_pixel{stage}',
                        nn.Sequential(
                            nn.Conv2d(
                                num_out_filters * block.expansion // (2 * (4 - stage)),
                                hidden_mlp,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(hidden_mlp,
                                      hidden_mlp,
                                      kernel_size=1,
                                      bias=False),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(
                                hidden_mlp,
                                num_out_filters * block.expansion // (2 * (4 - stage)),
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_out_filters * block.expansion //
                                           (2 * (4 - stage)),
                                           affine=False),
                        ))

            # projection for pixel-to-pixel 该通道1*1卷积，走三遍。最后输出跟project head shallow一样128
            self.projection_head_pixel = nn.Sequential(
                nn.Conv2d(num_out_filters * block.expansion,
                          hidden_mlp,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, hidden_mlp, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim, affine=False),
            )
            self.predictor_head_pixel = nn.Sequential(  #最后还是128
                nn.Conv2d(output_dim, output_dim, 1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, 1),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)  #原型就是一个线形层。500,  50

        if train_mode == 'pixelattn':
            self.fbg = Mulpixelattn(num_out_filters * block.expansion)  #0-1的mask（过两层线形层），这个mask是128维度的
        elif train_mode == 'finetune':
            self.last_layer = UPerHead(
                    #in_channels=[256, 512, 1024, 2048],
                    in_channels=[64, 128, 256, 512],
                    in_index=[0, 1, 2, 3],
                    pool_scales=(1, 2, 3, 6),
                    channels=512,
                    dropout_ratio=0.1,
                    num_classes=51,
                    norm_cfg = dict(type='BN'),
                    align_corners=False,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)
 
    def execute_backbone(self, x, avgpool=True):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        c1 = x
        x = self.layer2(x)
        c2 = x
        x = self.layer3(x)
        c3 = x
        x = self.layer4(x)

        if self.train_mode == 'finetune':
            return [c1,c2,c3,x]
        
        if self.eval_mode or self.train_mode != 'pretrain':
            return x

        if avgpool:
            x = self.avgpool(x)
            x = jt.flatten(x, 1)
            return x

        return x, c3, c2, c1  #pretrain时候，把最后的特征和每一层的特征拿出来。

    def execute_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def execute_head_shallow(self, x, stage):
        if (self.projection_head_shallow is not None
                and f'projection_head_shallow{stage}'
                in self.projection_head_shallow.keys()):
            x = self.projection_head_shallow.layers[
                f'projection_head_shallow{stage}'](x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, nn.matmul_transpose(x, self.prototypes.weight.detach()) 
        return x

    def execute_head_pixel(self, x, gridq, gridk):
        if self.projection_head_pixel is not None:
            x = self.projection_head_pixel(x)

        # grid sample 28 x 28
        grid = jt.concat([gridq, gridk], dim=0)
        x = nn.grid_sample(x, grid, align_corners=False, mode='bilinear')

        return x, self.predictor_head_pixel(x)

    def execute(self, inputs, gridq=None, gridk=None, mode='train'):
        if mode == 'cluster':
            output = self.inference_cluster(inputs)    #对输入进行聚类
            return output
        elif mode == 'inference_pixel_attention':
            return self.inference_pixel_attention(inputs)

        if self.train_mode == 'finetune':
            out = self.execute_backbone(inputs) #返回每个像素的特征
            return self.last_layer(out)         #最后投影到类别

        if not isinstance(inputs, list):  #跟dino一样，输入变成一个list，也就是将他们拼接在一起。
            inputs = [inputs]
        idx_crops, last_size = [0], inputs[0].shape[-1]
        for sample in [inp.shape[-1] for inp in inputs]:
            if sample == last_size:
                idx_crops[-1] += 1
            else:
                idx_crops.append(idx_crops[-1] + 1)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.execute_backbone(jt.concat(    #pretrain时候返回中间层，pixel-attention时候只返回最后一层。
                inputs[start_idx:end_idx]), avgpool=self.train_mode != 'pretrain')
            if start_idx == 0:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)        #最后一层，前景特征的avg-pool的global结果
                elif self.train_mode == 'pretrain':
                    _out, _c3, _c2, _c1 = _out    
                    (
                        embedding_deep_pixel,  #local的特征
                        output_deep_pixel,     #local的投影head
                    ) = self.execute_head_pixel(_out, gridq, gridk)  #把local的特征和投影拿出来

                    _stages = [_c1, _c2, _c3]
                    if self.shallow is not None:
                        output_c = []
                        for i, stage in enumerate(self.shallow):
                            _c = _stages[stage - 1]
                            _out_c = self.projection_head_pixel_shallow.layers[    #每一个中间层的特征拿出来，然后接上一个投影头
                                f'projection_head_pixel{stage}'](_c)
                     
                            _out_c = self.avgpool(_out_c)                          #对结果取平均，最后只有一个image的feature
                            _out_c = jt.flatten(_out_c, 1)
                            output_c.append(_out_c)                                #作为output C （统一为64维）
                    _out = self.avgpool(_out)                                      #再将最后一层取平均作为output=_out (图像的最后一层维度)
                    _out = jt.flatten(_out, 1)
                output = _out

            else:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)                      #local部分的前景特征的global结果。
                elif self.train_mode == 'pretrain':
                    _out, _, _, _ = _out                                           #local特征的中间层
                    _out = self.avgpool(_out)                                      #最后一层拿出来取平均，得到global的结果。
                    _out = jt.flatten(_out, 1)

                output = jt.concat((output, _out))
            start_idx = end_idx

        embedding, output = self.execute_head(output)                              #对global的特征先做非线性处理，再做norm，最后接一个protype聚类到500，输出聚类前的特征和聚类的结果。
                                                                       #pre-train和pixel attention都要过聚类。
        if self.shallow is not None:                                               #对浅层的global特征也做非线性处理，再做聚类，输出聚类前的特征和聚类结果。
            for i, stage in enumerate(self.shallow):
                embedding_c_, output_c_ = self.execute_head_shallow(output_c[i],
                                                                  stage=stage)

                embedding = jt.concat((embedding, embedding_c_))
                output = jt.concat((output, output_c_))
        if self.train_mode == 'pixelattn':
            return embedding, output                                               #输出聚类的全局特征，和聚类结果。
        elif self.train_mode == 'pretrain':

            return embedding, output, embedding_deep_pixel, output_deep_pixel      #输出所有图像的全局特征（global图像和local图像），所有聚类结果，global图像的local特征，global图像的local的投影特征
        return embedding, output

    def execute_pixel_attention(self, out, threshold=0.):  #执行pixel-attention的地方
        out = nn.interpolate(                                #首先插值
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4), 
            mode='bilinear')
        out = jt.normalize(out, dim=1, p=2)                  #归一化
        fg = self.fbg(out)                                   #前背景二值化
        if threshold is not None:                            #没有前背景的阈值
            fg[fg < threshold] = 0

        out = out * fg                                       #得到前景的特征
        out = self.avgpool(out)
        out = jt.flatten(out, 1)                             #前景的特征avg pool一下，输出

        return out

    def inference_cluster(self, x, threshold=0.):
        out = self.execute_backbone(x)       #对resnet的输出不加任何的投影，直接输出每一个pixel的embedding
        out = nn.interpolate(
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4),            #插值四倍
            mode='bilinear')
        nout = jt.normalize(out, dim=1, p=2)               #归一化
        fg = self.fbg(nout)                                #预测前背景
        if threshold is not None:                          #还是没有阈值，外面没传进来。
            fg[fg < threshold] = 0

        out = out * fg                                     
        out = self.avgpool(out)
        out = jt.flatten(out, 1)                           #输出前景的global结果

        return out

    def inference_pixel_attention(self, x):
        out = self.execute_backbone(x)            #提取出最后的pixel特征

        out = nn.interpolate(
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4),   #对pixel特征做插值
            mode='bilinear') 
        out_ = jt.normalize(out, dim=1, p=2)          
        fg = self.fbg(out_)                       #对pixel特征归一化的结果生成前景
        fg = fg.mean(dim=1, keepdims=True)        #前景对所有128维度的通道做一个mean，得到一个1维的H，W的mask

        return out, fg


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            setattr(self, 'prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def execute(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

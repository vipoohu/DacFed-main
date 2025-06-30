import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import torch
from torch.autograd import Function
from torchvision.models import mobilenet_v2

class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, out_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 调整第一层卷积层的步幅和填充
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

class LeNet(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class TwoConvOneFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoConvOneFc, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class medcnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(medcnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels[0], 16, kernel_size=3),  # 确保 in_channels 是整数
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),  # 假设输入尺寸为 (3, 28, 28)
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class cifar(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(cifar, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels[0], 16, kernel_size=3),  # in_channels 是整数
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),  # 输入尺寸改为 1600
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SFA(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(SFA, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_dim)

        self.step = 1
        self.dim = 512
        self.map_scale = nn.Parameter(torch.ones(self.dim))
        self.map_bias = nn.Parameter(torch.zeros(self.dim))



    def forward(self, x, mask=torch.rand(1024)):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        index = torch.argsort(mask).detach()

        index_global = torch.tile(torch.unsqueeze(torch.sort(index[:self.dim]).values, dim=0), (out.shape[0], 1))
        index_local = torch.tile(torch.unsqueeze(torch.sort(index[self.dim:]).values, dim=0), (out.shape[0], 1))

        if torch.cuda.is_available():
            index_global, index_local = index_global.cuda(), index_local.cuda()

        x_global = torch.gather(out, 1, index_global)
        x_local = torch.gather(out, 1, index_local)

        x_global_map = self.map_scale * x_global + self.map_bias
        x_global_dis = self.map_scale * x_global.detach() + self.map_bias

        t = torch.zeros_like(out)
        t = t.scatter(1, index_global, x_global_map)
        out = t.scatter(1, index_local, x_local)

        t_dis = torch.zeros_like(out)
        t_dis = t_dis.scatter(1, index_global, x_global_dis)
        out_dis = t_dis.scatter(1, index_local, x_local.detach())

        self.step += 1
        alpha = min(self.step / 1e5,1e-2)

        reverse_feature = ReverseLayerF.apply(out_dis, alpha)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, reverse_feature

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

class ResNet20(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet20(num_classes=10):
    return ResNet20(BasicBlock1, [3, 3, 3], num_classes)


class MobileNetPathMNIST(nn.Module):
    def __init__(self, num_classes=9):
        super(MobileNetPathMNIST, self).__init__()
        # 加载预训练的 MobileNetV2
        self.mobilenet = mobilenet_v2(pretrained=True)
        
        # 修改第一层卷积层以适应 28x28 的输入
        # 原始第一层卷积：Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # 修改为：Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.mobilenet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        # 修改最后的分类层以适应 PathMNIST 的类别数
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, 
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(MobileNetV2, self).__init__()
        in_channels = input_shape[0]
        last_channel = 1280

        def make_stage(in_c, out_c, stride, expand, n):
            blocks = []
            for i in range(n):
                s = stride if i == 0 else 1
                blocks.append(InvertedResidual(in_c if i == 0 else out_c, out_c, s, expand))
            return blocks

        self.features = []

        # Initial conv
        self.features.append(nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        ))

        # Inverted residual blocks
        self.features += make_stage(32, 16, 1, 1, 1)
        self.features += make_stage(16, 24, 2, 6, 2)
        self.features += make_stage(24, 32, 2, 6, 3)
        self.features += make_stage(32, 64, 2, 6, 4)
        self.features += make_stage(64, 96, 1, 6, 3)
        self.features += make_stage(96, 160, 2, 6, 3)
        self.features += make_stage(160, 320, 1, 6, 1)

        # Final layers
        self.features.append(nn.Sequential(
            nn.Conv2d(320, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ))

        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'logistic':
        return Logistic(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == 'cnn':
        return TwoConvOneFc(options['input_shape'], options['num_class'])
    elif model_name == 'medcnn':
        return medcnn(options['input_shape'], options['num_class'])
    elif model_name == 'mobile':
        return MobileNetPathMNIST(num_classes=9)
    elif model_name == 'medresnet':
        return resnet20(num_classes=options['num_class'])
    elif model_name == 'cifar':
        return cifar(options['input_shape'], options['num_class'])
    elif model_name == 'cifarcnn':
        return ResNet18(num_classes=options['num_class'])
    elif model_name == 'lenet':
        return LeNet(options['input_shape'], options['num_class'])
    elif model_name == 'mobilenet':
        return MobileNetV2(options['input_shape'], options['num_class'])
    elif model_name.startswith('vgg'):
        mod = importlib.import_module('src.models.vgg')
        vgg_model = getattr(mod, model_name)
        return vgg_model(options['num_class'])
    elif model_name == 'sfa':
        return SFA(options['input_shape'], options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))

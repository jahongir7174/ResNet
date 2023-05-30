import copy
import math

import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU(), 3, s, 1)
        self.conv2 = Conv(out_ch, out_ch, torch.nn.Identity(), 3, 1, 1)

        if self.add_m:
            self.conv3 = Conv(in_ch, out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv2.norm.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.add_m:
            x = self.conv3(x)

        return self.relu(x + y)


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.add_m = s != 1 or in_ch != self.expansion * out_ch

        self.conv1 = Conv(in_ch, out_ch, torch.nn.ReLU())
        self.conv2 = Conv(out_ch, out_ch, torch.nn.ReLU(), 3, s, 1)
        self.conv3 = Conv(out_ch, out_ch * self.expansion, torch.nn.Identity())

        if self.add_m:
            self.conv4 = Conv(in_ch, self.expansion * out_ch, torch.nn.Identity(), s=s)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv3.norm.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.add_m:
            x = self.conv4(x)

        return self.relu(x + y)


class ResNet(torch.nn.Module):
    def __init__(self, block, depth, num_classes):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        filters = [3, 64, 128, 256, 512]

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(), 7, 2, 3))
        # p2/4
        for i in range(depth[0]):
            if i == 0:
                self.p2.append(torch.nn.MaxPool2d(3, 2, 1))
                self.p2.append(block(filters[1], filters[1], 1))
            else:
                self.p2.append(block(block.expansion * filters[1], filters[1]))
        # p3/8
        for i in range(depth[1]):
            if i == 0:
                self.p3.append(block(block.expansion * filters[1], filters[2], 2))
            else:
                self.p3.append(block(block.expansion * filters[2], filters[2], 1))
        # p4/16
        for i in range(depth[2]):
            if i == 0:
                self.p4.append(block(block.expansion * filters[2], filters[3], 2))
            else:
                self.p4.append(block(block.expansion * filters[3], filters[3], 1))
        # p5/32
        for i in range(depth[3]):
            if i == 0:
                self.p5.append(block(block.expansion * filters[3], filters[4], 2))
            else:
                self.p5.append(block(block.expansion * filters[4], filters[4], 1))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(512 * block.expansion, num_classes))

        # initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, 0, 'fan_out', 'relu')
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
        for m in self.modules():
            if hasattr(m, 'zero_init'):
                m.zero_init()

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        return self.fc(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class EMA:
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module

        m_std = model.state_dict().values()
        e_std = self.model.state_dict().values()

        for m, e in zip(m_std, e_std):
            e.copy_(self.decay * e + (1. - self.decay) * m)


class StepLR:
    def __init__(self, lr, optimizer):
        self.lr = lr
        self.decay_rate = 0.973
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1E-6

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.warmup_lr_init

    def step(self, epoch, optimizer):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr_init + epoch * (self.lr - self.warmup_lr_init) / self.warmup_epochs
        else:
            lr = self.lr * (self.decay_rate ** ((epoch - self.warmup_epochs) // self.decay_epochs))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class CosineLR:
    def __init__(self, lr, args, optimizer):
        self.lr = lr
        self.min = 1E-5
        self.max = 1E-4
        self.args = args
        self.warmup_epochs = 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.max

    def step(self, epoch, optimizer):
        epochs = self.args.epochs
        if epoch < self.warmup_epochs:
            lr = self.max + epoch * (self.lr - self.max) / self.warmup_epochs
        else:
            if epoch < epochs:
                alpha = math.pi * (epoch - (epochs * (epoch // epochs))) / epochs
                lr = self.min + 0.5 * (self.lr - self.min) * (1 + math.cos(alpha))
            else:
                lr = self.min

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-3, weight_decay=0.0,
                 momentum=0.9, centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class CrossEntropyLoss(torch.nn.Module):
    """
    NLL Loss with label smoothing.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        mean = torch.mean(prob, dim=-1)

        index = torch.unsqueeze(targets, dim=1)

        nll_loss = torch.gather(prob, -1, index)
        nll_loss = torch.squeeze(nll_loss, dim=1)

        return ((self.epsilon - 1) * nll_loss - self.epsilon * mean).mean()


def resnet_18(num_classes: int = 1000):
    return ResNet(Residual, [2, 2, 2, 2], num_classes)


def resnet_34(num_classes: int = 1000):
    return ResNet(Residual, [3, 4, 6, 3], num_classes)


def resnet_50(num_classes: int = 1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet_101(num_classes: int = 1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet_152(num_classes: int = 1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def resnet_200(num_classes: int = 1000):
    return ResNet(Bottleneck, [3, 24, 36, 3], num_classes)

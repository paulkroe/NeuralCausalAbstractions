# Code adapted from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGANdeep.py
# Written by Andy Brock and Alex Andonian, authors of
# "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
# Link: https://arxiv.org/abs/1809.11096


import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter as P


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1, norm_style='bn', ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.norm_style == 'bn':
            out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                               self.training, 0.1, self.eps)
        elif self.norm_style == 'in':
            out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                  self.training, 0.1, self.eps)
        elif self.norm_style == 'nonorm':
            out = x
        return out * gain + bias


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x):
        return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                            self.bias, self.training, self.momentum, self.eps)


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.

# Channel ratio is the ratio of
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None, channel_ratio=4):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
            # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


def G_arch(ch=64, attention='64'):
    arch = {}
    arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
    arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4,  2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
    arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
    arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

    return arch


class BigGANDeconv(nn.Module):
    def __init__(self, feature_maps=64,
                 G_depth=2,
                 input_dim=128,
                 o_channels=3,
                 bottom_width=4,
                 resolution=128,
                 G_kernel_size=3,
                 G_attn='64',
                 num_G_SVs=1,
                 num_G_SV_itrs=1,
                 hier=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5,
                 SN_eps=1e-12,
                 G_fp16=False,
                 G_init='ortho',
                 skip_init=False,
                 G_param='SN',
                 norm_style='bn',
                 **kwargs):
        super(BigGANDeconv, self).__init__()
        # Channel width mulitplier
        self.feature_maps = feature_maps
        # Number of resblocks per stage
        self.G_depth = G_depth
        # Dimensionality of input
        self.input_dim = input_dim
        # Dimensionality of output
        self.o_channels = o_channels
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # Hierarchical latent space?
        self.hier = hier
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.feature_maps, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = functools.partial(self.which_linear, bias=False)
        self.which_bn = functools.partial(ccbn,
                                          which_linear=bn_linear,
                                          input_size=self.input_dim,
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        # Prepare model
        # First linear layer
        self.linear = self.which_linear(self.input_dim,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['in_channels'][index] if g_index == 0 else
                                    self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    which_bn=self.which_bn,
                                    activation=self.activation,
                                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] and g_index == (
                                                self.G_depth - 1) else None))]
                            for g_index in range(self.G_depth)]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1]),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], o_channels
                                                          ))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z):
        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, z)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=True, activation=None, downsample=None,
                 channel_ratio=4):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels // channel_ratio
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=1, padding=0)

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels - in_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = torch.cat([x, self.conv_sc(x)], 1)
        return x

    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.conv1(F.relu(x))
        # 3x3 convs
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = self.downsample(h)
            # final 1x1 conv
        h = self.conv4(h)
        return h + self.shortcut(x)


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64'):
    arch = {}
    arch[256] = {'in_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [item * ch for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


class BigGANDisc(nn.Module):
    def __init__(self,
                 img_i_channels=3,
                 feature_maps=64,
                 D_wide=True,
                 D_depth=2,
                 resolution=128,
                 D_kernel_size=3,
                 D_attn='64',
                 num_D_SVs=1,
                 num_D_SV_itrs=1,
                 D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12,
                 output_dim=1,
                 D_fp16=False,
                 D_init='ortho',
                 skip_init=False,
                 D_param='SN',
                 **kwargs):
        super(BigGANDisc, self).__init__()
        # Number of input channels
        self.img_i_channels = img_i_channels
        # Width multiplier
        self.feature_maps = feature_maps
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # How many resblocks per stage?
        self.D_depth = D_depth
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.feature_maps, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)

        # Prepare model
        # Stem convolution
        self.input_conv = self.which_conv(self.img_i_channels, self.arch['in_channels'][0])
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(
                in_channels=self.arch['in_channels'][index] if d_index == 0 else self.arch['out_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                wide=self.D_wide,
                activation=self.activation,
                preactivation=True,
                downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index == 0 else None))
                             for d_index in range(self.D_depth)]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        # Initialize weights
        if not skip_init:
            self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x):
        # Run input conv
        h = self.input_conv(x)
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        return out

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.init as init
import math

from src.datagen import SCMDataTypes as sdt
from .mlp import MLP_Module
from .cnn import CNN_Module, CNN_Deconv_Module
from .biggan import BigGANDeconv, BigGANDisc

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.datagen import SCMDataTypes as sdt
from .mlp import MLP_Module
from .cnn import CNN_Module, CNN_Deconv_Module
from .biggan import BigGANDeconv, BigGANDisc

class ResidualBlock(nn.Module):
    """
    A simple residual block for MLP layers.
    """
    def __init__(self, dim, use_layer_norm=True):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dim, dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.net(x))


class CustomNN(nn.Module):
    def __init__(
        self,
        pa_size,
        u_size,
        o_size,
        pa_type,
        o_type,
        img_size=32,
        img_embed_size=4,
        feature_maps=64,
        h_size=64,
        h_layers=3,
        use_batch_norm=True,
        mode="dcgan"
    ):
        super().__init__()
        # Store sizes and types
        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.pa_type = pa_type
        self.u_size = u_size
        self.o_size = o_size
        self.o_type = o_type
        self.feature_maps = feature_maps
        self.mode = mode

        # Split endogenous parents into image vs. real
        self.img_pa, self.real_pa = set(), set()
        self.img_i_size = self.real_i_size = 0
        for v in self.pa:
            if pa_type[v] == sdt.IMAGE:
                self.img_pa.add(v)
                self.img_i_size += pa_size[v]
            else:
                self.real_pa.add(v)
                self.real_i_size += pa_size[v]
        # Add exogenous dims
        self.real_i_size += sum(u_size[k] for k in self.u)

        # Image CNN setup
        self.img_size = img_size
        self.img_h_layers = int(round(np.log2(img_size))) - 3
        self.img_embed_size = img_embed_size
        use_cnn = self.img_i_size > 0
        use_deconv = (o_type == sdt.IMAGE)

        self.cnn_mod = None
        if use_cnn:
            self.real_i_size += img_embed_size
            if self.mode == "biggan_disc":
                self.cnn_mod = BigGANDisc(
                    img_i_channels=self.img_i_size,
                    feature_maps=feature_maps,
                    resolution=img_size,
                    output_dim=img_embed_size
                )
            else:
                self.cnn_mod = CNN_Module(
                    self.img_i_size,
                    img_embed_size,
                    feature_maps=feature_maps,
                    h_layers=self.img_h_layers,
                    use_batch_norm=use_batch_norm
                )

        # MLP (with residuals for non-image outputs)
        self.mlp_mod = None
        # Always build if any real inputs or deconv branch
        if True:
            mlp_out = o_size if not use_deconv else h_size
            if not use_deconv:
                layers = []
                # initial projection
                layers.append(nn.Linear(self.real_i_size, h_size))
                if use_batch_norm:
                    layers.append(nn.LayerNorm(h_size))
                layers.append(nn.ReLU(inplace=True))
                # residual blocks
                for _ in range(h_layers):
                    layers.append(ResidualBlock(h_size, use_layer_norm=use_batch_norm))
                # final projection
                layers.append(nn.Linear(h_size, mlp_out))
                self.mlp_mod = nn.Sequential(*layers)
            else:
                self.mlp_mod = MLP_Module(
                    self.real_i_size,
                    mlp_out,
                    h_size=h_size,
                    h_layers=h_layers,
                    use_layer_norm=use_batch_norm
                )

        # Deconvolution for image output
        self.deconv_mod = None
        if use_deconv:
            in_dim = h_size if self.mlp_mod is not None else self.real_i_size
            if self.mode == "dcgan":
                self.deconv_mod = CNN_Deconv_Module(
                    in_dim,
                    o_size,
                    feature_maps=feature_maps,
                    h_layers=self.img_h_layers,
                    use_batch_norm=use_batch_norm
                )
            else:
                self.deconv_mod = BigGANDeconv(
                    feature_maps=feature_maps,
                    input_dim=in_dim,
                    o_channels=o_size,
                    resolution=img_size
                )

        # Output activation
        self.activation = None
        if o_type in (sdt.BINARY, sdt.REP_BINARY):
            self.activation = nn.Sigmoid()
        elif o_type in (sdt.BINARY_ONES, sdt.REP_BINARY_ONES):
            self.activation = nn.Tanh()
        elif o_type == sdt.ONE_HOT:
            self.activation = nn.Softmax(dim=1)

        # A dummy parameter to bind the module to device
        self.device_param = nn.Parameter(T.empty(0))

        # Initialize weights with orthogonal for Linear and Kaiming for Conv
        self._init_depth_aware()

    def _init_depth_aware(self):
        # collect all the nn.Linear layers in the MLP
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        L = len(linear_layers)

        for idx, layer in enumerate(linear_layers):
            # layer index from 0…L-1
            # Option A: uniform scaling so total var stays ~1
            scale = 1.0 / math.sqrt(L)

            # Option B: progressive scaling so early layers get slightly
            # higher variance and late layers slightly lower:
            # scale = math.sqrt((idx+1) / L)

            gain = nn.init.calculate_gain('relu') * scale
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, pa, u, include_inp=False):
        # CNN forward
        img_out = None
        if self.cnn_mod:
            img_inp = T.cat([pa[k] for k in self.pa if k in self.img_pa], dim=1)
            img_out = self.cnn_mod(img_inp)

        # Build real inputs
        if not u and not self.real_pa:
            cur = img_out
        else:
            parts = []
            if self.real_pa:
                parts.append(T.cat([pa[k] for k in self.pa if k in self.real_pa], dim=1))
            if u:
                parts.append(T.cat([u[k] for k in self.u], dim=1))
            cur = T.cat(parts, dim=1)
            real_inp = cur
            if img_out is not None:
                cur = T.cat([cur, img_out], dim=1)

        # MLP forward
        if self.mlp_mod:
            cur = self.mlp_mod(cur)
        # Deconv forward
        if self.deconv_mod:
            cur = self.deconv_mod(cur)
        # Activation
        if self.activation:
            cur = self.activation(cur)
        # Optional inputs return
        if include_inp:
            return cur, (img_out, real_inp)
        return cur
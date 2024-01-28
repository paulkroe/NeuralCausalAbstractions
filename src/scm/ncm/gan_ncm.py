import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.scm.distribution.continuous_distribution import UniformDistribution, NeuralDistribution
from src.scm.nn.custom_nn import CustomNN
from src.scm.scm import SCM, expand_do
from src.datagen import SCMDataTypes as sdt


class GAN_NCM(SCM):
    def __init__(self, cg, v_size={}, v_type={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None):
        """
        General implementation of the GAN-NCM. Custom NN functions are used for each NCM function, depending on the
        data type of the inputs and outputs. Variable domains directly correspond to the type of the data.

        cg: Causal diagram inductive bias for the NCM.
        v_size: Dictionary of endogenous variable dimensionalities.
        v_type: Dictionary of endogenous variable data types.
        default_v_size: Inferred dimensionality of an endogenous variable if not provided.
        u_size: Dictionary of exogenous variable dimensionalities.
        default_u_size: Inferred dimensionality of an exogenous variable if not provided.
        f: Dictionary of functions in which, if variable V is present in f, then the NCM function for V will be
        overridden by the choice in f.
        hyperparams: NCM hyperparameters passed from main.
        """
        if hyperparams is None:
            hyperparams = dict()

        self.gan_mode = hyperparams.get("gan-mode", "vanilla")
        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size[k] if k in v_size else default_v_size for k in self.cg}
        self.v_type = v_type

        # Scale u size if necessary
        if hyperparams['scale-u-size']:
            for c2 in self.cg.c2:
                uv_size = 0
                for var in c2:
                    uv_size += self.v_size[var]
                self.u_size[c2] *= uv_size

        h_size = {k: hyperparams['h-size'] for k in self.v_size}
        if hyperparams['scale-h-size']:
            for var in self.v_size:
                inp_size = sum([self.u_size[k] for k in self.cg.v2c2[var]])
                for pa in self.cg.pa[var]:
                    if self.v_type[pa] == sdt.IMAGE:
                        inp_size += self.v_size[pa] * hyperparams['img-size']
                    else:
                        inp_size += self.v_size[pa]
                h_size[var] *= (inp_size + self.v_size[var])
        total_img_h_size = sum([h_size[k] for k in h_size if self.v_type[k] == sdt.IMAGE])

        # Constructs each of the NCM functions (i.e. the generators).
        gens = {}
        for var in self.cg:
            if var in f:
                gens[var] = f[var]
            else:
                gens[var] = CustomNN({k: self.v_size[k] for k in self.cg.pa[var]},
                                     {k: self.u_size[k] for k in self.cg.v2c2[var]},
                                     self.v_size[var], v_type, v_type[var], img_size=hyperparams['img-size'],
                                     img_embed_size=total_img_h_size, feature_maps=hyperparams['feature-maps'],
                                     h_size=h_size[var], h_layers=hyperparams['h-layers'],
                                     use_batch_norm=hyperparams['batch-norm'], mode=hyperparams['gan-arch'])
        gens = nn.ModuleDict(gens)

        # Determines whether P(U) is modeled by a uniform distribution or a neural-parameterized distribution.
        if hyperparams['neural-pu']:
            pu_dist = NeuralDistribution(self.cg.c2, self.u_size, hyperparams)
        else:
            pu_dist = UniformDistribution(self.cg.c2, self.u_size)

        # Inherits SCM functionality.
        super().__init__(
            v=list(cg),
            f=gens,
            pu=pu_dist
        )

    def convert_evaluation(self, samples):
        """
        If the output is intended to be a binary or one_hot vector, then it must be rounded.
        """
        convert_samples = {}
        for var in samples:
            if self.v_type[var] == sdt.BINARY:
                convert_samples[var] = T.gt(samples[var], 0.5).float()
            elif self.v_type[var] == sdt.BINARY_ONES:
                convert_samples[var] = (2 * T.gt(samples[var], 0.0).float()) - 1
            elif self.v_type[var] == sdt.ONE_HOT:
                convert_samples[var] = F.one_hot(T.argmax(samples[var], dim=1), num_classes=self.v_size[var])
            else:
                convert_samples[var] = samples[var]
        return convert_samples


class Discriminator(nn.Module):
    def __init__(self, v_size, v_type, disc_use_sigmoid=True, hyperparams=None):
        """
        The discriminator is a custom NN which maps all variables to a single value, which may pass through an
        activation depending on the GAN type.
        """
        super().__init__()

        h_size = {k: hyperparams['h-size'] for k in v_size}
        total_h_size = sum([h_size[k] for k in h_size])
        total_img_h_size = sum([h_size[k] for k in h_size if v_type[k] == sdt.IMAGE])

        disc_h_size = total_h_size
        if hyperparams.get('disc-h-size', -1) > 0:
            disc_h_size = hyperparams['disc-h-size']

        disc_type = sdt.REAL
        if disc_use_sigmoid:
            disc_type = sdt.BINARY

        self.f_disc = CustomNN(v_size, {}, 1, v_type, disc_type, img_size=hyperparams['img-size'],
                               img_embed_size=total_img_h_size,
                               feature_maps=hyperparams['feature-maps'],
                               h_size=disc_h_size, h_layers=hyperparams['disc-h-layers'],
                               use_batch_norm=hyperparams['batch-norm'],
                               mode="{}_disc".format(hyperparams['disc-type']))

    def forward(self, samples, include_inp=False):
        return self.f_disc(samples, {}, include_inp=include_inp)

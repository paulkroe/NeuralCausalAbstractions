import numpy as np
import torch as T
import torch.nn as nn

from src.datagen import SCMDataTypes as sdt
from .mlp import MLP_Module
from .cnn import CNN_Module, CNN_Deconv_Module
from .biggan import BigGANDeconv, BigGANDisc


class CustomNN(nn.Module):
    def __init__(self, pa_size, u_size, o_size, pa_type, o_type, img_size=32,
                 img_embed_size=4, feature_maps=64, h_size=64, h_layers=3,
                 use_batch_norm=True, mode="dcgan"):
        """
        Creates an NN function that matches the types for the inputs and outputs.
        Image inputs are passed through a CNN, which outputs an embedding.
        Real inputs, including other endogenous inputs, exogenous inputs, and the image embedding, are passed through
        an MLP, which outputs a real vector.
        The real vector is returned unless the output type is an image, in which case it is passed through a
        deconvolutional layer.
        The final output is passed through an activation function that depends on the output type.

        pa_size: Dictionary of endogenous parent dimensionality. Image dimensions are number of channels.
        u_size: Dictionary of exogenous parent dimensionality.
        o_size: Output dimensionality.
        o_type: Output type.
        img_size: Width or length of image data.
        img_embed_size: Dimensionality of the image embeddings processed by the CNN.
        feature_maps: Number of feature maps between convolutional layers.
        h_size: Width of MLP layers.
        h_layers: Number of MLP layers.
        use_batch_norm: set True to use layer norm or batch norm after each layer
        mode: dcgan, biggan
        """
        super().__init__()
        self.pa = sorted(pa_size)  # endogenous parent names in alphabetical order
        self.set_pa = set(self.pa)  # endogenous parents in set form for O(1) access
        self.u = sorted(u_size)  # exogenous parent names in alphabetical order
        self.pa_size = pa_size  # dictionary of endogenous parent dimensions
        self.pa_type = pa_type  # dictionary of endogenous parent types
        self.u_size = u_size  # dictionary of exogenous parent dimensions
        self.o_size = o_size  # dimensionality of output
        self.o_type = o_type  # output type
        self.feature_maps = feature_maps  # feature map hyperparameter
        self.mode = mode

        # Separating endogenous parents into images and non-images
        self.img_pa = set()
        self.real_pa = set()
        self.img_i_size = 0
        self.real_i_size = 0
        for v in self.pa:
            if pa_type[v] == sdt.IMAGE:
                self.img_i_size += pa_size[v]
                self.img_pa.add(v)
            else:
                self.real_i_size += pa_size[v]
                self.real_pa.add(v)
        # Exogenous parents are grouped with non-image features as inputs to NN
        self.real_i_size += sum(self.u_size[k] for k in self.u_size)

        self.img_size = img_size
        self.img_h_layers = int(round(np.log2(img_size))) - 3  # calculate number of CNN layers required based on image size
        self.img_embed_size = img_embed_size  # image embedding size hyperparameter

        use_cnn = self.img_i_size > 0  # do not use CNNs unless image inputs are present
        use_deconv = self.o_type == sdt.IMAGE  # do not use deconv module unless output is an image
        # always use MLP unless there is no real input and the output is an image
        use_mlp = True
        if use_deconv:
            use_mlp = self.real_i_size > 0

        # Construct CNN portion of the network
        self.cnn_mod = None
        if use_cnn:
            self.real_i_size += img_embed_size  # need to add embedding to the MLP portion
            if self.mode == "biggan_disc":
                self.cnn_mod = BigGANDisc(img_i_channels=self.img_i_size, feature_maps=feature_maps,
                                          resolution=self.img_size, output_dim=img_embed_size)
            else:
                self.cnn_mod = CNN_Module(self.img_i_size, img_embed_size, feature_maps=feature_maps,
                                          h_layers=self.img_h_layers, use_batch_norm=use_batch_norm)

        # Construct MLP portion of the network
        self.mlp_mod = None
        if use_mlp:
            mlp_out_size = o_size
            if use_deconv:
                mlp_out_size = h_size
            self.mlp_mod = MLP_Module(self.real_i_size, mlp_out_size, h_size=h_size, h_layers=h_layers,
                                            use_layer_norm=use_batch_norm)

        # Construct deconvolutional portion of the network
        self.deconv_mod = None
        if use_deconv:
            deconv_in_size = self.real_i_size
            if use_mlp:
                deconv_in_size = h_size
            if self.mode == "dcgan":
                self.deconv_mod = CNN_Deconv_Module(deconv_in_size, o_size, feature_maps=feature_maps,
                                                h_layers=self.img_h_layers, use_batch_norm=use_batch_norm)
            elif self.mode == "biggan":
                self.deconv_mod = BigGANDeconv(feature_maps=self.feature_maps, input_dim=deconv_in_size,
                                               o_channels=o_size, resolution=self.img_size)

        # Choose activation based on output type
        self.activation = None
        if o_type == sdt.BINARY or o_type == sdt.REP_BINARY:
            self.activation = nn.Sigmoid()
        elif o_type == sdt.BINARY_ONES or o_type == sdt.REP_BINARY_ONES:
            self.activation = nn.Tanh()
        elif o_type == sdt.ONE_HOT:
            self.activation = nn.Softmax(dim=1)

        self.device_param = nn.Parameter(T.empty(0))

    def forward(self, pa, u, include_inp=False):
        # CNN forward pass
        img_inp = None
        img_out = None
        if self.cnn_mod is not None:
            # Concatenate all image parents together and process through CNN
            img_inp = T.cat([pa[k] for k in self.pa if k in self.img_pa], dim=1)
            img_out = self.cnn_mod(img_inp)

        # Construct real input
        real_inp = None
        cur_inp = None
        if len(u.keys()) == 0 and len(self.real_pa) == 0:
            cur_inp = img_out
        else:
            # Concatenate all real endogenous and exogenous parents together
            if len(u.keys()) == 0:
                cur_inp = T.cat([pa[k] for k in self.pa if k in self.real_pa], dim=1)
            elif len(pa.keys()) == 0 or len(set(pa.keys()).intersection(self.set_pa)) == 0:
                cur_inp = T.cat([u[k] for k in self.u], dim=1)
            else:
                inp_u = T.cat([u[k] for k in self.u], dim=1)
                inp_pa = T.cat([pa[k] for k in self.pa if k in self.real_pa], dim=1)
                cur_inp = T.cat((inp_pa, inp_u), dim=1)

            # Save all non-image inputs
            real_inp = cur_inp

            # Concatenate image embedding if exists
            if img_out is not None:
                cur_inp = T.cat((cur_inp, img_out), dim=1)

        # MLP forward pass
        if self.mlp_mod is not None:
            # Process through MLP
            cur_inp = self.mlp_mod(cur_inp)

        # Deconv forward pass
        if self.deconv_mod is not None:
            cur_inp = self.deconv_mod(cur_inp)

        # Apply activation, if applicable
        if self.activation is not None:
            cur_inp = self.activation(cur_inp)

        # Return input with output, if gradients are needed
        if include_inp:
            return cur_inp, (img_inp, real_inp)

        return cur_inp


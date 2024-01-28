import torch as T
import torch.nn as nn


class CNN_Module(nn.Module):
    def __init__(self, i_channels, o_size, feature_maps=64, h_layers=3, use_batch_norm=True):
        """
        Maps images to real vectors with the following shapes:
            input shape: (i_channels, img_size, img_size)
            output shape: o_size * ((img_size / (2 ** (h_layers + 1))) - 3) ** 2
            output shape is o_size if img_size == 2 ** (h_layers + 3)

        Other parameters:
            feature_maps: number of feature maps between convolutional layers
            use_batch_norm: set True to use batch norm after each layer
        """
        super().__init__()
        self.i_channels = i_channels
        self.feature_maps = feature_maps
        self.o_size = o_size

        bias = not use_batch_norm

        # Shape: (b, i_channels, img_size, img_size)
        conv_layers = [nn.Conv2d(i_channels, feature_maps, 4, 2, 1, bias=True),
                       nn.LeakyReLU(0.2, inplace=True)]
        # Shape: (b, feature_maps, img_size / 2, img_size / 2)

        for h in range(h_layers):
            # Shape: (b, 2 ** h * feature_maps, img_size / (2 ** (h + 1)), img_size / (2 ** (h + 1)))
            conv_layers.append(nn.Conv2d(2 ** h * feature_maps,
                                         2 ** (h + 1) * feature_maps, 4, 2, 1, bias=bias))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(2 ** (h + 1) * feature_maps))
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            # Shape: (b, 2 ** (h + 1) * feature_maps, img_size / (2 ** (h + 2)), img_size / (2 ** (h + 2)))

        # Shape: (b, 2 ** h_layers * feature_maps, img_size / (2 ** (h_layers + 1)), img_size / (2 ** (h_layers + 1)))
        conv_layers.append(nn.Conv2d(2 ** h_layers * feature_maps, o_size, 4, 1, 0, bias=True))
        # Shape: (b, o_size, (img_size / (2 ** (h_layers + 1))) - 3, (img_size / (2 ** (h_layers + 1))) - 3)

        self.conv_nn = nn.Sequential(*conv_layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.conv_nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, include_inp=False):
        out = T.reshape(self.conv_nn(x), (x.shape[0], -1))
        if include_inp:
            return out, x
        return out


class CNN_Deconv_Module(nn.Module):
    def __init__(self, i_size, o_channels, feature_maps=64, h_layers=3, use_batch_norm=True):
        """
        Maps real vectors to images with the following shapes:
            input shape: i_size
            output shape: (o_channels, 2 ** (h_layers + 3), 2 ** (h_layers + 3))

        Other parameters:
            feature_maps: number of feature maps between deconvolutional layers
            use_batch_norm: set True to use batch norm after each layer
        """
        super().__init__()
        self.i_size = i_size
        self.o_channels = o_channels
        self.feature_maps = feature_maps

        bias = not use_batch_norm

        # Shape: (b, i_size, 1, 1)
        layers = [nn.ConvTranspose2d(self.i_size, 2 ** h_layers * feature_maps, 4, 1, 0, bias=bias)]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(2 ** h_layers * feature_maps))
        layers.append(nn.ReLU(True))
        # Shape: (b, 2 ** h_layers * feature_maps, 4, 4)

        for h in range(h_layers):
            # Shape: (b, 2 ** (h_layers - h) * feature_maps, 2 ** (h + 2), 2 ** (h + 2))
            layers.append(nn.ConvTranspose2d(2 ** (h_layers - h) * feature_maps,
                                             2 ** (h_layers - h - 1) * feature_maps,
                                             4, 2, 1, bias=bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(2 ** (h_layers - h - 1) * feature_maps))
            layers.append(nn.ReLU(True))
            # Shape: (b, 2 ** (h_layers - h - 1) * feature_maps, 2 ** (h + 3), 2 ** (h + 3))

        # Shape: (b, feature_maps, 2 ** (h_layers + 2), 2 ** (h_layers + 2))
        layers.append(nn.ConvTranspose2d(feature_maps, o_channels, 4, 2, 1, bias=True))
        layers.append(nn.Tanh())
        # Shape: (b, o_channels, 2 ** (h_layers + 3), 2 ** (h_layers + 3))

        self.conv_nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.conv_nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x, include_inp=False):
        x = T.reshape(x, (x.shape[0], -1, 1, 1))
        if include_inp:
            return self.conv_nn(x), x
        return self.conv_nn(x)


class CNN_Deconv(nn.Module):
    def __init__(self, pa_size, u_size, o_channels, feature_maps=64, h_layers=3, use_batch_norm=True):
        super().__init__()
        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.u_size = u_size
        self.o_channels = o_channels
        self.feature_maps = feature_maps

        self.i_size = sum(self.pa_size[k] for k in self.pa_size) + sum(self.u_size[k] for k in self.u_size)

        bias = not use_batch_norm

        layers = [nn.ConvTranspose2d(self.i_size, 2 ** h_layers * feature_maps, 4, 1, 0, bias=bias)]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(2 ** h_layers * feature_maps))
        layers.append(nn.ReLU(True))
        for h in range(h_layers):
            layers.append(nn.ConvTranspose2d(2 ** (h_layers - h) * feature_maps,
                                             2 ** (h_layers - h - 1) * feature_maps,
                                             4, 2, 1, bias=bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(2 ** (h_layers - h - 1) * feature_maps))
            layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(feature_maps, o_channels, 4, 2, 1, bias=True))
        layers.append(nn.Tanh())

        self.nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, pa, u, include_inp=False):
        if len(u.keys()) == 0:
            inp = T.cat([pa[k] for k in self.pa], dim=1)
        elif len(pa.keys()) == 0 or len(set(pa.keys()).intersection(self.set_pa)) == 0:
            inp = T.cat([u[k] for k in self.u], dim=1)
        else:
            inp_u = T.cat([u[k] for k in self.u], dim=1)
            inp_pa = T.cat([pa[k] for k in self.pa], dim=1)
            inp = T.cat((inp_pa, inp_u), dim=1)

        inp = inp.unsqueeze(dim=-1).unsqueeze(dim=-1)

        if include_inp:
            return self.nn(inp), inp

        return self.nn(inp)


class CNN(nn.Module):
    def __init__(self, real_size, img_channels, img_out_size, o_size, feature_maps=64, h_size=128, h_layers=3,
                 use_sigmoid=False, use_batch_norm=True):
        super().__init__()
        self.real_pa = sorted(real_size)
        self.set_real_pa = set(self.real_pa)
        self.img_pa = sorted(img_channels)
        self.set_img_pa = set(self.img_pa)
        self.real_size = real_size
        self.img_channels = img_channels
        self.img_out_size = img_out_size
        self.feature_maps = feature_maps
        self.h_size = h_size
        self.o_size = o_size

        self.total_channels = sum(self.img_channels[k] for k in self.img_channels)
        self.total_size = sum(self.real_size[k] for k in self.real_size) + img_out_size

        bias = not use_batch_norm

        conv_layers = [nn.Conv2d(self.total_channels, feature_maps, 4, 2, 1, bias=True),
                       nn.LeakyReLU(0.2, inplace=True)]
        for h in range(h_layers):
            conv_layers.append(nn.Conv2d(2 ** h * feature_maps,
                                         2 ** (h + 1) * feature_maps, 4, 2, 1, bias=bias))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(2 ** (h + 1) * feature_maps))
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        conv_layers.append(nn.Conv2d(2 ** h_layers * feature_maps, img_out_size, 4, 1, 0, bias=True))
        conv_layers.append(nn.Sigmoid())

        self.conv_nn = nn.Sequential(*conv_layers)

        mlp_layers = [nn.Linear(self.total_size, self.h_size)]
        if use_batch_norm:
            mlp_layers.append(nn.LayerNorm(self.h_size))
        mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
        for l in range(h_layers - 1):
            mlp_layers.append(nn.Linear(self.h_size, self.h_size))
            if use_batch_norm:
                mlp_layers.append(nn.LayerNorm(self.h_size))
            mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
        mlp_layers.append(nn.Linear(self.h_size, o_size))
        if use_sigmoid:
            mlp_layers.append(nn.Sigmoid())

        self.mlp_nn = nn.Sequential(*mlp_layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.conv_nn.apply(self.init_weights)
        self.mlp_nn.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight,
                                   gain=T.nn.init.calculate_gain('relu'))

    def forward(self, real_pa, img_pa, include_inp=False):
        img_inp = T.cat([img_pa[k] for k in self.img_pa], dim=1)
        conv_out = self.conv_nn(img_inp).squeeze()

        real_inp = None
        if len(self.real_pa) > 0:
            real_inp = T.cat([real_pa[k] for k in self.real_pa], dim=1)
            conv_out = T.cat([conv_out, real_inp], dim=1)

        out = self.mlp_nn(conv_out)

        if include_inp:
            return out, img_inp, real_inp

        return out



if __name__ == '__main__':
    s = CNN_Deconv_Module(4, 3, h_layers=3)
    print(s)
    x = T.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]).float()
    out_samp = s(x)

    print(out_samp)
    print(out_samp.shape)

    s2 = CNN_Module(3, 5, h_layers=3)
    print(s2)
    out = s2(out_samp)

    print(out)
    print(out.shape)

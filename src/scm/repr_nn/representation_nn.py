import torch as T
import torch.nn as nn

from src.datagen import SCMDataTypes as sdt
from src.scm.nn.custom_nn import CustomNN


class RepresentationalNN(nn.Module):
    def __init__(self, cg, v_size, v_type, default_v_size=1, hyperparams=None):
        super().__init__()
        self.cg = cg
        self.v = sorted(v_type.keys())
        self.v_size = {k: v_size[k] if k in v_size else default_v_size for k in v_type}
        self.v_type = v_type

        if hyperparams['rep-image-only']:
            self.encode_v = [v for v in self.v if v_type[v] == sdt.IMAGE]
        else:
            self.encode_v = self.v
        self.encode_v = set(self.encode_v)

        if hyperparams is None:
            hyperparams = dict()

        self.encoders = nn.ModuleDict({
            v: CustomNN({v: v_size[v]}, {}, hyperparams['rep-size'], {v: v_type[v]}, hyperparams['rep-type'],
                        img_size=hyperparams['img-size'],
                        img_embed_size=hyperparams['rep-h-size'], feature_maps=hyperparams['rep-feature-maps'],
                        h_size=hyperparams['rep-h-size'], h_layers=hyperparams['rep-h-layers'],
                        use_batch_norm=hyperparams['batch-norm'], mode=hyperparams['gan-arch'])
            for v in self.encode_v
        })

        self.decoders = nn.ModuleDict({
            v: CustomNN({v: hyperparams['rep-size']}, {}, v_size[v], {v: hyperparams['rep-type']}, v_type[v],
                        img_size=hyperparams['img-size'],
                        img_embed_size=hyperparams['rep-h-size'], feature_maps=hyperparams['rep-feature-maps'],
                        h_size=hyperparams['rep-h-size'], h_layers=hyperparams['rep-h-layers'],
                        use_batch_norm=hyperparams['batch-norm'], mode=hyperparams['gan-arch'])
            for v in self.encode_v
        })

        self.parent_heads = None
        if hyperparams['repr'] == "auto_enc_conditional":
            self.parent_heads = nn.ModuleDict({
                v: nn.Sequential(nn.Linear(hyperparams['rep-size'], sum([v_size[x] for x in cg.pa[v]])),
                                 nn.Sigmoid())
                for v in self.encode_v
            })

        self.device_param = nn.Parameter(T.empty(0))

    def encode(self, v_dict):
        return {
            v: self.encoders[v]({v: v_dict[v]}, {}) if v in self.encode_v else v_dict[v] for v in v_dict
        }

    def decode(self, rep_dict):
        return {
            v: self.decoders[v]({v: rep_dict[v]}, {}) if v in self.encode_v else rep_dict[v] for v in rep_dict
        }

    def classify(self, v_dict, rep_dict):
        if self.parent_heads is None:
            return None, None

        out = dict()
        truth = dict()
        for v in v_dict:
            if v in self.encode_v:
                pa_list = []
                for x in self.cg.pa[v]:
                    if self.v_type[x] == sdt.BINARY_ONES:
                        pa_list.append((v_dict[x] + 1) / 2)
                    elif self.v_type[x] == sdt.BINARY or self.v_type[x] == sdt.ONE_HOT:
                        pa_list.append(v_dict[x])
                truth[v] = T.cat(pa_list, dim=1)
                out[v] = self.parent_heads[v](rep_dict[v])

        return out, truth

    def forward(self, v_dict, classify=False):
        if classify:
            rep_dict = self.encode(v_dict)
            label_out, label_truth = self.classify(v_dict, rep_dict)
            recon = self.decode(rep_dict)
            return recon, label_out, label_truth
        return self.decode(self.encode(v_dict))

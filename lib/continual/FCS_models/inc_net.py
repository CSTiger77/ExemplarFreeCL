import copy
import logging
import torch
from torch import nn
from .convs.cifar_resnet import resnet32
from .convs.resnet import resnet18, resnet34, resnet50
from .convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from .convs.ucir_resnet import resnet18 as cosine_resnet18
from .convs.ucir_resnet import resnet34 as cosine_resnet34
from .convs.ucir_resnet import resnet50 as cosine_resnet50
from .convs.linears import FCSSimpleLinear
from .convs.modified_represnet import resnet18_rep, resnet34_rep
from .convs.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam


def get_convnet(cfg, pretrained=False):
    name = cfg.extractor.TYPE.lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained, dataset_name=cfg.DATASET.dataset_name)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, cfg, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(cfg, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class FCSIncrementalNet(BaseNet):
    def __init__(self, cfg, pretrained, gradcam=False):
        super().__init__(cfg, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = FCSSimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class FCSNet(FCSIncrementalNet):
    def __init__(self, cfg, pretrained, gradcam=False):
        super().__init__(cfg, pretrained, gradcam)
        self.cfg = cfg

        self.transfer = FCSSimpleLinear(self.feature_dim, self.feature_dim)

    def update_fc(self, num_old, num_total, num_aux):

        fc = self.generate_fc(self.feature_dim, num_total + num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

        transfer = FCSSimpleLinear(self.feature_dim, self.feature_dim)

        transfer.weight = nn.Parameter(torch.eye(self.feature_dim))
        transfer.bias = nn.Parameter(torch.zeros(self.feature_dim))

        del self.transfer
        self.transfer = transfer

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
                self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())

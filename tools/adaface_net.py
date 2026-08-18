"""
tools/adaface_net.py
====================
The AdaFace backbone, in PyTorch, purely so a released checkpoint can be
converted to ONNX once. NOTHING AT RUNTIME IMPORTS THIS - the pipeline
runs the ONNX graph through onnxruntime and never loads torch for face
recognition.

Why it is vendored
------------------
AdaFace is published as a PyTorch Lightning checkpoint, which is a
state dict and nothing else: it does not carry the architecture. Turning
one into ONNX therefore needs the network definition, and requiring
every deployment to clone the AdaFace repository first is a step that
tends not to happen on a site machine.

Is it safe to reconstruct?
--------------------------
Yes, because the failure mode is loud. The weights are loaded with
strict=True: every parameter name and every tensor shape must match
exactly or it raises. A reconstruction that is subtly wrong cannot load
a checkpoint at all, so there is no "it converted but the numbers are
quietly wrong" outcome. tools/export_adaface_onnx.py additionally
compares the exported graph's output against the PyTorch model's on a
fixed input and refuses to keep the file if they disagree.

If a future checkpoint uses an architecture this file does not know,
export it with the official repository instead:
    python tools/export_adaface_onnx.py --ckpt <file> --repo <AdaFace dir>

The backbone is the IResNet ("IR") used across the ArcFace family:
a 3x3 stem, four stages of pre-activation residual blocks, and a
flatten-and-project head producing 512 numbers. AdaFace's own forward
returns (normalised_feature, norm); the export takes the feature.
"""
import os
from collections import namedtuple

import torch
from torch import nn


Block = namedtuple("Block", ["in_channel", "depth", "stride"])


def _stage(in_channel, depth, num_units, stride=2):
    return ([Block(in_channel, depth, stride)] +
            [Block(depth, depth, 1) for _ in range(num_units - 1)])


def _blocks(num_layers):
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    else:
        # 152 and 200 exist in the family but switch to bottleneck blocks
        # and a 2048-wide head, which this file does not reconstruct. No
        # AdaFace checkpoint has been released at those depths; if one is,
        # convert it with the official repository (--repo).
        raise ValueError(
            f"unsupported depth {num_layers}. This file covers the released "
            f"AdaFace backbones (ir_18, ir_50, ir_101). For anything else "
            f"pass --repo <AdaFace checkout> to use the official definition.")
    widths = [64, 128, 256, 512]
    stages, in_channel = [], 64
    for depth, count in zip(widths, units):
        stages.append(_stage(in_channel, depth, count))
        in_channel = depth
    return stages


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * w


class BasicBlockIR(nn.Module):
    """Pre-activation residual block: BN -> 3x3 -> BN -> PReLU -> 3x3 -> BN."""

    def __init__(self, in_channel, depth, stride, use_se=False):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        layers = [
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        ]
        if use_se:
            layers.append(SEModule(depth, 16))
        self.res_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Backbone(nn.Module):
    """IR-18 / IR-50 / IR-101 as AdaFace publishes them."""

    def __init__(self, input_size=(112, 112), num_layers=100, mode="ir"):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))

        use_se = mode == "ir_se"
        modules = []
        for stage in _blocks(num_layers):
            for block in stage:
                modules.append(BasicBlockIR(block.in_channel, block.depth,
                                            block.stride, use_se))
        self.body = nn.Sequential(*modules)

        spatial = 7 if input_size[0] == 112 else 14
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * spatial * spatial, 512),
            nn.BatchNorm1d(512, affine=False))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.div(x, norm), norm


class FeatureOnly(nn.Module):
    """Export wrapper: one input, one output.

    AdaFace returns (feature, norm). The norm is only used by the loss
    during training; keeping it in the exported graph means every
    consumer has to know which output is which, and getting that wrong
    produces a 1-number "embedding" that fails in a confusing way much
    later. One tensor out, no ambiguity.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        feature, _norm = self.backbone(x)
        return feature


ARCHITECTURES = {
    "ir_18": (18, "ir"), "ir18": (18, "ir"),
    "ir_34": (34, "ir"), "ir34": (34, "ir"),
    "ir_50": (50, "ir"), "ir50": (50, "ir"),
    "ir_101": (100, "ir"), "ir101": (100, "ir"),
    "ir_se_50": (50, "ir_se"), "ir_se_101": (100, "ir_se"),
}


def guess_architecture(name):
    """Read the architecture out of a checkpoint filename.

    'adaface_ir101_webface12m.ckpt' -> 'ir101'
    """
    lowered = os.path.basename(name).lower()
    for key in ("ir_se_101", "ir_se_50", "ir_101", "ir101", "ir_50", "ir50",
                "ir_34", "ir34", "ir_18", "ir18"):
        if key in lowered:
            return key
    return None


def build(architecture="ir_101", input_size=(112, 112)):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"expected one of {sorted(set(ARCHITECTURES))}")
    layers, mode = ARCHITECTURES[architecture]
    return Backbone(input_size, layers, mode)

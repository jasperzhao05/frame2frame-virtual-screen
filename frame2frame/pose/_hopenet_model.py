"""Checkpoint-compatible Hopenet network construction.

The adapter in :mod:`frame2frame.pose.hopenet` owns model assets, preprocessing,
face crops, and public pose conversion. This module owns only the published
ResNet-50 architecture. Framework imports stay inside the factory so importing
``frame2frame`` never requires the optional Torch stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_hopenet(num_bins: int = 66) -> Any:
    """Build the checkpoint-compatible three-head Hopenet network."""
    import torch.nn as nn
    from torchvision.models.resnet import Bottleneck

    class Hopenet(nn.Module):
        def __init__(self, block: Any, layers: Sequence[int], bins: int) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7)
            features = 512 * block.expansion
            self.fc_yaw = nn.Linear(features, bins)
            self.fc_pitch = nn.Linear(features, bins)
            self.fc_roll = nn.Linear(features, bins)
            # The published checkpoint contains this unused compatibility head.
            # Removing it would make strict state-dict loading fail.
            self.fc_finetune = nn.Linear(features + 3, 3)

        def _make_layer(
            self,
            block: Any,
            planes: int,
            blocks: int,
            stride: int = 1,
        ) -> Any:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                output_planes = planes * block.expansion
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        output_planes,
                        1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_planes),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
            return nn.Sequential(*layers)

        def forward(self, inputs: Any) -> tuple[Any, Any, Any]:
            features = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
            features = self.layer4(self.layer3(self.layer2(self.layer1(features))))
            features = self.avgpool(features).flatten(1)
            return (
                self.fc_yaw(features),
                self.fc_pitch(features),
                self.fc_roll(features),
            )

    return Hopenet(Bottleneck, [3, 4, 6, 3], num_bins)

import torch
from torch import nn
from torchvision import models


class Conv2D(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(out_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class DangerResNet(nn.Module):
    def __init__(self, output_classes: int, max_danger_level: int, finetune: bool = True, **kwargs):
        """Instanciate a network with a pretrained resnet18 as feature extractor.

        Args:
            output_classes: The number of classes.
            max_danger_level: The number of danger levels.
            finetune: If False, then do not require gradients for the resnet layers.
                      (Note: be careful what parameters are given to the optimizer in that case)
        """
        super().__init__()
        self.n_classes = output_classes
        self.n_danger_levels = max_danger_level

        resnet = models.resnet18(pretrained=True)
        # The layer1 expects 64 channels as input.
        resnet_backbone = nn.Sequential(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        if not finetune:
            for param in resnet_backbone.parameters():
                param.requires_grad = False
        # The first conv and maxpool are the same as the resnet, except for the input channels.
        self.backbone = nn.Sequential(Conv2D(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                                      torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      resnet_backbone)

        backbone_out_channels = resnet.layer4[1].conv2.out_channels  # 512
        self.classification_head = nn.Conv2d(backbone_out_channels, self.n_classes, 3, 1, 1)
        self.danger_head = nn.Conv2d(backbone_out_channels, self.n_danger_levels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        return self.classification_head(x), self.danger_head(x)


if __name__ == "__main__":
    def _test():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DangerResNet(10, 3)
        dummy_input = torch.empty(2, 2, 320, 512, device=device)
        # print(model.resnet_backbone)
        cls_pred, danger_pred = model(dummy_input)
        print(cls_pred.shape)
        print(danger_pred.shape)
    _test()

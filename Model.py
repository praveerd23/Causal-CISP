class ResNet18(ResNet):
    def __init__(self, num_classes=10, in_channels=3, pretrained=False):
        # build ResNet-18 using BasicBlock and layers [2,2,2,2]
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        # adjust first conv to accept CIFAR (32x32) images and variable in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # remove the original maxpool to preserve spatial dims for small images
        self.maxpool = nn.Identity()
        # replace fc so we can return features too
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def forward(self, x):
        # follow ResNet forward but return features before final FC as z
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)      # this is z
        out = self.fc(features)
        return features, out

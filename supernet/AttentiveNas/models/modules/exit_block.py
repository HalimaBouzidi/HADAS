from torch import nn

class ExitBlock(nn.Module):

    def __init__(self, in_features, num_classes, conv_features=224):
        super(ExitBlock, self).__init__()

        self.conv = nn.Conv2d(in_features, conv_features, 3, stride=1, bias=False)

        self.bn_norm = nn.BatchNorm2d(conv_features)
        
        self.act = nn.LeakyReLU()

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.confidence = nn.Sequential(
            nn.Linear(conv_features, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(conv_features , num_classes),
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn_norm(x)
        x = self.act(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        conf = self.confidence(x)
        pred = self.classifier(x)
        
        return pred, conf
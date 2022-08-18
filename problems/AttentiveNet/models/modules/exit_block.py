from torch import nn

class ExitBlock(nn.Module):

    def __init__(self, in_features, num_classes):
        super(ExitBlock, self).__init__()

        self.layers = []        
        self.layers.append(nn.AdaptiveAvgPool2d((1,1)))

        self.confidence = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features , num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)

        conf = self.confidence(x)
        pred = self.classifier(x)
        
        return pred, conf
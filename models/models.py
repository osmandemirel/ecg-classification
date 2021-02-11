from models.base_models import *

class CXRNet(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels, out_channels)
        self.flatten = nn.Flatten()
        self.feature_classifier = FeatureClassifier(in_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        out = self.feature_classifier(x)
        return out

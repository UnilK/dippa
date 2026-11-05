import torch
import torch.nn as nn

class SunLiCNN(nn.Module):
    def __init__(self, feature_width, feature_height, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(24, 48, kernel_size=5, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=5, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Determine the flattened feature size for the configured image dimensions.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, feature_height, feature_width)
            feature_size = self.features(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_size, 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
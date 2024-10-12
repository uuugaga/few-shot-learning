import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import BertModel

class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes=None, input_channels=None):
        super(EfficientNetV2S, self).__init__()
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        if input_channels != 3:
            self.model.features[0][0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.model(x)

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)  # 假設有 2 個類別

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
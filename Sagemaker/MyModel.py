from torch import nn
import torch 
from efficientnet_pytorch import EfficientNet

class Model(nn.Module):
    
    def __init__(self, drop1=.3,drop2=.1):
        super().__init__()
        
        self.effnet = EfficientNet.from_name('efficientnet-b5')
        n_features = self.effnet._fc.in_features
        self.effnet._dropout = nn.Dropout(drop1)
        self.effnet._fc = nn.Sequential(
                            nn.Linear(n_features,1000),nn.ReLU(),
                            nn.Dropout(drop2),
                            nn.Linear(1000,6)
                            )    
        
    def forward(self,x):
        x = self.effnet(x)
        return torch.sigmoid(x)
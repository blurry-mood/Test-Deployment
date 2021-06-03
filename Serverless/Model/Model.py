import torch
from pytorch_lightning import LightningModule
from torch import nn

class Model(LightningModule):
    
    def __init__(self):
        super().__init__()        
        self.model = nn.Linear(3,1)
      
        
    def forward(self,x):
        x = self.model(x)
        return torch.sigmoid(x)


model = Model()
# torch.save(model.state_dict(), './Serverless/Model/model.pth')
model.load_state_dict(torch.load('./Serverless/Model/model.pth'))

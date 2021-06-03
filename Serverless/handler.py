import json
import torch
from Model.Model import model

def hello(event, context):
    x=torch.rand(4,3)
    
    body = {
        "message": "Go Serverless v2.0! Your function executed successfully!",
        "input": event,
        "output": model(x).tolist(),
    }

    return {"statusCode": 200, "body": json.dumps(body)}

if __name__ =='__main__':
    print(hello('Test Event',''))

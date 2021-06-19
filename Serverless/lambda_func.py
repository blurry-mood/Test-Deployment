import json
import boto3
import base64

endpoint_name ='torchserve-endpoint-2021-06-06-21-56-06'
session = boto3.Session(aws_access_key_id='xxxxx', aws_secret_access_key='/dgdgdf gdgvdfg dfgdf +cK')
s3 = session.client('s3')
sage = session.client('runtime.sagemaker')
    
bucket_name = 'uploaderr'

def lambda_handler(event, context):
    
    # Read the input image
    body = json.loads(event['body'])
    im_b64 = body['image']
    payload = base64.b64decode(im_b64.encode('utf-8'))
    
    # Pass the image to the model & wait for the results
    response = sage.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
    
    # Convert list of int to list of strings
    a = [str(t) for t in json.loads(response['Body'].read()) ]
    
    return {
        'statusCode': 200,
        'body': " ".join(a)
    }



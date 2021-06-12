import json
import boto3
import base64

endpoint_name ='torchserve-endpoint-2021-06-06-21-56-06'
session = boto3.Session(aws_access_key_id='xxxxx', aws_secret_access_key='/dgdgdf gdgvdfg dfgdf +cK')
s3 = session.client('s3')
sage = session.client('runtime.sagemaker')

def s3_read(bucket, key):
    """
    Read a file from an S3 source.

    Parameters
    ----------
    source : str
        Path starting with s3://, e.g. 's3://bucket-name/key/foo.bar'
    profile_name : str, optional
        AWS profile

    Returns
    -------
    content : bytes

    botocore.exceptions.NoCredentialsError
        Botocore is not able to find your credentials. Either specify
        profile_name or add the environment variables AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN.
        See https://boto3.readthedocs.io/en/latest/guide/configuration.html
    """
    s3_object = s3.get_object(Bucket=bucket_name, Key=key)
    body = s3_object['Body']
    return body.read()
    
bucket_name = 'uploaderr'

def lambda_handler(event, context):
    
    body = json.loads(event['body'])
    im_b64 = body['image']
    payload = base64.b64decode(im_b64.encode('utf-8'))
    
    response = sage.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
        
    a = [str(t) for t in json.loads(response['Body'].read()) ]
    
    return {
        'statusCode': 200,
        'body': " ".join(a)
    }



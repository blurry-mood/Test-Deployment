# Test-Deployment

### Steps:
> * npm install -g serverless
> * serverless
> * cd {folder}
> * \# change the serverless.yml according to your requirements
> * serverless deploy -v
> * sls deploy


### To test:
> * curl -X POST https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev/{function name}
> * serverless invoke -f hello -d '{"body": "not a json string"}' 
> * serverless logs -f hello -t


### Remove service:
> * serverless remove

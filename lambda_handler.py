# lambda_handler.py
import os
import json
import boto3
import time
from botocore.config import Config

sagemaker_runtime = boto3.client("sagemaker-runtime", config=Config(retries={'max_attempts': 2}))
ENDPOINT_NAME = os.environ.get("SM_ENDPOINT", "medical-rag-endpoint")

def call_sagemaker(payload: dict, content_type="application/json"):
    resp = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=content_type,
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp['Body'].read()
    # model container should return JSON
    return json.loads(body)

def lambda_handler(event, context):
    try:
        data = json.loads(event.get("body") or "{}")
        query = data.get("query")
        if not query:
            return {"statusCode":400, "body": json.dumps({"error":"query missing"})}
        payload = {"query": query, "top_k": data.get("top_k", 3)}
        start = time.time()
        result = call_sagemaker(payload)
        latency = time.time() - start
        return {"statusCode":200, "body": json.dumps({"result": result, "latency": latency})}
    except Exception as e:
        # record exception and return sanitized message
        print("Error:", str(e))
        return {"statusCode":500, "body": json.dumps({"error":"internal server error"})}

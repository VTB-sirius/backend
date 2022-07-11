import os
import boto3

from pymongo import MongoClient
from urllib.parse import quote_plus as quote

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
	aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
    user=quote(os.environ['MONGODB_USERNAME']),
    pw=quote(os.environ['MONGODB_PASSWORD']),
    hosts=','.join([
        os.environ['MONGODB_HOST']
    ]),
    rs='rs01',
    auth_src=os.environ['MONGODB_DATABASE'])

db = MongoClient(url, tlsCAFile='./assets/CA.pem')['classify']

YA_RELOAD_TIME = 60
YA_SERVICE_ACC_ID = os.getenv("YA_SERVICE_ACC_ID")
YA_SERVICE_KEY_ID = os.getenv("YA_SERVICE_KEY_ID")

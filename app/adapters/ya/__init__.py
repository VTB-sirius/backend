import logging
import threading
import time
from datetime import datetime
from time import sleep

import jwt
import requests
import json

from app.settings import YA_RELOAD_TIME, YA_SERVICE_KEY_ID, YA_SERVICE_ACC_ID


class YandexDataSphere:
    aim_token: str

    def __init__(self):
        t = threading.Thread(target=self.task)
        t.start()

    def task(self):
        while True:
            self.aim_token = self._get_token()
            # Do something
            sleep(YA_RELOAD_TIME)

    def _gen_jwt_token(self):
        with open("assets/YA_SERVICE_PRIVATE_FILE.pem", 'r') as private:
            private_key = private.read()

        payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': YA_SERVICE_ACC_ID,
            'iat': int(time.time()),
            'exp': int(time.time()) + 360
        }

        return jwt.encode(
            payload,
            private_key,
            algorithm='PS256',
            headers={'kid': YA_SERVICE_KEY_ID}
        )


    def _get_token(self) -> str:
        r = requests.post('https://iam.api.cloud.yandex.net/iam/v1/tokens', json = {
            'jwt': str(self._gen_jwt_token()),
        })

        data = r.json()

        return data.get('iamToken')

yc = YandexDataSphere()

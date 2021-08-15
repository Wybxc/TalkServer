import asyncio
import random
import time
from collections import defaultdict, deque
from fastapi.exceptions import HTTPException
from fastapi.params import Depends

import torch
from fastapi import FastAPI, Request

from beam_search import get_response
from network import AttnDecoderRNN, EncoderRNN, device

BIAO_DIAN = r'，。、；‘’“”【】（）()[]{},./\;:<>?《》？|-=——+_`~·！!@#$%^&*()￥"' + "'"

str_preprocessor = torch.load('str_preprocessor.pickle')

encoder = EncoderRNN(str_preprocessor.n_word, 256).to(device)
encoder.load_state_dict(torch.load('encoder.state', map_location=device))

decoder = AttnDecoderRNN(256, str_preprocessor.n_word).to(device)
decoder.load_state_dict(torch.load('decoder.state', map_location=device))

app = FastAPI()


class Limiter():
    def __init__(self, whitelist):
        self.whitelist = whitelist
        self._records = defaultdict(deque)
        self._clean_timer = None

    def clean(self):
        for record in self._records.values():
            while record and time.time() - record[0] > 60:
                record.popleft()
        for host in list(self._records.keys()):
            if not self._records[host]:
                del self._records[host]

    async def clean_timer(self):
        while True:
            self.clean()
            await asyncio.sleep(120)

    async def __call__(self, request: Request):
        client_host = request.client.host
        if client_host in self.whitelist:
            return

        record = self._records[client_host]
        while record and time.time() - record[0] > 60:
            record.popleft()
        if len(record) >= 10:
            raise HTTPException(status_code=403, detail="Rate limit exceeded.")
        record.append(time.time())

        if not self._clean_timer:
            self._clean_timer = asyncio.create_task(self.clean_timer())

    def __del__(self):
        if self._clean_timer:
            self._clean_timer.cancel()


@app.get('/',
         dependencies=[Depends(Limiter(whitelist=['localhost', '127.0.0.1']))])
def get_message(msg: str, request: Request):
    msg = msg[:990]
    if msg == '':
        return {'result': '', 'score': 0.}
    if not msg[-1] in BIAO_DIAN:
        msg = msg + '。'
    input_tensor = str_preprocessor.str2tensor(msg)
    results, score = get_response(input_tensor,
                                  encoder,
                                  decoder,
                                  str_preprocessor,
                                  beam_search=25,
                                  best_n=5,
                                  weight_lambda=0.65)
    result = results[random.randint(0, 4)]
    while result[-1] == result[-2] and result[-1] == result[-3] and len(
            result) >= 3:
        result = result[:-1]
    client_host = request.client.host
    print(f'[{client_host}] {msg} -> {result}')
    return {'result': result, 'score': score}

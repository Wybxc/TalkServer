import random

import torch
from fastapi import FastAPI

from beam_search import get_response
from network import AttnDecoderRNN, EncoderRNN, device

BIAO_DIAN = r'，。、；‘’“”【】（）()[]{},./\;:<>?《》？|-=——+_`~·！!@#$%^&*()￥"' + "'"

str_preprocessor = torch.load('str_preprocessor.pickle')

encoder = EncoderRNN(str_preprocessor.n_word, 256).to(device)
encoder.load_state_dict(torch.load('encoder.state', map_location=device))

decoder = AttnDecoderRNN(256, str_preprocessor.n_word).to(device)
decoder.load_state_dict(torch.load('decoder.state', map_location=device))

app = FastAPI()


@app.get('/')
def get_message(msg: str):
    msg = msg[:990]
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
    return {'result': result, 'score': score}

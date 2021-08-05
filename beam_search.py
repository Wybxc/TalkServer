import random
import torch
import torch.nn as nn
from network import device

loss_function = nn.NLLLoss()


def get_response(input_tensor, encoder, decoder, str_preprocessor, max_length=1000, beam_search=5,
                 weight_lambda=0.65, best_n=1):
    # pylint: disable=not-callable, unused-variable
    encoder_hidden, decoder_hidden = encoder.initHidden(), decoder.initHidden()

    input_length = input_tensor.size(0)
    best_n = best_n if beam_search >= best_n > 0 else beam_search

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[str_preprocessor.SOS]], device=device)

    # (decoder_input, decoder_hidden, decoder_output, history, score)
    beam = []
    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                encoder_outputs)
    topv, topi = decoder_output.topk(beam_search)
    for i in topi.squeeze().detach():
        beam.append((i, decoder_hidden, decoder_output, [i], decoder_output[0][i].item()))
    beam.sort(key=lambda x:x[-1], reverse=True)
    for di in range(1, max_length):
        possibility = []
        for b in range(beam_search):
            decoder_input, decoder_hidden, decoder_output, history, score = beam[b]
            if len(history) >= 3 and history[-1] == history[-2] and history[-1] == history[-3]:
                history = history[:-1]
                possibility.append((decoder_input, decoder_hidden, decoder_output, history, score))
                continue
            if decoder_input.item() != str_preprocessor.EOS:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
                topv, topi = decoder_output.topk(beam_search)
                for i in topi.squeeze().detach():
                    score = score * (0.9975 + random.random() * 0.005)
                    if weight_lambda != 1:
                        weight_lambda_d = weight_lambda ** (di + 1)
                        new_score = score * (weight_lambda - weight_lambda_d) + decoder_output[0][i].item() * (1 - weight_lambda)
                        new_score /= 1 - weight_lambda_d
                    else:
                        new_score = (score * di + decoder_output[0][i].item()) / (di + 1)
                    possibility.append((i, decoder_hidden, decoder_output, history[:], new_score))
            else:
                possibility.append((decoder_input, decoder_hidden, decoder_output, history, score))
        possibility.sort(key=lambda x: x[-1], reverse=True)
        for b in range(beam_search):
            decoder_input, decoder_hidden, decoder_output, history, score = possibility[b]
            history.append(decoder_input.item())
            beam[b] = (decoder_input, decoder_hidden, decoder_output, history, score)


        for b in range(beam_search):
            decoder_input, decoder_hidden, decoder_output, history, score = beam[b]
            if decoder_input.item() != str_preprocessor.EOS:
                break
        else:
            break

    results = []
    for i in range(best_n):
        decoder_input, decoder_hidden, decoder_output, history, score = beam[i]
        result = str_preprocessor.code2str(history)
        results.append(result)
    return results, beam[0][-1]

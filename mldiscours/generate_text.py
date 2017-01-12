import json
import os
import logging
import numpy as np
import progressbar

# log stuff
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def generate_txt_from_json(filename):
    logger.info('Generate Text')
    with open(filename) as input:
        raw_json = json.load(input)
    raw_text = ''
    with progressbar.ProgressBar(max_value=len(raw_json)) as bar:
        for idx, r in enumerate(raw_json):
            for item in r['discours']:
                raw_text += item
            bar.update(idx)
    return raw_text

def list_chars(text):
    chars = sorted(list(set(text)))
    return chars


def generate_text_slices(path, seqlen=40, step=3):
    text = path
    # limit the charset, encode uppercase etc
    logger.info('corpus length: %s' % len(text))
    yield len(text), text[:seqlen]

    while True:
        for i in range(0, len(text) - seqlen, step):
            sentence = text[i: i + seqlen]
            next_char = text[i + seqlen]
            yield sentence, next_char


def generate_arrays(path, chars, seqlen=40, step=3, batch_size=10):
    slices = generate_text_slices(path, seqlen, step)
    text_len, seed = next(slices)
    samples = (text_len - seqlen + step - 1)/step
    char_indices = dict((c, i) for i, c in enumerate(chars))
    yield samples, seed

    while True:
        X = np.zeros((batch_size, seqlen, len(chars)), dtype=np.bool)
        y = np.zeros((batch_size, len(chars)), dtype=np.bool)
        for i in range(batch_size):
            sentence, next_char = next(slices)
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_char]] = 1
        yield X, y




if __name__ == '__main__':
    #fucking long for a 541Mb json file but hey, political speeches are that boring...
    if not os.path.exists('/home/lotso/PycharmProjects/discourspublic/data/discours.txt'):
        raw_text = generate_txt_from_json('/home/lotso/PycharmProjects/discourspublic/scrapdiscourspublic/discours.json')
        with open('/home/lotso/PycharmProjects/discourspublic/data/discours.txt', 'w') as output:
            output.write(raw_text)
    else:
        with open('/home/lotso/PycharmProjects/discourspublic/data/discours.txt') as input:
            raw_text = input.read()
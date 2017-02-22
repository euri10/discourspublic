import sys
import os
import json
import logging

import multiprocessing
import numpy as np
import progressbar

# log stuff
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)

def process_json(text_list):
    raw_text = ''
    for item in text_list:
        raw_text += item
    return raw_text


def generate_txt_from_json(filename):
    logger.info('Generate Text')
    with open(filename) as input:
        raw_json = json.load(input)

    processes = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(processes)
    discours = ((r['discours'] for r in raw_json))
    pool_outputs = pool.map(process_json, discours)
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks
    return '\n\n'.join(pool_outputs)

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

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    # this is stupid but np.random.multinomial throws an error if the probabilities
    # sum to > 1 - which they do due to finite precision
    while sum(a) > 1:
        a /= 1.000001
    return np.argmax(np.random.multinomial(1, a, 1))

def generate(model, seed, diversity, chars):
    _, maxlen, _ = model.input_shape
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    assert len(seed) >= maxlen
    sentence = seed[len(seed)-maxlen: len(seed)]
    while True:
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        yield next_char
        sentence = sentence[1:] + next_char


def generate_and_print(model, seed, diversity, n, chars):
    sys.stdout.write('generating with seed: \n')
    sys.stdout.write(''.join(seed))
    sys.stdout.write('\n=================================\n')

    generator = generate(model, seed, diversity, chars)
    sys.stdout.write(''.join(seed))

    full_text = []
    for _ in range(n):
        next_char = next(generator)
        sys.stdout.write(next_char)
        sys.stdout.flush()
        full_text.append(next_char)

    return ''.join(full_text)

if __name__ == '__main__':
    #fucking long for a 541Mb json file but hey, political speeches are that boring...
    if not os.path.exists('/home/lotso/PycharmProjects/discourspublic/data/discours/discours_pool.txt'):
        raw_text = generate_txt_from_json('/home/lotso/PycharmProjects/discourspublic/scrapdiscourspublic/discours.json')
        with open('/home/lotso/PycharmProjects/discourspublic/data/discours_pool.txt', 'w') as output:
            output.write(raw_text)
    else:
        with open('/home/lotso/PycharmProjects/discourspublic/data/discours/discours_pool.txt') as input:
            raw_text = input.read()
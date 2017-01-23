import os

import click as click
import logging

from keras.layers import GRU, Dropout, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

from mldiscours.generate_text import list_chars, generate_and_print, \
    generate_arrays
from mldiscours.train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig()


@click.group()
@click.option('--debug/--no_debug', default=False, help='Set to true to see debug logs on top of info')
def cli(debug):
    if debug:
        logging.root.setLevel(level=logging.DEBUG)
    else:
        logging.root.setLevel(level=logging.INFO)


@click.command()
@click.option('--corpus_directory', '-cf', type=click.Path(exists=True, file_okay=False))
@click.option('--maxlen', '-l', default=40, type=click.INT)
def train(corpus_directory, maxlen):

    # getting the text
    logger.info('Getting text')
    filename = os.path.basename(corpus_directory) + '.txt'
    filepath = os.path.join(corpus_directory, filename)
    base_dir = os.path.split(os.path.split(corpus_directory)[0])[0]
    model_dir = os.path.split(os.path.split(filepath)[0])[1]

    if os.path.exists(filepath):
        with open(filepath) as input:
            raw_text = input.read()
        chars = list_chars(raw_text)
    else:
        logger.error('error getting text')

    logger.info('Building model')

    model_filename = os.path.join(base_dir, 'output', model_dir, 'best.hdf5')
    if os.path.exists(model_filename):
        if click.confirm('are you sure you want to overwrite existing model ?'):
            model = Sequential()
            model.add(GRU(512, input_shape=(maxlen, len(chars)),
                          return_sequences=True))
            model.add(Dropout(0.20))
            model.add(GRU(512, return_sequences=True))
            model.add(Dropout(0.20))
            model.add(GRU(256, return_sequences=False))
            model.add(Dropout(0.20))
            model.add(Dense(len(chars)))
            # use 20% dropout on all LSTM layers: http://arxiv.org/abs/1312.4569
            model.add(Activation('softmax'))
            optimizer = RMSprop()
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            model.metadata = {'epoch': 0, 'loss': [], 'val_loss': []}
        else:
            model = load_model(model_filename)
    else:
        model = Sequential()
        model.add(GRU(512, input_shape=(maxlen, len(chars)),
                      return_sequences=True))
        model.add(Dropout(0.20))
        model.add(GRU(512, return_sequences=True))
        model.add(Dropout(0.20))
        model.add(GRU(256, return_sequences=False))
        model.add(Dropout(0.20))
        model.add(Dense(len(chars)))
        # use 20% dropout on all LSTM layers: http://arxiv.org/abs/1312.4569
        model.add(Activation('softmax'))
        optimizer = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        model.metadata = {'epoch': 0, 'loss': [], 'val_loss': []}
    train_model(model=model, raw_text=raw_text, chars=chars, batch_size=1024)

@click.command()
@click.option('--model_directory', '-m', type=click.Path(exists=True, file_okay=False))
@click.option('--seed', '-s', type=click.STRING)
@click.option('--diversity', '-d', type=click.FLOAT, default=0.5)
@click.option('--chars_number', '-n', type=click.INT, default=10000)
def generate(model_directory, seed, diversity, chars_number):
    filename = 'best.hdf5'
    filepath = os.path.join(model_directory, filename)
    if not os.path.exists(filepath):
        logger.error('your model doesnt exist')
    else:
        logger.info('Loading model: {}'.format(model_directory))
        model = load_model(filepath)


    if seed is None:
        if click.confirm('you provided no seed, generate one automatically, if not the program will stop, you need one', abort=True):

            base_dir = os.path.split(os.path.split(model_directory)[0])[0]
            input_dir = os.path.split(os.path.split(filepath)[0])[1]
            inputpath = os.path.join(base_dir, 'data', input_dir , input_dir+'.txt')
            if os.path.exists(inputpath):
                with open(inputpath) as input:
                    raw_text = input.read()
                chars = list_chars(raw_text)
            else:
                logger.error('error getting text')

            train_gen = generate_arrays(raw_text, chars, seqlen=40, step=3, batch_size=1024)
            _, seed = next(train_gen)

        else:
            logger.info('need a seed')
    else:
        base_dir = os.path.split(os.path.split(model_directory)[0])[0]
        input_dir = os.path.split(os.path.split(filepath)[0])[1]
        inputpath = os.path.join(base_dir, 'data', input_dir,
                                 input_dir + '.txt')
        if os.path.exists(inputpath):
            with open(inputpath) as input:
                raw_text = input.read()
            chars = list_chars(raw_text)
        else:
            logger.error('error getting text')
    generate_and_print(model, seed, diversity, chars_number, chars)

cli.add_command(train)
cli.add_command(generate)

if __name__ == '__main__':
    cli()

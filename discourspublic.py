import click as click
import logging

from keras.layers import GRU, Dropout, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

from mldiscours.generate_text import list_chars
from mldiscours.train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig()

@click.command()
@click.option('--corpus_filename', '-cf', type=click.Path(exists=True, dir_okay=False))
@click.option('--model_filename', '-m', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--maxlen', '-l', default=40, type=click.INT)
@click.option('--debug/--no_debug', default=False, help='Set to true to see debug logs on top of info')
def cli(corpus_filename, model_filename, maxlen, debug):
    if debug:
        logging.root.setLevel(level=logging.DEBUG)
    else:
        logging.root.setLevel(level=logging.INFO)

    # getting the text
    logger.info('Getting text')

    with open(corpus_filename) as input:
        raw_text = input.read()
    chars = list_chars(raw_text)
    print(len(chars))

    logger.info('Building model')
    if model_filename is None:
        model = Sequential()
        model.add(GRU(512, input_shape=(maxlen, len(chars)), return_sequences=True))
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

    train_model(model=model, raw_text=raw_text, chars=chars, batch_size=1024)

if __name__ == '__main__':
    cli()

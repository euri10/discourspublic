import logging
import os
from keras.callbacks import ModelCheckpoint
from mldiscours.generate_text import generate_arrays

# log stuff
from mldiscours.utils import LivePlotHistory, DrawWeights

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def train_model(model, raw_text, chars, step=3, batch_size=1024, iters=1000):

    logger.info('Training model')
    logger.info('Splitting text')
    # shitty split I know

    train_text = raw_text[:int(0.8 * len(raw_text))]
    test_text = raw_text[int(0.2 * len(raw_text)):]

    _, maxlen, _ = model.input_shape
    train_gen = generate_arrays(train_text, chars, seqlen=maxlen, step=step, batch_size=batch_size)
    samples, seed = next(train_gen)

    logger.info('samples per epoch {:,.2f}'.format(samples))
    last_epoch = model.metadata.get('epoch', 0)

    # leading / !!!
    filebase = os.path.dirname('/home/lotso/PycharmProjects/discourspublic/output/')

    # removed the iter loop with 1 on epoch
    # for epoch in range(last_epoch + 1, last_epoch + iters + 1):
    val_gen = generate_arrays(test_text, chars, seqlen=maxlen, step=step, batch_size=batch_size)
    val_samples, _ = next(val_gen)

    #define callbacks
    # filepath = os.path.join(filebase , "epoch{epoch:05d}-{loss:.4f}.hdf5")
    filepath = os.path.join(filebase, "best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # history = LivePlotHistory()
    drawweights = DrawWeights(figsize=(4, 4), layer_id=0, param_id=0)
    # earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=0, mode='auto')

    callbacks_list = [checkpoint, drawweights]

    hist = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=val_samples, samples_per_epoch=samples, nb_epoch=iters, verbose=1, callbacks=callbacks_list)
    # hist = model.fit_generator(train_gen, samples_per_epoch=samples, nb_epoch=10000, verbose=1, callbacks=callbacks_list)
    # hist = model.fit(train_gen, samples_per_epoch=samples, nb_epoch=10000, verbose=1, callbacks=callbacks_list)
import os
import h5py
import tempfile
import numpy as np
from keras import backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, Sequential

from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.models import save_model, load_model, save_mxnet_model


model = Sequential()
model.add(Dense(2, input_shape=(3,)))
model.add(RepeatVector(3))
model.add(TimeDistributed(Dense(3)))
model.compile(loss=losses.MSE,
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=[metrics.categorical_accuracy],
              sample_weight_mode='temporal')
x = np.random.random((1, 3))
y = np.random.random((1, 3, 3))
model.train_on_batch(x, y)

out = model.predict(x)
print("First output - ", out)
_, fname = tempfile.mkstemp('.h5')
save_model(model, fname)

new_model = load_model(fname, context="eia")
print("Loaded new model - ", new_model)
print("Context is - ", new_model._context)
os.remove(fname)

out2 = new_model.predict(x)
print("Second output - ", out2)
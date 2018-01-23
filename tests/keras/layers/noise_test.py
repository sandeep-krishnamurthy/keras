import pytest
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras.layers import noise
from keras import backend as K


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk' or K.backend() == 'mxnet'),
                    reason="cntk/mxnet does not support it yet")
def test_GaussianNoise():
    layer_test(noise.GaussianNoise,
               kwargs={'stddev': 1.},
               input_shape=(3, 2, 3))


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk' or K.backend() == 'mxnet'),
                    reason="cntk/mxnet does not support it yet")
def test_GaussianDropout():
    layer_test(noise.GaussianDropout,
               kwargs={'rate': 0.5},
               input_shape=(3, 2, 3))


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk' or K.backend() == 'mxnet'),
                    reason="cntk/mxnet does not support it yet")
def test_AlphaDropout():
    layer_test(noise.AlphaDropout,
               kwargs={'rate': 0.1},
               input_shape=(3, 2, 3))


if __name__ == '__main__':
    pytest.main([__file__])

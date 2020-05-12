# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for testing Addons."""

import os
import random
import threading

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.utils import resource_loader
from tensorflow import keras

from tensorflow.python.framework import (
    test_util,
    tensor_shape,
    tensor_spec,
)  # noqa: F401
from tensorflow.python.util import tf_inspect  # noqa: F401
from tensorflow.python.eager import context  # noqa F401

NUMBER_OF_WORKERS = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
WORKER_ID = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0")[2])
NUMBER_OF_GPUS = len(tf.config.list_physical_devices("GPU"))

# Some configuration before starting the tests.

# we only need one core per worker.
# This avoids context switching for speed, but it also prevents TensorFlow to go
# crazy on systems with many cores (kokoro has 30+ cores).
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

if NUMBER_OF_GPUS != 0:
    # We use only the first gpu at the moment. That's enough for most use cases.
    # split the first gpu into chunks of 100MB per pytest worker.
    # It's the user's job to limit the amount of pytest workers depending
    # on the available memory.
    # In practice, each process takes a bit more memory.
    # There must be some kind of overhead but it's not very big (~200MB more)
    first_gpu = tf.config.list_physical_devices("GPU")[0]

    tf.config.set_logical_device_configuration(
        first_gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=100)],
    )


def finalizer():
    tf.config.experimental_run_functions_eagerly(False)


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, tf.DType):
        return val.name
    if val is False:
        return "no_" + argname
    if val is True:
        return argname


@pytest.fixture(scope="function", params=["eager_mode", "tf_function"])
def maybe_run_functions_eagerly(request):
    if request.param == "eager_mode":
        tf.config.experimental_run_functions_eagerly(True)
    elif request.param == "tf_function":
        tf.config.experimental_run_functions_eagerly(False)

    request.addfinalizer(finalizer)


@pytest.fixture(scope="function", params=["channels_first", "channels_last"])
def data_format(request):
    return request.param


@pytest.fixture(scope="function", autouse=True)
def set_seeds():
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--skip-custom-ops",
        action="store_true",
        help="When a custom op is being loaded in a test, skip this test.",
    )


@pytest.fixture(scope="session", autouse=True)
def set_global_variables(request):
    if request.config.getoption("--skip-custom-ops"):
        resource_loader.SKIP_CUSTOM_OPS = True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "with_device(devices): mark test to run on specific devices."
    )
    config.addinivalue_line("markers", "needs_gpu: mark test that needs a gpu.")


@pytest.fixture(autouse=True, scope="function")
def _device_placement(request):
    device = request.param
    if device == "no_device":
        yield
    else:
        if device in ["cpu", "gpu"]:
            # we use GPU:0 because the virtual device we created is the
            # only one in the first GPU (so first in the list of virtual devices).
            device += ":0"
        else:
            raise KeyError("Invalid device: " + device)
        with tf.device(device):
            yield


def get_marks(device_name):
    marks = []
    if device_name == "gpu":
        marks.append(pytest.mark.needs_gpu)
        if NUMBER_OF_GPUS == 0:
            skip_message = "The gpu is not available."
            marks.append(pytest.mark.skip(reason=skip_message))
    return marks


def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("with_device")
    if marker is None:
        # tests which don't have the "with_device" mark are executed on CPU
        # to ensure reproducibility. We can't let TensorFlow decide
        # where to place the ops.
        devices = ["cpu"]
    else:
        devices = marker.args[0]

    parameters = [pytest.param(x, marks=get_marks(x)) for x in devices]
    metafunc.parametrize("_device_placement", parameters, indirect=True)


def assert_allclose_according_to_type(
    a,
    b,
    rtol=1e-6,
    atol=1e-6,
    float_rtol=1e-6,
    float_atol=1e-6,
    half_rtol=1e-3,
    half_atol=1e-3,
    bfloat16_rtol=1e-2,
    bfloat16_atol=1e-2,
):
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    a = np.array(a)
    b = np.array(b)
    # types with lower tol are put later to overwrite previous ones.
    if (
        a.dtype == np.float32
        or b.dtype == np.float32
        or a.dtype == np.complex64
        or b.dtype == np.complex64
    ):
        rtol = max(rtol, float_rtol)
        atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
        rtol = max(rtol, half_rtol)
        atol = max(atol, half_atol)
    if a.dtype == tf.bfloat16.as_numpy_dtype or b.dtype == tf.bfloat16.as_numpy_dtype:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.experimental_run_tf_function = None


@test_util.use_deterministic_cudnn
def layer_test(
    layer_cls,
    kwargs=None,
    input_shape=None,
    input_dtype=None,
    input_data=None,
    expected_output=None,
    expected_output_dtype=None,
    expected_output_shape=None,
    validate_training=True,
    adapt_data=None,
):
    """Test routine for a layer with a single input and single output.

    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
      expected_output: Numpy array of the expected output.
      expected_output_dtype: Data type expected for the output.
      expected_output_shape: Shape tuple for the expected shape of the output.
      validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
      adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.

    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.

    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        if not input_dtype:
            input_dtype = "float32"
        if isinstance(input_shape, list):
            multi_input = True
        else:
            multi_input = False
            input_shape = [list(input_shape)]
        input_data_shape = [list(x_shape) for x_shape in input_shape]
        input_data = []
        for x_shape in input_data_shape:
            for i, e in enumerate(x_shape):
                if e is None:
                    x_shape[i] = np.random.randint(1, 4)
            x_data = 10 * np.random.random(x_shape)
            if input_dtype[:5] == "float":
                x_data -= 0.5
            x_data = x_data.astype(input_dtype)
            input_data.append(x_data)
    elif input_shape is None:
        input_shape = []
        if isinstance(input_data, list):
            multi_input = True
            for i in range(len(input_data)):
                input_shape.append(input_data[i].shape)
        else:
            multi_input = False
            input_shape.append(input_data.shape)
    if input_dtype is None:
        if multi_input:
            input_dtype = input_data[0].dtype
        else:
            input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if "weights" in tf_inspect.getargspec(layer_cls.__init__):
        kwargs["weights"] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    if multi_input:
        x = []
        for input_layer_shape in input_shape:
            x.append(keras.layers.Input(shape=input_layer_shape[1:], dtype=input_dtype))
    else:
        x = keras.layers.Input(shape=input_shape[0][1:], dtype=input_dtype)

    y = layer(x)
    if isinstance(y, list):
        multi_output = True
        y_type = keras.backend.dtype(y[0])
    else:
        multi_output = False
        y_type = keras.backend.dtype(y)

    if y_type != expected_output_dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output "
            "dtype=%s but expected to find %s.\nFull kwargs: %s"
            % (layer_cls.__name__, x, y_type, expected_output_dtype, kwargs)
        )

    def assert_shapes_equal(expected, actual):
        """Asserts that the output shape from the layer matches the actual shape."""
        if len(expected) != len(actual):
            raise AssertionError(
                "When testing layer %s, for input %s, found output_shape="
                "%s but expected to find %s.\nFull kwargs: %s"
                % (layer_cls.__name__, x, actual, expected, kwargs)
            )

        for expected_dim, actual_dim in zip(expected, actual):
            if isinstance(expected_dim, tensor_shape.Dimension):
                expected_dim = expected_dim.value
            if isinstance(actual_dim, tensor_shape.Dimension):
                actual_dim = actual_dim.value
            if expected_dim is not None and expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s, for input %s, found output_shape="
                    "%s but expected to find %s.\nFull kwargs: %s"
                    % (layer_cls.__name__, x, actual, expected, kwargs)
                )

    if expected_output_shape is not None:
        if multi_output:
            if len(y) != len(expected_output_shape):
                raise AssertionError(
                    "When testing layer %s, for input %s, found output length"
                    "equal to %s but expected length %s.\nFull kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        len(y),
                        len(expected_output_shape),
                        kwargs,
                    )
                )

            for i in range(len(y)):
                assert_shapes_equal(
                    tensor_shape.TensorShape(expected_output_shape[i]), y[i].shape
                )
        else:
            assert_shapes_equal(
                tensor_shape.TensorShape(expected_output_shape), y.shape
            )

    # check shape inference
    model = keras.models.Model(x, y)
    if multi_input:
        input_tensor_shape = [
            tensor_shape.TensorShape(x_shape) for x_shape in input_shape
        ]
    else:
        input_tensor_shape = tensor_shape.TensorShape(input_shape[0])
    if multi_output:
        computed_output_shapes = layer.compute_output_shape(input_tensor_shape)
        computed_output_signatures = layer.compute_output_signature(
            tensor_spec.TensorSpec(shape=input_shape[0], dtype=input_dtype)
        )
        actual_output = model.predict(input_data)
        for i in range(len(actual_output)):
            assert_shapes_equal(
                tuple(computed_output_shapes[i].as_list()), actual_output[i].shape
            )
            assert_shapes_equal(
                computed_output_signatures[i].shape, actual_output[i].shape
            )
        signature_dtype = computed_output_signatures[0].dtype
        actual_dtype = actual_output[0].dtype
    else:
        computed_output_shape = tuple(
            layer.compute_output_shape(input_tensor_shape).as_list()
        )
        computed_output_signature = layer.compute_output_signature(
            tensor_spec.TensorSpec(shape=input_shape[0], dtype=input_dtype)
        )
        actual_output = model.predict(input_data)
        actual_output_shape = actual_output.shape
        assert_shapes_equal(computed_output_shape, actual_output_shape)
        assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
        signature_dtype = computed_output_signature.dtype
        actual_dtype = actual_output.dtype

    if signature_dtype != actual_dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output_dtype="
            "%s but expected to find %s.\nFull kwargs: %s"
            % (layer_cls.__name__, x, actual_dtype, signature_dtype, kwargs)
        )
    if expected_output is not None:
        if multi_output:
            for i in range(len(actual_output)):
                np.testing.assert_allclose(
                    actual_output[i], expected_output[i], rtol=1e-3, atol=1e-6
                )
        else:
            np.testing.assert_allclose(
                actual_output, expected_output, rtol=1e-3, atol=1e-6
            )

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = keras.models.Model.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        if multi_output:
            for i in range(len(output)):
                np.testing.assert_allclose(
                    output[i], actual_output[i], rtol=1e-3, atol=1e-6
                )

        else:
            np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    if validate_training:
        model = keras.models.Model(x, layer(x))
        if _thread_local_data.run_eagerly is not None:
            model.compile(
                "rmsprop",
                "mse",
                weighted_metrics=["acc"],
                run_eagerly=should_run_eagerly(),
            )
        else:
            model.compile("rmsprop", "mse", weighted_metrics=["acc"])
        model.train_on_batch(input_data, actual_output)

    if multi_input or multi_output:
        # Sequential API doesn't support multi-input nor multi-output
        return actual_output

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config["batch_input_shape"] = input_shape[0]
    layer = layer.__class__.from_config(layer_config)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    model = keras.models.Sequential()
    model.add(layer)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(computed_output_shape, actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s **after deserialization**, "
                    "for input %s, found output_shape="
                    "%s but expected to find inferred shape %s.\nFull kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        actual_output_shape,
                        computed_output_shape,
                        kwargs,
                    )
                )
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-6)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = keras.models.Sequential.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

    # for further checks in the caller function
    return actual_output


def should_run_eagerly():
    """Returns whether the models we are testing should be run eagerly."""
    if _thread_local_data.run_eagerly is None:
        raise ValueError(
            "Cannot call `should_run_eagerly()` outside of a "
            "`run_eagerly_scope()` or `run_all_keras_modes` "
            "decorator."
        )

    return _thread_local_data.run_eagerly and context.executing_eagerly()

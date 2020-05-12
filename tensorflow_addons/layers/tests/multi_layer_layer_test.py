import numpy as np

from tensorflow_addons.layers.octave_convolutional import OctaveConv1D
from tensorflow_addons.utils import test_utils


# Single input, multi output
def test_octave_conv1d():
    valid_input = np.ones((1, 10, 2)).astype(np.float32)
    output_1_shape = (None, 10, 2)
    output_2_shape = (None, 5, 1)
    output_shape = [output_1_shape, output_2_shape]
    test_utils.layer_test(
        OctaveConv1D,
        kwargs={
            "filters": 3,
            "kernel_size": 3,
            "low_freq_ratio": 0.5,
            "padding": "same",
        },
        input_data=valid_input,
        expected_output_shape=output_shape,
    )


def test_octave_conv1d_shape():
    valid_input_shape = (1, 10, 2)
    output_1_shape = (None, 10, 2)
    output_2_shape = (None, 5, 1)
    output_shape = [output_1_shape, output_2_shape]
    test_utils.layer_test(
        OctaveConv1D,
        kwargs={
            "filters": 3,
            "kernel_size": 3,
            "low_freq_ratio": 0.5,
            "padding": "same",
        },
        input_shape=valid_input_shape,
        expected_output_shape=output_shape,
    )


def test_octave_output():
    valid_input = np.ones((1, 2, 2)).astype(np.float32)
    expected_output_1 = np.array([[4.0, 4.0], [4.0, 4.0]]).astype(np.float32)
    expected_output_1 = np.reshape(expected_output_1, (1, 2, 2))
    expected_output_2 = np.array([[2.0]]).astype(np.float32)
    expected_output_2 = np.reshape(expected_output_2, (1, 1, 1))
    expected_outputs = [expected_output_1, expected_output_2]
    test_utils.layer_test(
        OctaveConv1D,
        kwargs={
            "filters": 3,
            "kernel_size": 3,
            "low_freq_ratio": 0.5,
            "kernel_initializer": "ones",
        },
        input_data=valid_input,
        expected_output=expected_outputs,
    )


# Multi input, multi output
def test_octave_conv1d_mimo():
    input_1 = np.ones((1, 10, 2)).astype(np.float32)
    input_2 = np.ones((1, 5, 1)).astype(np.float32)
    inputs = [input_1, input_2]
    output_1_shape = (None, 10, 2)
    output_2_shape = (None, 5, 1)
    output_shape = [output_1_shape, output_2_shape]
    test_utils.layer_test(
        OctaveConv1D,
        kwargs={
            "filters": 3,
            "kernel_size": 3,
            "low_freq_ratio": 0.5,
            "padding": "same",
        },
        input_data=inputs,
        expected_output_shape=output_shape,
    )


# Multi input, single output
def test_octave_conv1d_miso():
    input_1 = np.ones((1, 10, 2)).astype(np.float32)
    input_2 = np.ones((1, 5, 1)).astype(np.float32)
    inputs = [input_1, input_2]
    output_shape = (None, 10, 3)
    test_utils.layer_test(
        OctaveConv1D,
        kwargs={
            "filters": 3,
            "kernel_size": 3,
            "low_freq_ratio": 0.0,
            "padding": "same",
        },
        input_data=inputs,
        expected_output_shape=output_shape,
    )

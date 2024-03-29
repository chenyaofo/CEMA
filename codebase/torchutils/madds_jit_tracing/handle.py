'''
Code is from https://github.com/facebookresearch/fvcore

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

The full text of the license is avaiable on 
https://raw.githubusercontent.com/facebookresearch/fvcore/master/LICENSE
'''


import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union

from numpy import prod


Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def get_shape(val: Any) -> Optional[List[int]]:
    """
    Get the shapes from a jit value object.
    Args:
        val (torch._C.Value): jit value object.
    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None


"""
Below are flop/activation counters for various ops. Every counter has the following signature:
Args:
    inputs (list(torch._C.Value)): The inputs of the op in the form of a list of jit object.
    outputs (list(torch._C.Value)): The outputs of the op in the form of a list of jit object.
Returns:
    number: The number of flops/activations for the operation.
    or Counter[str]
"""


def generic_activation_jit(op_name: Optional[str] = None) -> Handle:
    """
    This method return a handle that counts the number of activation from the
    output shape for the specified operation.
    Args:
        op_name (str): The name of the operation. If given, the handle will
            return a counter using this name.
    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(
        i: Any, outputs: List[Any]
    ) -> Union[typing.Counter[str], Number]:
        """
        This is a generic jit handle that counts the number of activations for any
        operation given the output shape.
        """
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)
        if op_name is None:
            return ac_count
        else:
            return Counter({op_name: ac_count})

    return _generic_activation_jit


def addmm_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # Count madds for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    madds = batch_size * input_dim * output_dim
    return madds


def linear_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # Inputs is a list of length 3; unlike aten::addmm, it is the first
    # two elements that are relevant.
    input_shapes = [get_shape(v) for v in inputs[0:2]]
    # input_shapes[0]: [dim0, dim1, ..., input_feature_dim]
    # input_shapes[1]: [output_feature_dim, input_feature_dim]
    assert input_shapes[0][-1] == input_shapes[1][-1]
    madds = prod(input_shapes[0]) * input_shapes[1][0]
    return madds


def bmm_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    madds = n * c * t * d
    return madds


def conv_madds_count(x_shape: List[int], w_shape: List[int], out_shape: List[int]) -> Number:
    """
    Count madds for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        int: the number of madds
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = prod(out_shape[2:])
    kernel_size = prod(w_shape[2:])
    madds = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    return madds


def conv_madds_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    Count madds for convolution.
    """
    # Inputs of Convolution should be a list of length 12 or 13. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # starting with #40737 it will be 12) user_enabled_tf32
    assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))

    # use a custom name instead of "_convolution"
    return Counter({"conv": conv_madds_count(x_shape, w_shape, out_shape)})


def einsum_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count madds for the einsum operation. We currently support
    two einsum operations: "nct,ncp->ntp" and "ntg,ncg->nct".
    """
    # Inputs of einsum should be a list of length 2.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) == 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        madds = n * c * t * p
        return madds

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        madds = n * t * g * c
        return madds

    else:
        raise NotImplementedError("Unsupported einsum operation.")


def matmul_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    madds = prod(input_shapes[0]) * input_shapes[-1][-1]
    return madds


def norm_madds_counter(affine_arg_index: int) -> Handle:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_madds_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        has_affine = get_shape(inputs[affine_arg_index]) is not None
        assert 2 <= len(input_shape) <= 5, input_shape
        # 5 is just a rough estimate
        flop = prod(input_shape) * (5 if has_affine else 4)
        return flop

    return norm_madds_jit


def elementwise_madds_counter(input_scale: float = 1, output_scale: float = 0) -> Handle:
    """
    Count flops by
        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale
    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def elementwise_madds(inputs: List[Any], outputs: List[Any]) -> Number:
        ret = 0
        if input_scale != 0:
            shape = get_shape(inputs[0])
            ret += input_scale * prod(shape)
        if output_scale != 0:
            shape = get_shape(outputs[0])
            ret += output_scale * prod(shape)
        return ret

    return elementwise_madds

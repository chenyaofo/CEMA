from torch.autograd.function import Function


class GEFunction(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        output = input.ge(threshold).float()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors
        return grad_output * output, None


class LTFunction(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        output = input.lt(threshold).float()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors
        return grad_output * -output, None


def autograd_ge(input, threshold):
    return GEFunction.apply(input, threshold)


def autograd_lt(input, threshold):
    return LTFunction.apply(input, threshold)
def calc_padding(kernel_size, dilation):
    return int((kernel_size - 1) * dilation / 2)

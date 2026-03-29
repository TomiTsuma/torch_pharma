def inflate_batch_array(array, target):
    target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
    return array.view(target_shape)
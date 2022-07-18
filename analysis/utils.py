

def unnormalize_tensor(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    :param x: image to un-normalize Expected to be of shape (3, H, W)
    :param mean: a list of size 3, with mean of each channel of the image
    :param std: a list of size 3, the std of each channel of the image
    :return:
    """

    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)

    return x

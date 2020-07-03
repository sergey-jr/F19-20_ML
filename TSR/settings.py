"""
Default settings for project
"""
import cv2

n_classes = 43
channels = 3
img_size = (30, 30)  # image output size
to_crop = True  # crop image by borders?
to_shrink = True  # shrink dataSet?
shrink_size = 750 if to_shrink else None  # shrinkage_size
to_augment = True  # apply augmentation?
to_normalize = True  # apply normalization
"""
parameters for augmentation
"""
elastic_params = {
    "alpha_affine": 15,
    "interpolation": cv2.INTER_CUBIC,
    "p": 1
}
gamma_params = {
    "gamma_limit": (80, 200),
    "p": 1
}
r_shift_params = {
    "r_shift_limit": 40,
    "g_shift_limit": 40,
    "b_shift_limit": 40,
    "p": 1
}
rotate_params = {
    "limit": 10,
    "p": 1
}
brightness_params = {
    "limit": (0.1, 0.25),
    "p": 1
}
all_other = {
    "p": 1
}

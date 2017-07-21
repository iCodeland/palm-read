IMG_WIDTH, IMG_HEIGHT = 256, 256

FLOW_PARAMS = {
    'batch_size': 16,
}

TUNED_PARAMS = {
    'epochs': 10,
    'steps_per_epoch': 100,
    'validation_steps': 80,
}

IMG_AUG = {
    'rescale': 1. / 255,
    'featurewise_center': False,
    'samplewise_center': False,
    'featurewise_std_normalization': False,
    'samplewise_std_normalization': False,
    'zca_whitening': False,
    'rotation_range': 180.,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.,
    'zoom_range': 0.2,
    'channel_shift_range': 0.,
    'fill_mode': 'nearest',
    'cval': 0.,
    'horizontal_flip': True,
    'vertical_flip': True,
}

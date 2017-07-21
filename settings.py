IMG_WIDTH, IMG_HEIGHT = 512, 512

TUNED_PARAMS = {
    'epochs': 10,
    'batch_size': 16,
    'nb_train_samples': 2000,
    'nb_validation_samples': 800,
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

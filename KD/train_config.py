"""
配置文件
"""

# data_config
DATA_CONFIG = {
    'batch_size': 256,
    'data_shape': (28, 28, 1),
}

# train strategy
TRAIN_CONFIG = {
    'lr_schedule': {
        'initial_lr': 1e-3,
        'decay_step': 2000,
        'alpha': 0.1
        # teacher alpha 0.01
        # student alpha 0.1
    },
    'epoch': 50,
}


import os, glob, math, random, shutil

import yaml
import tensorboard as tb
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.legacy import *

from model.loss import *
from utils.utils import *
from model.model_fast import *
from model.dataloader import DataLoader
from model.eval import CTCAccuracyCallback


# load yaml
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpus']

# set random seed
seed = cfg['seed']
random.seed = seed
np.random.seed = seed
tf.random.set_seed(seed)

#############################################
img_shape = (*cfg['img_shape'],)
mode = cfg['mode']
epochs = cfg['epochs']
#############################################
strategy = tf.distribute.MirroredStrategy()
print('GPUs: {}'.format(strategy.num_replicas_in_sync))

if mode == 'ctc':
    warmup = 0
    batch_size = 32 * strategy.num_replicas_in_sync
    learning_rate = 5e-3
    opt = 'nadam'
    weight_path = ""

if mode == 'label':
    warmup = cfg['warmup']
    batch_size = cfg['batch_size'] * strategy.num_replicas_in_sync
    learning_rate = cfg['learning_rate']
    opt = cfg['optimizer']['name']
    weight_path = cfg['weight_path']

    if weight_path == '':
        weight_path = glob.glob(os.path.join('checkpoints', '*.h5'))
        weight_path.sort(key=lambda x: os.path.getmtime(x))
        weight_path = weight_path[-1]
        print('load checkpoint:', weight_path)

# load from txt
num_class = len(load_dict("data/label.names"))
print('num_class:', num_class)

# set data path
train_path = "/Users/haoyu/Downloads/lpr/train"
val_path = "/Users/haoyu/Downloads/lpr/val"
# set dataloader
train_loader = DataLoader(
    train_path,
    mode=mode,
    img_shape=img_shape,
    time_steps=cfg['time_steps'],
    label_len=cfg['label_len'],
    num_class=num_class,
    batch_size=cfg['batch_size'],
    data_augmentation=True,
    shuffle=True,
)
val_loader = DataLoader(
    val_path,
    mode=mode,
    img_shape=img_shape,
    time_steps=cfg['time_steps'],
    label_len=cfg['label_len'],
    num_class=num_class,
    batch_size=1024,
    shuffle=False,
    data_augmentation=False,
)

# clear log
log_path = os.path.join('logs')
if os.path.exists(log_path): shutil.rmtree(log_path)
os.makedirs(log_path, exist_ok=True)

tb = TensorBoard(
    log_dir=log_path,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None,
)

# learning rate schedule
def lr_schedule(epoch, lr):
    warmup_epochs = warmup
    max_lr = learning_rate
    warmup_lr = 1e-6
    if epoch < warmup_epochs and warmup_epochs > 0:
        lr = (max_lr - warmup_lr) / warmup_epochs * epoch + warmup_lr
    else:
        lr = max_lr * 0.5 * (1 + math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi))
    return lr


lr_callback = LearningRateScheduler(lr_schedule)

# CTC accuracy and license plate accuracy
ctc_acc_callback = CTCAccuracyCallback(
    validation_data=val_loader,
    log_dir=log_path,
    blank_index=num_class,
    name='ctc_acc',
)

if opt == 'sgd':
    optimizer = SGD(
        learning_rate=learning_rate,
        momentum=cfg['optimizer']['sgd']['momentum'],
        nesterov=cfg['optimizer']['sgd']['nesterov'],
    )
else:
    optimizer = Nadam(learning_rate=learning_rate)

print('optimizer: {}'.format(optimizer.__class__.__name__))


with strategy.scope():
    model = TinyLPR(
        time_steps=cfg['time_steps'],
        n_class=num_class+1,
        n_feat=cfg['feat_dims'],
        width_multiplier=cfg['width_multiplier'],
        train=True,
    ).build(img_shape)
    model.summary()

    # load checkpoint
    if weight_path is not None and os.path.exists(weight_path):
        print('load checkpoint:', weight_path)
        model.load_weights(weight_path, by_name=True, skip_mismatch=True)

    # compile model
    model.compile(
        optimizer=optimizer,
        loss={
            'mask': DiceBCELoss(),
            # 'mat_ctc': CTCCenterLoss(n_class=num_class+1, feat_dims=cfg['feat_dims']),
            'ctc': FocalCTCLoss(
                alpha=cfg['focal_ctc_loss']['alpha'],
                gamma=cfg['focal_ctc_loss']['gamma'],
            ),
        },
        loss_weights={
            'mask': cfg['loss_weights']['dice_bce_loss'],
            # 'mat_ctc': cfg['loss_weights']['ctc_center_loss'],
            'ctc': cfg['loss_weights']['focal_ctc_loss'],
        },
    )


if __name__ == '__main__':
    # train
    model.fit(
        train_loader,
        epochs=cfg['epochs'],
        callbacks=[lr_callback, tb, ctc_acc_callback],
        # verbose=2,
        # workers=128,
        # use_multiprocessing=True,
        shuffle=True,
        batch_size=cfg['batch_size'],
    )

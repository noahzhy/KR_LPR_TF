import os, glob, math, random, shutil

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


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set random seed
seed = 2023
random.seed = seed
np.random.seed = seed
tf.random.set_seed(seed)

strategy = tf.distribute.MirroredStrategy()
print('GPUs: {}'.format(strategy.num_replicas_in_sync))

# ### TPU
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)
# print('TPUs: {}'.format(strategy.num_replicas_in_sync))

#############################################
time_steps = 16
label_len = 8
feat_dims = 64
width_multiplier = 0.25
img_shape = (64, 128, 1)

epochs = 100
decay = 1e-4
weight_decay = 5e-4

# mode = 'ctc'
mode = 'label'

if mode == 'ctc':
    warmup = 0
    batch_size = 32 * strategy.num_replicas_in_sync
    learning_rate = 5e-3
    opt = 'nadam'
    weight_path = ""

if mode == 'label':
    warmup = 5
    batch_size = 8 * strategy.num_replicas_in_sync
    learning_rate = 1e-4
    opt = 'nadam'
    # opt = 'sgd'
    weight_path = 'checkpoints/ctc_0.9915_char_0.9989.h5'

    if weight_path == '':
        weight_path = glob.glob(os.path.join('checkpoints', '*.h5'))
        weight_path.sort(key=lambda x: os.path.getmtime(x))
        weight_path = weight_path[-1]
        print('load checkpoint:', weight_path)

# load from txt
dict_path = os.path.join('data', 'label.names')
num_class = len(load_dict(dict_path))
print('num_class:', num_class, 'len_label:', load_dict(dict_path))

# set data path
train_path = "/Users/haoyu/Downloads/lpr/train"
val_path = "/Users/haoyu/Downloads/lpr/val"
# set dataloader
train_loader = DataLoader(
    train_path,
    mode=mode,
    img_shape=img_shape,
    time_steps=time_steps,
    label_len=label_len,
    num_class=num_class,
    batch_size=batch_size,
    data_augmentation=True,
    shuffle=True,
)
val_loader = DataLoader(
    val_path,
    mode=mode,
    img_shape=img_shape,
    time_steps=time_steps,
    label_len=label_len,
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

# Define your learning rate schedule
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
        momentum=0.95,
        nesterov=True,
    )
else:
    optimizer = Nadam(learning_rate=learning_rate)

print('optimizer: {}'.format(optimizer.__class__.__name__))


with strategy.scope():
    model = TinyLPR(
        time_steps=time_steps,
        n_class=num_class+1,
        n_feat=feat_dims,
        width_multiplier=width_multiplier,
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
            # 'mat_ctc': CTCCenterLoss(n_class=num_class+1, feat_dims=feat_dims),
            'ctc': FocalCTCLoss(alpha=0.8, gamma=3.0),
        },
        loss_weights={
            'mask': 0.5,
            # 'mat_ctc': 0.01,
            'ctc': 1.5,
        },
    )


if __name__ == '__main__':
    # train
    model.fit(
        train_loader,
        epochs=epochs,
        callbacks=[lr_callback, tb, ctc_acc_callback],
        # verbose=2,
        # workers=128,
        # use_multiprocessing=True,
        shuffle=True,
        batch_size=batch_size,
    )

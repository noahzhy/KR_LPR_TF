import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

from KD.distilling import *
from model.loss import *
from utils.utils import *
from model.model_fast import *
from model.dataloader import DataLoader


# Define your learning rate schedule
def lr_schedule(epoch, lr):
    warmup_epochs = 5
    max_lr = learning_rate
    warmup_lr = 1e-6
    if epoch < warmup_epochs and warmup_epochs > 0:
        lr = (max_lr - warmup_lr) / warmup_epochs * epoch + warmup_lr
    else:
        lr = max_lr * 0.5 * (1 + math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi))
    return lr


def distilling_train(t_model, s_model, train_data, test_data, t_weight, epoch=50):
    # teacher_model = t_model.load_weights(t_weight)
    # get teacher model's output without softmax
    teacher_model = Model(t_model.inputs, t_model.outputs[:-1])
    teacher_model.load_weights(t_weight, by_name=True, skip_mismatch=True)
    # freeze teacher model
    teacher_model.trainable = False
    # distilling model
    dist = Distilling(student_model=s_model, teacher_model=teacher_model)
    dist.compile(
        optimizer=Nadam(learning_rate=learning_rate, decay=decay),
        ctc_loss=FocalCTCLoss(alpha=0.8, gamma=3.0),
        seg_loss=DiceBCELoss(),
        kd_loss=KLDivergence(),
        T=2,
        alpha=0.9,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy('acc')
        ]
    )
    lr_callback = LearningRateScheduler(lr_schedule)
    dist.fit(
        x=train_data,
        epochs=epoch,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
            lr_callback,
        ],
        validation_data=test_data,
        verbose=1,
    )
    dist.evaluate(train_data)
    dist.evaluate(test_data)


if __name__ == '__main__':
    # load from txt
    dict_path = os.path.join('data', 'label.names')
    num_class = len(load_dict(dict_path))

    learning_rate = 1e-3
    decay = 1e-4
    time_steps = 16
    label_len = 8
    img_shape = (64, 128, 1)
    n_class = num_class + 1
    batch_size = 32

    teacher_config = {
        'width_multiplier': 1.0,
        'feat_dims': 96,
    }

    student_config = {
        'width_multiplier': 0.25,
        'feat_dims': 64,
    }

    teacher_model = TinyLPR(
        n_class=n_class,
        n_feat=teacher_config['feat_dims'],
        width_multiplier=teacher_config['width_multiplier'],
        train=True,
    ).build(img_shape)
    # teacher_model.summary()

    student_model = TinyLPR(
        n_class=n_class,
        n_feat=student_config['feat_dims'],
        width_multiplier=student_config['width_multiplier'],
        train=True,
    ).build(img_shape)
    # student_model.summary()

    # set data path
    train_path = "data/train"
    val_path = "data/val"
    # set dataloader
    train_loader = DataLoader(
        train_path,
        mode='label',
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
        mode='label',
        img_shape=img_shape,
        time_steps=time_steps,
        label_len=label_len,
        num_class=num_class,
        batch_size=1024,
        shuffle=False,
        data_augmentation=False,
    )

    # distilling train
    distilling_train(
        t_model=teacher_model,
        s_model=student_model,
        train_data=train_loader,
        test_data=val_loader,
        t_weight=r'checkpoints\backup\model.h5',
        epoch=50,
    )


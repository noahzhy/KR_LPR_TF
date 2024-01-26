import tensorflow as tf


class Distilling(tf.keras.models.Model):
    def __init__(self, student_model=None, teacher_model=None, **kwargs):
        super(Distilling, self).__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.T = 0.
        self.alpha = 0.

        self.ctc_loss = None
        self.kd_loss = None
        self.ctc_loss_tracker = tf.keras.metrics.Mean(name='ctc_loss')
        self.kd_loss_tracker = tf.keras.metrics.Mean(name='kd_loss')
        self.sum_loss_tracker = tf.keras.metrics.Mean(name='sum_loss')

    def compile(self, ctc_loss=None, kd_loss=None, T=0., alpha=0.,  **kwargs):
        super(Distilling, self).compile(**kwargs)
        self.ctc_loss = ctc_loss
        self.kd_loss = kd_loss
        self.T = T
        self.alpha = alpha

    @property
    def metrics(self):
        metrics = [self.sum_loss_tracker, self.ctc_loss_tracker, self.kd_loss_tracker]

        if self.compiled_metrics is not None:
            metrics += self.compiled_metrics.metrics

        return metrics

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            logits_pre = self.student_model(x)
            t_logits_pre = self.teacher_model(x, training=False)

            ctc_loss_value = self.ctc_loss(y, tf.math.softmax(logits_pre))
            kd_loss_value = self.kd_loss(tf.math.softmax(t_logits_pre/self.T), tf.math.softmax(logits_pre/self.T))
            sum_loss_value = self.alpha * ctc_loss_value + (1-self.alpha) * kd_loss_value

        self.optimizer.minimize(sum_loss_value, self.student_model.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, tf.math.softmax(logits_pre))

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.ctc_loss_tracker.update_state(ctc_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        logits_pre = self.student_model(x, training=False)
        t_logits_pre = self.teacher_model(x, training=False)

        ctc_loss_value = self.ctc_loss(y, tf.math.softmax(logits_pre))
        kd_loss_value = self.kd_loss(tf.math.softmax(t_logits_pre / self.T), tf.math.softmax(logits_pre / self.T))
        sum_loss_value = self.alpha * ctc_loss_value + (1 - self.alpha) * kd_loss_value

        self.compiled_metrics.update_state(y, tf.math.softmax(logits_pre))

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.ctc_loss_tracker.update_state(ctc_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

import tensorflow as tf
import random
import numpy as np
import tensorflow.keras as keras
from datetime import datetime

inputs = keras.layers.Input(batch_shape=(1, 1, 3))
x = inputs # keras.layers.Dense(1)(inputs)
x = keras.layers.LSTM(48, stateful=True)(x)
policy_output = keras.layers.Dense(2, activation='softmax')(x)
critic_output = keras.layers.Dense(1, activation='linear')(x)
outputs = keras.layers.concatenate(inputs=[policy_output, critic_output])

model = kerae.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=7e-4, epsilon=0.1, decay=0.99)
episodes = 50
steps_per_episode = 100


def main():
    r = 0
    a = 0

    for episode in range(episodes):
        reward_output = random.randint(0, 1)
        model.reset_states()
        reward_sum = 0

        with tf.GradientTape() as tape:
            loss = 0
            for step in range(steps_per_episode):
                neural_net_input = [a, r, step]
                model_input = np.expand_dims(np.expand_dims(neural_net_input, axis=0), axis=0).astype('float32')
                global_step = episode * steps_per_episode + step

                # single batch single time step
                output = model(model_input)[0]
                a = np.argmax(output[0:2])
                value_prediction = output[2]
                r = (a == reward_output)
                reward_sum += r * (step > 0)

                policy_loss = -tf.multiply(tf.math.log(output[a]), r - 0.5)  # value_prediction)
                entropy = -tf.reduce_sum(tf.math.log(output[0:2]) * output[0:2])
                loss += policy_loss + 0.1 * entropy

                tf.summary.scalar('policy_loss', policy_loss, global_step)
                tf.summary.scalar('action', a, global_step)
                tf.summary.scalar('value_prediction', value_prediction, global_step)
                tf.summary.scalar('output_0', output[0], global_step)
                tf.summary.scalar('output_1', output[1], global_step)

        if episode < (episodes * 0.9):
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tf.summary.scalar('loss', loss, global_step)
        tf.summary.scalar('reward_output', reward_output, global_step)
        tf.summary.scalar('reward', reward_sum / steps_per_episode, episode)


if __name__ == '__main__':
    log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        main()

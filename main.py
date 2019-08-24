import tensorflow as tf
import random
import numpy as np
import tensorflow.keras as keras
from datetime import datetime


inputs = keras.layers.Input(batch_shape=(1, 1, 3))
x = keras.layers.Dense(6)(inputs)
x = keras.layers.LSTM(8, stateful=True)(x)
policy_output = keras.layers.Dense(2, activation='softmax' )(x)
critic_output = keras.layers.Dense(1, activation='linear')(x)
outputs = keras.layers.concatenate(inputs=[policy_output, critic_output])

model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=7e-4, epsilon=0.1, decay=0.99)
# optimizer = keras.optimizers.SGD(lr=0.01, decay=0.0001)
gamma = 0

def main():
    r = 0
    a = 0
    
    for episode in range(500):
        reward_output = random.randint(0,1)
        model.reset_states()
        for step in range(5):
            neural_net_input = [a, r, step]
            model_input = np.expand_dims(np.expand_dims(neural_net_input, axis=0), axis=0).astype('float32')
            step = episode * 5 + step
            
            # single batch single time step
            with tf.GradientTape() as tape:
                output = model(model_input)[0]
                a = np.argmax(output[0:2])
                value_prediction = output[2]
                r = 1 if a == reward_output else 0
                policy_loss = -tf.multiply(tf.math.log(output[a]), r - 0.5) # value_prediction)

                entropy = -tf.reduce_sum(tf.math.log(output[0:2]) * output[0:2])
                loss = policy_loss + 0.01 * entropy
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.summary.scalar('reward', r, step)
            tf.summary.scalar('loss', loss, step)
            tf.summary.scalar('policy_loss', policy_loss, step)
            tf.summary.scalar('action', a, step)
            tf.summary.scalar('value_prediction', value_prediction, step)
            tf.summary.scalar('output_0', output[0], step)
            tf.summary.scalar('output_1', output[1], step)

    reward_sum = [0, 0, 0, 0, 0] 
    for test in range(50):
        a = 0
        r = 0
        reward_output = random.randint(0, 1)
        for step in range(5):

            neural_net_input = [a, r, step]
            model_input = np.expand_dims(np.expand_dims(neural_net_input, axis=0), axis=0).astype('float32')

            output = model(model_input)
            a = np.argmax(output[0:2])
            r = 1 if a == reward_output else 0
            reward_sum[step] += r

    print(reward_sum)
    mean_reward = sum(reward_sum) /250.0
    print("ANN achieved mean reward of: " + str(mean_reward))
    print("perfect score is: " + str(225/250.0))

if __name__ == '__main__':
    log_dir = "./runs/" + datetime.now().strftime("%d.%m_%H:%M:%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        main()

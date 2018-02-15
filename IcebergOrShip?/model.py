import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

train_file = open('train.json')
train_data = json.load(train_file)

a=np.array(train_data[4]['band_2']).reshape(75, 75)
b=np.array(train_data[0]['band_2']).reshape(75, 75)
plt.imshow(a)
plt.show()

# print(train_data[1])
# #extract first and second image
# #0 -> [1, 0], 1 -> [0, 1]
# #0 - isn't iceberg, 1 - is iceberg
# def data_formatting(train_data, num_classes=2):
#     images = []
#     labels = []
#     for train_sample in train_data:
#         images.append(train_sample['band_1'] + train_sample['band_2'])
#         labels.append(np.eye(num_classes)[train_sample['is_iceberg']])
#     images = np.array(images)
#     labels = np.array(labels)
#     return images, labels
#
# data_x, data_l = data_formatting(train_data)
#
# train_data_x, test_data_x, train_data_l, test_data_l = train_test_split(data_x, data_l, test_size=0.3)
#
# x = tf.placeholder(tf.float32, [None, 2*75*75], name='x')
# y = tf.placeholder(tf.float32, [None, 2], name='y')
#
# weights = {
#     'hidden_1': tf.Variable(tf.random_normal([2*75*75, 175], seed=42), name='hidden_1'),
#     'hidden_2': tf.Variable(tf.random_normal([175, 75], seed=42), name='hidden_2'),
#     'hidden_3': tf.Variable(tf.random_normal([75, 2], seed=42), name='hidden_3')
# }
#
# biases = {
#     'hidden_1': tf.Variable(tf.random_normal([175], seed=42), name='bias_1'),
#     'hidden_2': tf.Variable(tf.random_normal([75], seed=42), name='bias_2'),
#     'hidden_3': tf.Variable(tf.random_normal([2], seed=42), name='bias_3')
# }
#
# hidden_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['hidden_1'])
# hidden_1 = tf.nn.relu(hidden_1)
#
# hidden_2 = tf.add(tf.matmul(hidden_1, weights['hidden_2']), biases['hidden_2'])
# hidden_2 = tf.nn.relu(hidden_2)
#
# output_layer = tf.add(tf.matmul(hidden_2, weights['hidden_3']), biases['hidden_3'], name='output_layer')
#
# distribution = tf.nn.softmax(output_layer, name='softmax')
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y), name='loss')
#
# learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cost)
#
# init = tf.initialize_all_variables()
#
# pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1), name='pred_temp')
# accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"), name='accuracy')
#
# saver = tf.train.Saver()
#
# error_train = []
# error_test = []
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for i in range(100):
#         batch_x, batch_y = train_data_x, train_data_l
#         _, c = sess.run([optimizer, cost], feed_dict = {learning_rate: 0.001, x: batch_x, y: batch_y})
#         error_train.append(c)
#         batch_x, batch_y = test_data_x, test_data_l
#         c_test = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
#         error_test.append(c_test)
#         print('---', i, ':', accuracy.eval({x: batch_x, y: batch_y}))
#         print('--- ---Loss-train:', c)
#         print('--- ---Loss--test:', c_test)
#         #print('--- ---Labels_true:', batch_y)
#         #print('--- ---Labels_pred:', distribution.eval({x: batch_xa}))
#
#     saver.save(sess, 'my_test_model')
#
# plt.plot(error_train, label='train')
# plt.plot(error_test, label='test')
# plt.legend()
# plt.show()

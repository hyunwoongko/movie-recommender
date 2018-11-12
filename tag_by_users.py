import os

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rating = pd.read_csv('ratings.csv')
tag = pd.read_csv('tags.csv')
movie = pd.read_csv('movies.csv')
# 각종 데이터를 로드합니다.

rating_df = pd.DataFrame(rating)
tag_df = pd.DataFrame(tag)
movie_df = pd.DataFrame(movie)
# 데이터 프레임을 만듭니다.

my_array = []
tmp_array = []
tmp_for_movie_id = 60756
tmp_for_user_id = 2

for userId, movieId, tag, time in tag_df.values:

    if tmp_for_user_id == userId:
        if tmp_for_movie_id == movieId:
            tmp_array.append(tag)

        else:
            my_array.append(list(tmp_array))
            tmp_for_movie_id = movieId
            tmp_array.clear()
            tmp_array.append(tag)

    else:
        tmp_for_user_id = userId


print(my_array)

words = []
s_array = []

for separate_array in my_array:
    arr = list(separate_array)
    s_array.append(arr)
    for word in separate_array:
        words.append(word)

word2int = {}
int2word = {}

for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

sentences = []
for sentence in s_array:
    sentences.append(sentence)

WINDOW_SIZE = 3

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

df = pd.DataFrame(data, columns=['input', 'label'])

ONE_HOT_DIM = len(words)


# 큰숫자 (예를들어 35, 43 등)를 원핫 인코딩 시키는 함수.
# 35 -> (0,0,0,0,.....,1,0,0,0)
# 36 -> (0,0,0,0,.....,0,1,0,0)
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding


X = []  # 입력 배열입니다.
Y = []  # 타겟단어입니다.

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[x]))
    Y.append(to_one_hot_encoding(word2int[y]))

# 넘파이 어레이로 변경
X_train = np.asarray(X)
Y_train = np.asarray(Y)

# 학습과정을 위한 placeholder 생성
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

EMBEDDING_DIM = 256

# 이 두개의 값은 각각 히든레이어의 변수가 됩니다.
W1 = tf.get_variable("W1", shape=[ONE_HOT_DIM, EMBEDDING_DIM], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1]))
L1 = tf.add(tf.matmul(x, W1), b1)

# 출력값
W2 = tf.get_variable("W5", shape=[EMBEDDING_DIM, ONE_HOT_DIM], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(L1, W2), b2))

# 코스트합수 : 크로스 엔트로피
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# 학습과정
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 800

for i in range(iteration):
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    print('학습 ' + str(i) + ' 현재 코스트 : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

vectors = sess.run(W1 + b1)

print('================================')
print('================================')
print('================================')


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def find_closest(word_index, vectors):
    min_dist = 100000
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


test1 = 'action'
test2 = 'dark'
test3 = 'funny'
test4 = 'Sci-fi'

closest1 = int2word[find_closest(word2int[test1], vectors)]
closest2 = int2word[find_closest(word2int[test2], vectors)]
closest3 = int2word[find_closest(word2int[test3], vectors)]
closest4 = int2word[find_closest(word2int[test4], vectors)]

print(test1, '와 가장 비슷한 단어는 ', closest1)
print(test2, '와 가장 비슷한 단어는 ', closest2)
print(test3, '와 가장 비슷한 단어는 ', closest3)
print(test4, '와 가장 비슷한 단어는 ', closest4)

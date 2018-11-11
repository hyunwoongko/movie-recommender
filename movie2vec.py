from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

rating = pd.read_csv('ratings.csv')
tag = pd.read_csv('tags.csv')
movie = pd.read_csv('movies.csv')
# 각종 데이터를 로드합니다.

rating_df = pd.DataFrame(rating)
tag_df = pd.DataFrame(tag)
movie_df = pd.DataFrame(movie)
# 데이터 프레임을 만듭니다.

tag_df = tag_df.drop('userId', 1)
tag_df = tag_df.drop('timestamp', 1)
# 일단 tag 데이터프레임에서 당장에 필요없는 userId 열과 timestamp 열을 제거합니다.

tag_df = tag_df.sort_values(by='movieId')

my_array = []
tmp_array = []
tmp_for_movie_id = 1

for movieId, tag in tag_df.values:

    if tmp_for_movie_id == movieId:
        tmp_array.append(tag)

    else:
        my_array.append(set(tmp_array))
        tmp_for_movie_id = movieId
        tmp_array.clear()
        tmp_array.append(tag)

words = []
s_array = []

for separate_array in my_array:
    arr = list(separate_array)
    s_array.append(arr)
    for word in separate_array:
        words.append(word)

word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in s_array:
    sentences.append(sentence)

WINDOW_SIZE = 2

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

# 임베딩 차원 = 2
EMBEDDING_DIM = 2

# 이 두개의 값은 각각 히든레이어의 변수가 됩니다.
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1]))
hidden_layer = tf.add(tf.matmul(x, W1), b1)

# 출력값
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))

# 코스트합수 : 크로스 엔트로피
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# 학습과정
train_op = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 30000
for i in range(iteration):
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    print('학습 ' + str(i) + ' 현재 코스트 : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

vectors = sess.run(W1 + b1)
print(vectors)

w2v_df = pd.DataFrame(vectors, columns=['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
# 한국어 폰트 설정

fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)

plt.show()
# matplotlib을 이용한 시각화

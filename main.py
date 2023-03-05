__author__ = "NullAndrei"
__version__ = "1.0.0"
__github__ = "https://github.com/NullAndrei/"

# imports
import os.path
import sys

# check the dataset and we files
dataset_we_check = ['news.csv', 'glove.6B.50d.txt']
if os.path.exists(dataset_we_check[0]) is not True or os.path.exists(dataset_we_check[1]) is not True:
	print("\nMissing one of required files!\n\nPlease download the data.zip from the link below and unzip it in this directory\n\nhttps://drive.proton.me/urls/PHA9NTS41W#YfBsBk2q6GDh\n\nPassword is: gVwMLTu39\n")
	sys.exit()

#other imports
import numpy as np
import pandas as pd
import json
import csv
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
  
import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
tf.disable_eager_execution()

# start the script
print("""
#############################################################################################
#       ______      __           _   __                     ____       __            __     #
#      / ____/___ _/ /_____     / | / /__ _      _______   / __ \___  / /____  _____/ /_    #
#     / /_  / __ `/ //_/ _ \   /  |/ / _ \ | /| / / ___/  / / / / _ \/ __/ _ \/ ___/ __/    #
#    / __/ / /_/ / ,< /  __/  / /|  /  __/ |/ |/ (__  )  / /_/ /  __/ /_/  __/ /__/ /_      #
#   /_/    \__,_/_/|_|\___/  /_/ |_/\___/|__/|__/____/  /_____/\___/\__/\___/\___/\__/      #
#                                                                                           #
#############################################################################################

""")

# sample text to check if fake or not
X = str(input(">>> INPUT YOUR NEWS TEXT: "))

# Reading the data
data = pd.read_csv("news.csv")
data.head()

data = data.drop(["Unnamed: 0"], axis=1)
data.head(5)

# encoding the labels
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1

title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(
	sequences1, padding=padding_type, truncating=trunc_type)
split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

embeddings_index = {}
with open('glove.6B.50d.txt') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

# generating embeddings
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
for word, i in word_index1.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embeddings_matrix[i] = embedding_vector

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size1+1, embedding_dim,
							input_length=max_length, weights=[
								embeddings_matrix],
							trainable=False),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv1D(64, 5, activation='relu'),
	tf.keras.layers.MaxPooling1D(pool_size=4),
	tf.keras.layers.LSTM(64),
	tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
			optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 50

training_padded = np.array(training_sequences1)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences1)
testing_labels = np.array(test_labels)

history = model.fit(training_padded, training_labels,
					epochs=num_epochs,
					validation_data=(testing_padded,
									testing_labels),
					verbose=2)

# detection
sequences = tokenizer1.texts_to_sequences([X])[0]
sequences = pad_sequences([sequences], maxlen=54,
						padding=padding_type,
						truncating=trunc_type)
if(model.predict(sequences, verbose=0)[0][0] >= 0.5):
    print(f"""
#############################################################################################
#       ______      __           _   __                     ____       __            __     #
#      / ____/___ _/ /_____     / | / /__ _      _______   / __ \___  / /____  _____/ /_    #
#     / /_  / __ `/ //_/ _ \   /  |/ / _ \ | /| / / ___/  / / / / _ \/ __/ _ \/ ___/ __/    #
#    / __/ / /_/ / ,< /  __/  / /|  /  __/ |/ |/ (__  )  / /_/ /  __/ /_/  __/ /__/ /_      #
#   /_/    \__,_/_/|_|\___/  /_/ |_/\___/|__/|__/____/  /_____/\___/\__/\___/\___/\__/      #
#                                                                                           #
#############################################################################################

Checked news: {X}
    >>> Result: This news are REAL!
    """)
else:
    print(f"""
#############################################################################################
#       ______      __           _   __                     ____       __            __     #
#      / ____/___ _/ /_____     / | / /__ _      _______   / __ \___  / /____  _____/ /_    #
#     / /_  / __ `/ //_/ _ \   /  |/ / _ \ | /| / / ___/  / / / / _ \/ __/ _ \/ ___/ __/    #
#    / __/ / /_/ / ,< /  __/  / /|  /  __/ |/ |/ (__  )  / /_/ /  __/ /_/  __/ /__/ /_      #
#   /_/    \__,_/_/|_|\___/  /_/ |_/\___/|__/|__/____/  /_____/\___/\__/\___/\___/\__/      #
#                                                                                           #
#############################################################################################

Checked news: {X}
    >>> Result: This news are FAKE!
    """)

![GIF example](https://gifyu.com/images/ezgif.com-video-to-gif25c19cf7475c2212.gif)

# AI Fake News Detector
This is a simple Fake News Detector written in python

[How does the script work?](#How-does-the-script-work?)

[How the script is build?](#How-the-script-is-build?)

[How to use the script?](#How-to-use-the-script?)

[For the end](#For-the-end)

![https://i.ibb.co/f8yMm5m/2023-03-04-20-48.png](https://i.ibb.co/f8yMm5m/2023-03-04-20-48.png)

***What is [fake news](https://en.wikipedia.org/wiki/Fake_news)?***
> Fake news is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. Although false news has always been spread throughout history, the term "fake news" was first used in the 1890s when sensational reports in newspapers were common.

# How does the script work?
> This script is developed an deep learning model using [Tensorflow](https://www.tensorflow.org/) and use this model to detect whether the news is fake or not. This script also use fake_news_dataset, which contains News text and corresponding label fake or real.

**Libraries:**
- [NumPy](https://numpy.org/): To perform different mathematical functions.
- [Pandas](https://pandas.pydata.org/): To load dataset.
- [Tensorflow](https://www.tensorflow.org/): To preprocessing the data and to create the model.
- [SkLearn](https://scikit-learn.org/stable/): For train-test split and to import the modules for model evaluation.

- [x] Importing Libraries and dataset
- [x] Preprocessing Dataset
- [x] Generating Word Embeddings
- [x] Model Architecture
- [x] Model Evaluation and Prediction

# How the script is build?
```python
# Reading the data
data = pd.read_csv("news.csv")
data.head()
```
**Preprocessing Dataset**
- The dataset contains one unnamed column. So I drop that column from the dataset.
```python
data = data.drop(["Unnamed: 0"], axis=1)
data.head(5)
```
**Data Encoding**
```python
# encoding the labels
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])
```
- It converts the categorical column (label in out case) into numerical values.
- These are some variables required for the model training.
```python
embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1
```
**Tokenization**
```python
title = []
text = []
labels = []
for x in range(training_size):
	title.append(data['title'][x])
	text.append(data['text'][x])
	labels.append(data['label'][x])
```
- This process divides a large piece of continuous text into distinct units or tokens basically. Here I use columns separately for a temporal basis as a pipeline just for good accuracy.
- Applying Tokenization
```python
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
```
**Generating Word Embedding**
```python
embeddings_index = {}
with open('glove.6B.50d.txt') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

# Generating embeddings
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
for word, i in word_index1.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embeddings_matrix[i] = embedding_vector
```
- It allows words with similar meanings to have a similar representation. Here each individual word is represented as real-valued vectors in a predefined vector space. For that I will use glove.6B.50d.txt. It has the predefined vector space for words.
**Creating Model Architecture**
```python
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
```
- Now it’s time to introduce TensorFlow to create the model. Here I use the TensorFlow embedding technique with Keras Embedding Layer where I map original input data into some set of real-valued dimensions.
```python
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
```
**Model Evaluation and Prediction**
```python
# sample text to check if fake or not
X = str(input(">>> INPUT YOUR NEWS TEXT: "))

# detection
sequences = tokenizer1.texts_to_sequences([X])[0]
sequences = pad_sequences([sequences], maxlen=54,
						padding=padding_type,
						truncating=trunc_type)
if(model.predict(sequences, verbose=0)[0][0] >= 0.5):
	print(f"Checked news: {X}\n>>> Result: This news are REAL!")
else:
	print(f"Checked news: {X}\n>>> Result: This news are FAKE!")
```
- Now, the detection model is built using TensorFlow. Now I will try to test the model by using some news text by predicting whether it is true or false.

# How to use the script?
- Clone the repo
```
git clone https://github.com/NullAndrei/AI-Fake-News-Detector.git
```
- Download the **data.zip** from the link: [data.zip](https://drive.proton.me/urls/PHA9NTS41W#YfBsBk2q6GDh)

The password is: **gVwMLTu39**
- Unzip the **data.zip** in the **AI-Fake-News-Detector/** directory
- Now install the requirements
```
pip install -r requirements.txt
```
- Now you are ready to run the script
```
python main.py
```
- Input the sample text to check if fake or not.

# For the end
> Thanks to everyone who took the time to read this repo. If you like it, throw a star. Everyone has the right to improve this script and make it even better. Follow my Github profile for more projects like this.

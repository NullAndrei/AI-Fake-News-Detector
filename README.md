![GIF example](https://gifyu.com/images/ezgif.com-video-to-gif25c19cf7475c2212.gif)

# AI Fake News Detector
This is a simple Fake News Detector written in python

![https://i.ibb.co/f8yMm5m/2023-03-04-20-48.png](https://i.ibb.co/f8yMm5m/2023-03-04-20-48.png)

***What is [fake news](https://en.wikipedia.org/wiki/Fake_news)?***
> Fake news is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. Although false news has always been spread throughout history, the term "fake news" was first used in the 1890s when sensational reports in newspapers were common.

### How does the script work?:
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

### How the script is build?
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

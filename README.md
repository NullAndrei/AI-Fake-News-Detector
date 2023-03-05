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


# coding: utf-8

# # Task Description
# 
# #### Main idea of project:
# 
# Collect samples of tweets from twitter and then try to use Natural Language Processing(NLP) algorithms to analyse the data and look for the physiologic indicators. Classify the tweets according to the physiologic behavior individuals.
# 
# 1. It should have 2-3 algorithms to collect the tweets & 2-3 algorithms to analyze the tweets. Finally, It should have 4-5 algorithms in total.
# 
# 2. What algorithms did you use and what is the result of the analysis after using Linguistic Inquiry and Word Count(LIWC) library?
# 
# 3. Which algorithm is producing the accurate result?
# 
# 4. Either it should append the previous tweets to the current set of tweets OR It should collect more than 1 million tweets.
# 
# 5. Plot those results in a graph.
# 
# 
# 
# ## Reference:
# 
# 
# 1. https://arxiv.org/abs/1705.00335
# 
# 2. https://www.rand.org/content/dam/rand/pubs/rgs_dissertations/RGSD300/RGSD391/RAND_RGSD391.pdf
#   
# 3. https://www.researchgate.net/publication/329203859_Linguistic_analysis_of_the_autobiographical_memories_of_individuals_with_major_depressive_disorder
#     
# 4. https://github.com/lakshmanboddu/Quantifying-Mental-Health-Signals-on-Twitter
# 

# ## Import required libraries

# In[1]:


# Twitter Streaming
from twython import Twython

# DataFrame
import csv
import json
import numpy as np
import pandas as pd
#set some pandas options controling output format
pd.set_option('display.notebook_repr_html',True) # output as flat text and not HTML
pd.set_option('display.max_rows', None) # this is the maximum number of rows we will display
pd.set_option('display.max_columns', None) # this is the maximum number of rows we will display
import pandas as pd
import seaborn as sns

# Matplot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools


# In[2]:


# # Instantiate an object
python_tweets = Twython('OuU4ihzPX8aOFtt9gvb7envzM','yYicuj0suUFJNUfRoSxubEDguVoBzVyVw6RXsqJOV2HQ9tbzNd')


# In[5]:


# Create our query
topic = 'ADHD OR #ADHD OR Attention deficit hyperactivity disorder OR Attention-deficit/hyperactivity disorder -RT'
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}

with open('txt/ADHD.txt', 'a+', encoding="utf-8") as f:
    with open('json/ADHD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[6]:


topic = 'PTSD OR #PTSD OR Post traumatic stress disorder -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/PTSD.txt', 'a+', encoding="utf-8") as f:
    with open('json/PTSD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[7]:


topic = 'Anxiety OR Anxiety disorder OR anxiety disorder -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2019-01-01', 'language':'en'}
with open('txt/AD.txt', 'a+', encoding="utf-8") as f:
    with open('json/AD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[8]:


topic = 'Eating disorder OR eating disorder OR bulimia -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01','language':'en'}
with open('txt/ED.txt', 'a+', encoding="utf-8") as f:
    with open('json/ED.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[9]:


topic = 'Depression OR clinical depression OR depressed -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/CD.txt', 'a+', encoding="utf-8") as f:
    with open('json/CD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[10]:


topic = 'Bipolar Disorder OR bipolar disorder OR bipolar -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/BD.txt', 'a+', encoding="utf-8") as f:
    with open('json/BD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[16]:


topic = 'Suicide -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/Suicide.txt', 'a+', encoding="utf-8") as f:
    with open('json/Suicide.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[15]:


topic = 'Autism -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/Autism.txt', 'a+', encoding="utf-8") as f:
    with open('json/Autism.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[14]:


topic = 'Depression -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/Depression.txt', 'a+', encoding="utf-8") as f:
    with open('json/Depression.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[13]:


topic = 'Addiction or substance use or drugs -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/Addiction.txt', 'a+', encoding="utf-8") as f:
    with open('json/Addiction.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[12]:


topic = 'Anxiety -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/Anxiety.txt', 'a+', encoding="utf-8") as f:
    with open('json/Anxiety.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# In[11]:


topic = 'Obsessive-Compulsive Disorder OR OCD -RT'
# Create our query
query = {'q': topic,'count':10000, 'since':'2017-01-01', 'language':'en'}
with open('txt/OCD.txt', 'a+', encoding="utf-8") as f:
    with open('json/OCD.json', 'a+') as f1:
        for status in python_tweets.search(**query)['statuses']:
            text = status['text']
            text = text.replace('\n',' ')
            f.write(text)
            f.write("\n")
            json.dump(status,f1)
            f1.write("\n")
            print(status['text'])
    f1.close()
f.close()


# ## Merging all the files in the dataframe

# In[3]:


import glob

path = r'C://Users/shiva/Desktop/RaviKrishna/txt/' # use your path
all_files = glob.glob(path + "*.txt")

li = []

for filename in all_files:
    disorder_type = str(filename[40:-4])
    df = pd.read_csv(filename, sep="\n", header=None,error_bad_lines=False,encoding='utf8', quoting=csv.QUOTE_NONE)
    df['Disorder Type'] = disorder_type 
    li.append(df)
tweetsDF = pd.concat(li, axis=0, ignore_index=True)


# In[4]:


tweetsDF.columns = ['Tweets', 'Disorder Type']
print("Shape of the DataFrame", tweetsDF.shape)
tweetsDF.head()


# ### Define Helper Functions

# In[5]:


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def analize_sentiment(tweet):
    # Simple implementation of the sgn(x) function to make the analysis more comprenesive. 
    
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# In[6]:


from textblob import TextBlob
import re

tweetsDF['Sentiment'] = np.array([ analize_sentiment(tweet) for tweet in tweetsDF['Tweets'] ])

pos_tweets = [ tweet for index, tweet in enumerate(tweetsDF['Tweets']) if tweetsDF['Sentiment'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(tweetsDF['Tweets']) if tweetsDF['Sentiment'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(tweetsDF['Tweets']) if tweetsDF['Sentiment'][index] < 0]

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(tweetsDF['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(tweetsDF['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(tweetsDF['Tweets'])))

display(tweetsDF.head(20))


# In[7]:


### Define more helper functions
def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string 
    #convert the tweet to lower case
    tweet.lower()
    #convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to ""
    tweet = re.sub('@[^\s]+','', tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


# Clean a tweet
def clean_tweet(text):
    # Removal of URLs
    text = re.sub(r"http\S+", "", text)
    # Removal of mentions
    text = re.sub("@[^\s]*", "", text)
    # Removal of hashtags
    text = re.sub("#[^\s]*", "", text)
    # Removal of numbers
    text = re.sub('[0-9]*[+-:]*[0-9]+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Apostrophe lookup
    text = re.sub("'ll", " will", text)
    text = re.sub("'ve", " have", text)
    text = re.sub("n't", " not", text)
    text = re.sub("'d", " would", text)
    text = re.sub("'re", " are", text)
    text = re.sub("i'm", "i am", text)
    text = re.sub("it's", "it is", text)
    text = re.sub("she's", "she is", text)
    text = re.sub("he's", "he is", text)
    text = re.sub("here's", "here is", text)
    text = re.sub("that's", "that is", text)
    text = re.sub("there's", "there is", text)
    text = re.sub("what's", "what is", text)
    text = re.sub("who's", "who is", text)
    text = re.sub("'s", "", text)
    # Handling slang words
    text = re.sub(r"\btmrw\b", "tomorrow", text)
    text = re.sub(r"\bur\b", "your", text)
    text = re.sub(r"\burs\b", "yours", text)
    text = re.sub(r"\bppl\b", "people", text)
    text = re.sub(r"\byrs\b", "years", text)
    # Handling acronyms
    text = re.sub(r"\b(rt)\b", "retweet", text)
    text = re.sub(r"\b(btw)\b", "by the way", text)
    text = re.sub(r"\b(asap)\b", "as soon as possible", text)
    text = re.sub(r"\b(fyi)\b", "for your information", text)
    text = re.sub(r"\b(tbt)\b", "throwback thursday", text)
    text = re.sub(r"\b(tba)\b", "to be announced", text)
    text = re.sub(r"\b(tbh)\b", "to be honest", text)
    text = re.sub(r"\b(faq)\b", "frequently asked questions", text)
    text = re.sub(r"\b(icymi)\b", "in case you missed it", text)
    text = re.sub(r"\b(aka)\b", "also known as", text)
    text = re.sub(r"\b(ama)\b", "ask me anything", text)
    # Word lemmatization
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
#df['SentimentText'] = df['SentimentText'].apply(lambda text: clean_tweet(text))

def feature_extraction(data, method = "tfidf"):
    #arguments: data = all the tweets in the form of array, method = type of feature extracter
    #methods of feature extractions: "tfidf" and "doc2vec"
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") # we need to give proper stopwords list for better performance
        features=tfv.fit_transform(data)
    elif method == "doc2vec":
        None
    else:
        return "Incorrect inputs"
    return features


# In[8]:


from bs4 import BeautifulSoup
from html.parser import HTMLParser
from nltk import WordNetLemmatizer
tweetsDF['Cleaned Tweets'] = tweetsDF['Tweets'].apply(preprocess_tweet)
tweetsDF['Cleaned Tweets'] = tweetsDF['Tweets'].apply(clean_tweet)


# In[9]:


tweetsDF.head(10)


# ## Exploratory Data Visualization

# In[10]:


plt.style.use('fivethirtyeight')
dist = tweetsDF.groupby(["Sentiment"]).size()
dist = dist / dist.sum()
fig, ax = plt.subplots(figsize=(15,7))
sns.barplot(dist.keys(), dist.values);
plt.title("Distribution of Sentiments in all Tweets");


# In[11]:


plt.style.use('bmh')
dist = tweetsDF.groupby(["Disorder Type"]).size()
dist = dist / dist.sum()
fig, ax = plt.subplots(figsize=(15,7))
sns.barplot(dist.keys(), dist.values);
plt.title("Distribution of Disorder Type in all Tweets");


# In[12]:


# Display sentiment percentages as pie chart
# Create a numpy vector mapped to labels
sentiment = np.zeros(3)

labels = np.array(["Positive", "Neutral", "Negative"])

sentiment[0] = len(pos_tweets)*100/len(tweetsDF['Tweets'])
sentiment[1] = len(neu_tweets)*100/len(tweetsDF['Tweets'])
sentiment[2] = len(neg_tweets)*100/len(tweetsDF['Tweets'])

sentiment /= 100

# Plot pie chart:
pie_chart = pd.Series(sentiment, index=labels, name='Sentiment')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(17,7))
plt.title("% of Sentiment Distribution");


# In[13]:


sns.set_context('talk')
plt.style.use('classic')
sns.set(rc={'figure.figsize':(15,7)})
sns.countplot('Sentiment', hue='Disorder Type',data=tweetsDF,palette=("Accent"))
plt.title("Sentiment Distribution across the various Disorder Types");


# In[14]:


tweetsDF['Tweets'].str.len().plot.hist(color = 'pink', figsize = (12, 7))
plt.title('Distribution of the length of Tweets');


# In[15]:


tweetsDF['len'] = tweetsDF['Tweets'].str.len()


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(tweetsDF.Tweets)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7))
plt.title("Most Frequently Occuring Words - Top 20");


# In[18]:


from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'white', width = 1800, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(17,8))
plt.imshow(wordcloud);


# ---

# # Machine Learning Models

# ### TFIDF

# In[17]:


import warnings
warnings.filterwarnings('always')
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

def train_test(features, labels):
    docs_trn, docs_tst, y_trn, y_tst = train_test_split(features, label)
    return docs_trn, docs_tst, y_trn, y_tst
data = np.array(tweetsDF.Tweets)
label = np.array(tweetsDF['Disorder Type'])
features = feature_extraction(data, method = "tfidf")

X_train, X_test, y_train, y_test = train_test(features, labels)


# In[18]:


classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[19]:


log


# In[25]:


sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for TF-IDF Feature Extraction Algorithm');


# ### CountVect

# In[20]:


import string
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def feature_extraction(data, method = "tfidf"):
    #arguments: data = all the tweets in the form of array, method = type of feature extracter
    #methods of feature extractions: "tfidf" and "doc2vec"
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfv=TfidfVectorizer(sublinear_tf=True, stop_words = "english") # we need to give proper stopwords list for better performance
        features=tfv.fit_transform(data)
    elif method == "countVect":
        from sklearn.feature_extraction.text import CountVectorizer
        tfv=CountVectorizer(analyzer=text_process) # we need to give proper stopwords list for better performance
        features=tfv.fit_transform(data)
    else:
        return "Incorrect inputs"
    return features


# In[21]:


warnings.simplefilter("ignore", category=ResourceWarning)
data = np.array(tweetsDF.Tweets)
label = np.array(tweetsDF['Disorder Type'])
features = feature_extraction(data, method = "countVect")
X_train, X_test, y_train, y_test = train_test(features, labels)


# In[22]:


classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[23]:


log


# In[24]:


sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for Count2vec Feature Extraction Algorithm');


# ### Boosting Accuracy with LIWC and other embeddings and Neural Networks

# # LIWC

# !pip install -U liwc

# In[25]:


import string
from collections import defaultdict
import multiprocessing as mp
from nltk import word_tokenize
import re
import pickle
import nltk

liwcPath = 'liwc/liwc-english-mod.dic'

TRANSLATE_TABLE = dict((ord(char), None) for char in string.punctuation)


class LIWC():
    """Top-level class"""

    def __init__(self, dict_path):
        self.lexicon, self.category_names = self._read_dic(dict_path)
        self.trie = self._build_trie(self.lexicon)

    def process_text(self, text):
        """Run LIWC on string"""

        tokenized = word_tokenize(text.lower().translate(TRANSLATE_TABLE))
        counts = defaultdict(int)
        dict_count = len(tokenized)

        for token in tokenized:
            classifications = list(self._parse_token(token))

            if not classifications:
                dict_count -= 1
            else:
                for category in classifications:
                    counts[category] += 1

        category_scores = {category: (
            counts[category] / len(tokenized)) * 100 for category in counts.keys()}

        return category_scores

    def process_df_mp(self, df, col):
        """Multi-process version of process_df"""
        cpu_count = mp.cpu_count()
        p = mp.Pool(cpu_count)

        batches = np.array_split(df, cpu_count)

        pool_results = p.starmap(self.process_df,[(batch, col) for batch in batches if len(batch) > 0])
        p.close()
        
        return pd.concat(pool_results, axis=0)

    def process_df(self, df, col):
        """Run LIWC on a dataframe column"""
        df[col] = df[col].astype(str)

        def apply_df(row, col):
            score = self.process_text(row[col])
            scores = {}
            
            for category in score:
                scores[category] = score[category]

            return pd.Series(scores)


        res = df.apply(apply_df, args=(col,), axis=1)

        return res


    def _read_dic(self, filepath):
        category_mapping = {}
        category_names = []
        lexicon = {}
        mode = 0    # the mode is incremented by each '%' line in the file
        with open(filepath) as dict_file:
            for line in dict_file:
                tsv = line.strip()
                if tsv:
                    parts = tsv.split('\t')
                    if parts[0] == '%':
                        mode += 1
                    elif mode == 1:
                        # definining categories
                        category_names.append(parts[1])
                        category_mapping[parts[0]] = parts[1]
                    elif mode == 2:
                        lexicon[parts[0]] = [category_mapping[category_id]
                                             for category_id in parts[1:]]
        return lexicon, category_names

    def _build_trie(self, lexicon):
        '''
        Build a character-trie from the plain pattern_string -> categories_list
        mapping provided by `lexicon`.

        Some LIWC patterns end with a `*` to indicate a wildcard match.
        '''
        trie = {}
        for pattern, category_names in lexicon.items():
            cursor = trie
            for char in pattern:
                if char == '*':
                    cursor['*'] = category_names
                    break
                if char not in cursor:
                    cursor[char] = {}
                cursor = cursor[char]
            cursor['$'] = category_names
        return trie

    def _search_trie(self, trie, token, token_i=0):
        '''
        Search the given character-trie for paths that match the `token` string.
        '''
        if '*' in trie:
            return trie['*']
        elif '$' in trie and token_i == len(token):
            return trie['$']
        elif token_i < len(token):
            char = token[token_i]
            if char in trie:
                return self._search_trie(trie[char], token, token_i + 1)
        return []

    def _parse_token(self, token):
        for category_name in self._search_trie(self.trie, token):
            yield category_name
            
            
''' 
Make LIWC feature extractor into class
'''


def makeLIWCDictionary(liwcPath, picklePath):
    '''
        Make lookup data structure from LIWC dictionary file
    '''
    LIWC_file = open(liwcPath, 'rb') # LIWC dictionary
    catNames = {}
    LIWC_file.readline() #skips first '%' line
    line = LIWC_file.readline()
    lookup = []
    while '%' not in line:
        keyval = line.split('\t')
        key = keyval[0]
        value = keyval[1].strip()
        catNames[key] = {'name' : value,
                         'words' : []}
        line = LIWC_file.readline()
    mapCategoriesToNumbers = catNames.keys()
    line = LIWC_file.readline() # skips second '%' line

    #return mapCategoriesToNumbers
    while line: #iterate through categories
        data = line.strip().split('\t')
        reString = '^'+data[0].replace('*', '.*') + '$'
        indeces = [mapCategoriesToNumbers.index(d) for d in data[1:]]
        lookupCell = (re.compile(reString), indeces)
        lookup.append(lookupCell)
        for cat in data[1:]:
            catNames[cat]['words'] += (data[0], reString)
        cats = data[1:]
        line = LIWC_file.readline()
    toPickle = {'categories' : catNames, 'lookup' : lookup, 'cat_to_num' : mapCategoriesToNumbers}
    pickle.dump(toPickle, open(picklePath, 'w'))
    return toPickle

class liwcExtractor():
    def __init__(self,
                tokenizer=None,
                ignore=None,
                dictionary=None,
                newCategories=None,
                keepNonDict=True,
                liwcPath=None):
        self.liwcPath = liwcPath
        self.dictionary = dictionary
        if tokenizer is None:
            self.tokenizer = self.nltk_tokenize
        if liwcPath is not None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        elif self.dictionary==None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        self.ignore = ignore
        self.newCategories = newCategories
        self.nonDictTokens = []
        self.keepNonDict = keepNonDict

    def getCategoryIndeces(self):
        indeces = [x['name'] for x in self.categories.values()]
        indeces += ['wc', 'sixltr','dic','punc','emoticon'] # These last two are not built yet.
        return indeces

    def extract(self, corpus):
        corpusFeatures = []
        for doc in corpus:
            features = self.extractFromDoc(doc)
            corpusFeatures.append(features)
        return corpusFeatures

    def extractFromDoc(self, document):
        tokens = self.tokenizer(document)
        #print tokens
        features = [0] * 70 # 66 = wc, total word count
                            # 67 = sixltr, six letter words
                            # 68 = dic, words found in LIWC dictionary
                            # 70 = punc, punctuation
                            # 71 = emoticon
        features[66] = len(tokens)

        for t in tokens: #iterating through tokens of a message
            #print "Token : " + t
            if len(t) > 6: # check if more than six letters
                features[67] += 1
            inDict = False
            for pattern, categories in self.lookup:
                if len(pattern.findall(t)) > 0:
                    inDict = True
                    for c in categories:
                        features[int(c)] += 1
            if inDict:
                features[68] += 1
            else:
                self.nonDictTokens.append(t)
        return features

    def patternsMatchedFromDoc(self, document):
        tokens = self.tokenizer(document)
        patterns = [l[0] for l in self.lookup]
        features = [0] * len(patterns)
        for t in tokens:
            for i, pattern in enumerate(patterns):
                if len(pattern.findall(t)) > 0:
                    features[i] += 1
        return features

    def nltk_tokenize(self, message):
        '''
            takes in a text string and returns a list of tokenized words using nltk methods
        '''
        # sentence tokenize
        stList = nltk.sent_tokenize(message)
        # word tokenize
        tokens = []
        for sent in stList:
            tokens += nltk.word_tokenize(sent)
        return tokens


# In[26]:


# initialize liwc 
liwc = LIWC("LIWC2015_English_Flat.dic")
liwc_df = liwc.process_df(tweetsDF, col='Cleaned Tweets')
# Save the dataframe
liwc_df.to_csv('LIWC_features.csv', index = False)


# In[33]:


# Replace Nans with zeros
liwc_df = liwc_df.fillna(0)


# #### Checking LIWC features only

# In[34]:


X_train, X_test, y_train, y_test = train_test(liwc_df, labels)
    
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[36]:


print(log)
sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for LIWC Feature Extraction Algorithm');


# ### Adding Extra Features

# In[37]:


import re
def adding_extra_feature(df, tweet_column):
    
    # Print Number of Exclamation
    #length_of_excl = (len(re.findall(r'!', string)))
    df['number_of_exclamation'] = tweet_column.apply(lambda x: (len(re.findall(r'!', x))))
    
    # Number of ?
    #length_of_questionmark = (len(re.findall(r'?', string)))
    df['number_of_questionmark'] = tweet_column.apply(lambda x: (len(re.findall(r'[?]', x))))
    
    # Number of #
    df['number_of_hashtag'] = tweet_column.apply(lambda x: (len(re.findall(r'#', x))))
    
    # Number of @
    df['number_of_mention'] = tweet_column.apply(lambda x: (len(re.findall(r'@', x))))
    
    # Number of Quotes
    df['number_of_quotes'] = tweet_column.apply(lambda x: (len(re.findall(r"'", x))))

    # Number if underscore
    df['number_of_underscore'] = tweet_column.apply(lambda x: (len(re.findall(r'_', x))))
    
    
    return df


# In[38]:


tweetsDF = adding_extra_feature(tweetsDF, tweetsDF["Tweets"])


# In[39]:


tweetsDF.describe()


# ### Adding Emoticons

# - Here, users emoticons in a tweet also matters, so we will find the emoticons in a users tweet.

# In[40]:


## Emoticon Detector

class EmoticonDetector:
    emoticons = {}

    def __init__(self, emoticon_file="emoticons.txt"):
        from pathlib import Path
        content = Path(emoticon_file).read_text()
        positive = True
        for line in content.split("\n"):
            if "positive" in line.lower():
                positive = True
                continue
            elif "negative" in line.lower():
                positive = False
                continue

            self.emoticons[line] = positive

    def is_positive(self, emoticon):
        if emoticon in self.emoticons:
            return self.emoticons[emoticon]
        return False

    def is_emoticon(self, to_check):
        return to_check in self.emoticons
ed = EmoticonDetector()

processed_data = tweetsDF.copy()

def add_column(column_name, column_content):
    processed_data.loc[:, column_name] = pd.Series(column_content, index=processed_data.index)

def count_by_lambda(expression, word_array):
    return len(list(filter(expression, word_array)))

add_column("splitted_text", map(lambda txt: txt.split(" "), processed_data["Tweets"]))

positive_emo = list(
    map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
        processed_data["splitted_text"]))
add_column("number_of_positive_emo", positive_emo)

negative_emo = list(map(
    lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
    processed_data["splitted_text"]))

add_column("number_of_negative_emo", negative_emo)


# In[44]:


emoticons_df = processed_data[['number_of_positive_emo', 'number_of_negative_emo']] 
extra_featuresDF = tweetsDF[['number_of_exclamation', 'number_of_questionmark', 'number_of_hashtag', 'number_of_mention', 'number_of_quotes', 'number_of_underscore']]


# ## Merging LIWC and TFIDF features

# In[47]:


# TFIDF features has the shape
features.shape


# In[48]:


from scipy.sparse import hstack
# New Dataframe has the shape
tfidf_liwcDF = hstack([features, np.array(liwc_df)])
print(tfidf_liwcDF.shape)


# In[51]:


X_train, X_test, y_train, y_test = train_test(tfidf_liwcDF, labels)
    
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
print('\n')
print(log)
sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for LIWC and TFIDF features');


# ## Merging LIWC, TFIDF, Extra Features features

# In[49]:


# New Dataframe has the shape
tfidf_liwc_EFDF = hstack([tfidf_liwcDF, np.array(extra_featuresDF)])
print(tfidf_liwc_EFDF.shape)


# In[54]:


X_train, X_test, y_train, y_test = train_test(tfidf_liwc_EFDF, labels)
    
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
print('\n')
print(log)
sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for LIWC, TFIDF, Extra Features features');


# ## Merging LIWC, TFIDF Extra features and Emoticons

# In[50]:


# New Dataframe has the shape
tfidf_liwc_EF_emoDF = hstack([tfidf_liwc_EFDF, np.array(emoticons_df)])
print(tfidf_liwc_EF_emoDF.shape)


# In[55]:


X_train, X_test, y_train, y_test = train_test(tfidf_liwc_EF_emoDF, labels)
    
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    #print("Classification Report")
    #print(classification_report(y_test, train_predictions))
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
print('\n')
print(log)
sns.barplot(x='Accuracy', y='Classifier', data=log);
plt.title('Accuracies of the Models for LIWC, TFIDF, Extra Features features');


import tweepy
import pandas as pd
import re
from neo4j import GraphDatabase
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from main import twpy



def connectTwitter():
    consumer_key = 'NWZXm0sdLsYMkH7X6f1WRrScL'
    consumer_secret = 'TZilfuUyaRVmNIvB92RwPbD0Gy3TpAS0rBpkl3apHVyuyOYqCG'
    access_token = '442617689-erRBZvyQCeElTcJ3ZHHKoVh2kx1xy3vVjam22kpe'
    access_token_secret = '5edIDZdDdUGDpnHfZbuocdZKp9gb0NPQxtNeqcxmJNafq'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    tpy = tweepy.API(auth)
    print('Lemantizer requires Wordnet')
    nltk.download('wordnet')
    return tpy





def getText(data):
    document={}
    document = data[['text']]
    document['index'] = document.index
    tex = []
    for text in document['text']:
        text = re.sub('\s([@][\w_]+)', '', text)
        tex.append(text)
    document['text']=tex
    final_doc = pd.DataFrame()
    final_doc['words'] = document['text'].map(preprocess)
    return LDA(final_doc['words'])


def getUserMentions(data):
    a = data['entities'].apply(lambda x: [a['screen_name'] for a in x['user_mentions']])
    return LDA(pd.DataFrame(a)['entities'])


def getHashtags(data):
    a = data['entities'].apply(lambda x: [a['text'] for a in x['hashtags']])
#     print(pd.DataFrame(a)['entities'])
    return LDA(pd.DataFrame(a)['entities'])


izer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return izer.lemmatize(text, pos='v')

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token[0]!='@':
            result.append(lemmatize_stemming(token))
    return result


# In[225]:


def topic_modeling(bow_corpus,dictionary): 
    from gensim import corpora, models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=3, id2word=dictionary, passes=2, workers=4)
    topics = {}
    for idx, topic in lda_model_tfidf.print_topics(-1):
        val=re.findall("(([01]\.[0-9]{3,})\*\"([a-zA-Z]+)\")", topic)
        topics.update({tup[2]:float(tup[1]) for tup in val})
    return topics


# In[226]:


def LDA(final_doc):
    dictionary = gensim.corpora.Dictionary(final_doc)
    bow_corpus = [dictionary.doc2bow(doc) for doc in final_doc]
#   print(len(bow_corpus[0]))
    topics={}
    if (len(bow_corpus[0])>0) :
        topics=topic_modeling(bow_corpus,dictionary)    
    return topics


# In[214]:


def connectNeo4j():
    driver= GraphDatabase.driver("bolt://localhost:11001", auth=("neo4j", "password"))
    return driver


# In[215]:


def add_edge(tx, node1, node2, edge_weight, label1, label2, rel):
    result = tx.run("MERGE (h:" + label1 + " {name: $node1})" "MERGE (t:" + label2 + "{name: $node2}) MERGE (h)-[:" + rel + "{tf_score : $edge_weight}]->(t)",
           node1=node1, node2=node2, edge_weight=edge_weight)


# In[216]:


def delAlledges(tx):
    result = tx.run("MATCH (h)-[r]-(t) DELETE r")
    result = tx.run(" MATCH (n) DELETE n")


# In[217]:


def del_edge(tx, node1, node2, edge_weight, label1, label2, rel):
    result = tx.run("MATCH (h:" + label1 +" {name: $node1})-[r:" + rel +" {tf_score : $edge_weight}]->(t:"+label2 +"{name: $node2}) DELETE r",
           node1=node1, node2=node2, edge_weight=edge_weight)


# In[ ]:


# In[ ]:





# In[ ]:


# temp=[]
# for i, tweet in enumerate(tweepy.Cursor(api.search, q="NourishEveryDog", lang='en', rpp=50).items(50)):
#     temp.append(tweet._json)
# data = pd.DataFrame(data = temp)
# data=data[['created_at', 'text', 'entities', 'retweet_count']]
# topics = getText(data)
# hashtags = getHashtags(data)
# usermentions = getUserMentions(data)
# for topic,score in topics.items(): 
#     print (topic,score)
# print('\n')
# for topic,score in hashtags.items(): 
#     print (topic,score)
# print('\n') 
# for topic,score in usermentions.items(): 
#     print (topic,score)
# print('\n')

# driver=connectNeo4j()
# print("Driver = ",driver)
# driver=connectNeo4j()
# session=driver.session()
# session.write_transaction(delAlledges)
# for topic,score in usermentions.items():
#     session.write_transaction(add_edge,"NourishEveryDog",topic,score,"hashtag","topic", "HAS_USER")


# In[ ]:





# In[ ]:





# In[ ]:





# In[222]:

def extract(session,hashtag,api):
    print (hashtag+'\n')
    temp=[]
    for i, tweet in enumerate(tweepy.Cursor(api.search, q=hashtag, lang='en', rpp=50).items(50)):
        temp.append(tweet._json)
    data = pd.DataFrame(data = temp)
    if(data.empty):
        print('empty')
        return
    data=data.loc[:,['created_at', 'text', 'entities', 'retweet_count']]
    topics = getText(data)
    hashtags = getHashtags(data)
    usermentions = getUserMentions(data)
#     for topic,score in topics.items(): 
#         print (topic,score)
#     for topic,score in hashtags.items(): 
#         print (topic,score)
    for topic,score in hashtags.items():
        session.write_transaction(add_edge,hashtag,topic,score,"hashtag","tag", "HAS_TAG")
    for topic,score in topics.items():
        session.write_transaction(add_edge,hashtag,topic,score,"hashtag","topic", "HAS_TOPIC")
    for topic,score in usermentions.items():
        session.write_transaction(add_edge,hashtag,topic,score,"hashtag","user", "HAS_USER")
    return topics, hashtags, usermentions



# In[223]:


# def explore(word):
#     session=driver.session()
#     session.write_transaction(delAlledges)
#     topics = extract(session,word)
#     for names in topics.keys():
#         extract(session,names)


# In[224]:


# explore("demonetization")


# In[ ]:


# temp=[]
# for i, tweet in enumerate(tweepy.Cursor(api.search, "#ayodhyaverdict", lang='en', rpp=50).items(50)):
#     temp.append(tweet._json)
# data = pd.DataFrame(data = temp)
# data=data[['created_at', 'text', 'entities', 'retweet_count']]


# In[228]:





# In[ ]:





# In[ ]:





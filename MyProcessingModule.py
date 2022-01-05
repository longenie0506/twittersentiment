import re
import string
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
from textblob import TextBlob
import tensorflow as tf

import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

def greeting(name):
    return name

def preprocessing(tweets,entity='Facebook'):
    # Word segmentation
    tweets_token = word_tokenize(tweets)
    # Remove stopword and Lemmatizing
    lemmatiser = WordNetLemmatizer()
    stop_english = stopwords.words('english')
    sentence = [lemmatiser.lemmatize(word) for word in tweets_token if word not in (stop_english) and (word.isalpha())]
    # Dimensinalize by tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokens = tokenizer.texts_to_matrix([sentence])
    # Entity extraction
    entities = unique_entity()
    seri = pd.Series(0, index=entities)
    seri[entity] = 1
    # Pretrained model features
    polar,subject = pretrain_analyzer(tweets)
    # Finish dataform
    data = np.append(tokens,seri)
    data = np.append(data,[polar,subject])
    data = data.reshape(1,1034)
    return data

def unique_entity():
    entity = ['Nvidia','Dota2','CallOfDuty','RedDeadRedemption(RDR)','HomeDepot','johnson&johnson','Hearthstone','FIFA','TomClancysGhostRecon','Cyberpunk2077','Xbox(Xseries)','TomClancysRainbowSix','GrandTheftAuto(GTA)','Borderlands','CallOfDutyBlackopsColdWar','Facebook','MaddenNFL','CS-GO','Microsoft','Verizon','PlayStation5(PS5)','Fortnite','LeagueOfLegends','Amazon','Google','NBA2K','WorldOfCraft','ApexLegends','AssassinsCreed','PlayerUnknownsBattlegrounds(PUBG)','Battlefield','Overwatch']
    return sorted(entity)

def pretrain_analyzer(tweet):
    polarity = TextBlob(tweet).sentiment[0]
    subjectivity = TextBlob(tweet).sentiment[1]
    return polarity,subjectivity

def getlabel():
    with open('labelencoder.pickle', 'rb') as handle:
        labelencoder = pickle.load(handle)
    return labelencoder
import sys
import os
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk


def load_data(path):
    dataset = open(path, "r").read().split(",\r\n")
    print path + " loaded in memory"
    return dataset

def extract_relevant_data(dataSet):
    loa = {}
    for i,data in enumerate(dataSet):
        data = data.split(",")
        typeOfAuthor = data[2]
        if typeOfAuthor not in loa:
            loa[typeOfAuthor] = []
        try:
            loa[typeOfAuthor].append([float(data[3]), data[6]])
        except ValueError:
            pass
    x = loa.keys()
    for key in x:
        print key, " ---- ", len(loa[key])
    data = {"m":{}, "f":{}}
    data["m"]["data"] = loa.pop("male", None)
    data["f"]["data"] = loa.pop("female", None)
    print "relevant data extracted"
    return data

def count_text_params(minLen, maxLen, score, text_li):
    cnt = 0
    for text in text_li:
        confScore = float(text[0])
        textLen = float(len(text[1]))
        if confScore >= score and textLen <= maxLen and textLen > minLen:
            cnt += 1
    return cnt

def clean_data(text_li, num):
    temp_list = []
    cnt = 0
    for i in text_li:
        text = i[1]
        text = text.strip()
        text = re.split(':|\"|\s|(?<!\d)[,.]|[,.](?!\d)', text)
        temp_li = []
        for j in range(len(text)):
            if len(text[j]) <= 3:
                continue
            if text[j][0] == " ":
                continue
            if text[j][0] == '@' or text[j][0] == '#':
                continue
            if len(text[j]) >= 4 and text[j][0:4] == "http":
                continue
            temp_li.append(text[j])
        temp_li = " ".join(temp_li)
        temp_list.append(temp_li)
        cnt += 1
        if cnt == num:
            break
    return temp_list

def load_stop_words(path):
    stopWords = open(path, "r").read().strip().split("\n")
    return stopWords
    print "stopWords loaded in memory."
    stop_words = {}
    for w in stopWords:
        stop_words[w] = True
    return stop_words

def preprocess(text):
    text = text.strip('\'"')
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+','', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', text)
    text = text.lower()
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    text = rpt_regex.sub(r"\1\1", text)
    return text

def stem(text, flag):
    stemmer = nltk.stem.PorterStemmer()
    text_stem = []
    text = text.lower()
    words = text.split()
    for i,w in enumerate(words):
        if flag == 1 and w in stop_words:
            continue
        try:
            p = stemmer.stem(w)
        except:
            continue
        text_stem.append(p)
    text_stem = ' '.join(text_stem)
    return text_stem

stopWords = load_stop_words("../data/stopWords.txt")

import time
import os
import re
import pickle
from collections import Counter

import pandas as pd

from bs4 import BeautifulSoup as bs4

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def kelly_pickler(filename, dump=None):
    if dump:
        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def weighter(filename=None):
    if filename:
        d = kelly_pickler(filename)
        return d

    else:
        df = pd.read_csv("../data/share_repurchase_paragraphs.csv")
        train = df.paragraph_text.tolist()

        lem = WordNetLemmatizer()

        for i in range(len(train)):
            arr = [lem.lemmatize(w, pos='v') for w in word_tokenize(train[i].lower())]
            train[i] = [w for w in arr if w not in stopwords.words('english') and len(w) > 2]

        d = []
        for collec in train:
            d.extend(collec)

        counts = Counter(d)
        counts = dict(counts.most_common(15))

        counts['repurchase'] += 3000

        kelly_pickler("counts.pkl", counts)

        return counts


def new_files(direc):
    # find new files based on filename.pkl's keys
    contents = os.listdir(direc)
    if os.path.exists("paras.pkl"):
        d = kelly_pickler("paras.pkl")
    else:
        d = {}
    files = d.keys()

    missing = []

    for f in contents:
        if f not in files:
            missing.append(f)

    return missing


def runner(contents, directory):
    tokens = []
    paras = []

    dnl = re.compile(r'\n{2,}')
    nl = re.compile(r'([^\n])\n([^\n])')

    start = time.time()

    lem = WordNetLemmatizer()

    for i in range(len(contents)):

        print(i + 1)

        with open(directory + contents[i], "rb") as test:
            test_text = test.read()

        soup = bs4(test_text, 'html.parser')
        soup.find('head').extract()
        text = soup.get_text()

        text = re.sub(nl, r'\1 \2', text)
        text = re.sub(dnl, r'\n', text)

        text = text.split('\n')

        t = []
        p = []

        for paragraph in text:
            if not paragraph:
                continue
            p.append(paragraph)
            tokes = [lem.lemmatize(w, pos='v') \
                for w in word_tokenize(paragraph.lower()) if w.isalpha() and len(w) > 2]
            tokes = [t for t in tokes if t not in stopwords.words('english')]
            t.append(tokes)

        tokens.append(t)
        paras.append(p)

    print(time.time() - start)

    return tokens, paras


def pullfiles(tokens, paras, missing, files=dict()):
    for i in range(len(tokens)):
        v = {}
        for index, para in enumerate(tokens[i]):
            total = 0
            for word in para:
                total += counts.get(word, 0)

            v[index] = total
        files[missing[i]] = paras[i][sorted([i for i in v.items()], key=lambda x: x[1], reverse=True)[0][0]]

    return files


if __name__ == "__main__":

    # comment out after first run through
    # counts = weighter()
    #
    # nf = new_files("../uploads/")
    #
    # tokens, paras = runner(nf, "../uploads/")
    #
    # pk = pullfiles(tokens, paras, nf)
    #
    # kelly_pickler("paras.pkl", pk)

    # uncomment after first run through
    counts = weighter("counts.pkl")

    nf = new_files("../uploads/")

    tokens, paras = runner(nf, "../uploads/")

    pk = kelly_pickler("paras.pkl")

    pk = pullfiles(tokens, paras, nf, pk)

    kelly_pickler("paras.pkl", pk)

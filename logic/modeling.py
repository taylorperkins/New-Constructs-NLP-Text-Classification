# -*- coding: utf-8 -*-
import os
import time
import csv

from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup as bs4

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # pos='v'

from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel

from collections import Counter


def print_class(func):
    """Decorator used to print the name of the function being called, and also serves to print out how long the func
     took to run

    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        print(func.__name__)
        t1 = time.time()
        val = func(*args, **kwargs)
        print(f"\tTook {time.time() - t1} secs\n")

        return val

    return wrapper


class NewConstructs:
    """NewConstructs modeling class"""
    def __init__(self, category, weighted=None, required_field_score=3):
        """
        :param category: `data_key_friendly_name` category from the share_repurchase_paragraphs.csv
        :param weighted: dict() {word: weight} pairs
        """
        self._lemmatizer = WordNetLemmatizer()
        self._token_dictionary = Dictionary()

        self._category = category

        self._weighted = weighted
        self._required_field_score = required_field_score

    @print_class
    def _create_corpus_from_tokens(self, tokens):
        return [(key, self._token_dictionary.doc2bow(document=doc)) for key, doc in tokens.items()]

    @print_class
    def _tokenize_doc(self, paragraphs, required_words=None):
        """Processing method over all paragraphs passed in. If there are required words present, we exclude all
        paragraphs that dont have those.

        Steps to processing:
            1. Lowercase all words
            2. Keep only alphabetical words
            3. Skip all non-english stopwords
            4. Lemmatize the word
            5. If required words are present, we keep track on the overall paragraph to determine if the paragraph
               contains at least one required word

        :param paragraphs: list() paragraphs to process into usable tokens
        :param required_words: list() words to help exclude some paragraphs we know not to be important
        :return: list() of lists
        """
        tokenized = dict()

        for ind, doc in enumerate(paragraphs):
            lowercased = [word.lower() for word in word_tokenize(doc) if word.isalpha()]

            required_met = set()
            tokenized_doc = list()
            for word in lowercased:
                if word not in stopwords.words('english'):
                    lem = self._lemmatizer.lemmatize(word)

                    if required_words is not None:
                        if lem in required_words:
                            required_met.add(lem)

                    tokenized_doc.append(lem)

            allowed = True if required_words is None else len(required_met) == len(required_words)

            if allowed:
                tokenized[ind] = tokenized_doc

        return tokenized

    @print_class
    def _get_training_paragraphs(self, df):
        """Iterates over df, and returns all paragraphs where data_key_friendly_name matches self._category

        :param df: pd.DataFrame()
        :return: list()
        """
        paragraphs = list()

        for _, row in df[df.data_key_friendly_name == self._category].iterrows():
            paragraphs.append(row.paragraph_text)

        return paragraphs

    @staticmethod
    @print_class
    def _create_training_weights(tokens):
        """Creates a list of normalized weights to apply to the tf-idf model once it has been determined

        :param tokens: list of lists
        :return: dict() {word: weight}
            0 <= word <= 1
        """
        # create count of all words in training set
        counts = Counter()
        for doc in tokens:
            for word in doc:
                counts[word] += 1

        # Grab the min and max counts in the set
        min_counts = counts.most_common()[-1]
        max_counts = counts.most_common(1)[0]

        min_max = min_counts[1], max_counts[1]
        diff = min_max[1] - min_max[0]

        return {word: ((count - min_max[0]) / diff) for word, count in counts.items()}

    @print_class
    def _read_HTML(self, accession_number):
        """read in an HTML file from the nc_training_filings directory.

        :param index:
        :return:
        """
        directory = "./data/nc_training_filings/"
        contents = os.listdir(directory)

        with open('./data/nc_training_filings/' + accession_number, 'r', encoding='utf-8') as file:
            test_text = file.read()

        soup = bs4(test_text, 'html.parser')
        soup.find('head').extract()
        t = soup.get_text()

        return [p.replace('\n', ' ').strip() for p in t.split('\n\n') if p.strip()]

    @staticmethod
    @print_class
    def _get_intersections(tokens):
        """Find all unique processed words from the training paragraphs.

        :param tokens: list of lists
        :return:
        """
        intersections = set()
        for token in tokens:
            if not intersections:
                intersections = set(token)
            else:
                intersections = intersections.intersection(set(token))

        return list(intersections)

    @print_class
    def _score_paragraphs(self, tfidf_model, corpus, weights, required_words):
        """Responsible for applying scores to specific paragraphs.

        Each document takes into consideration the tf-idf applied over the tokens, the weights created based on
        term frequency in the training set, and the required words determined by intersection of all tokens within
        the training set.

        :param tfidf_model: TfidfModel()
        :param corpus: list() of tuples --> original paragraph index, tokens
        :param weights: dict() word --> weight
        :param required_words: list()
        :return: dict()
        """
        p_scores = dict()

        for p_ind, doc in corpus:
            tfidf = tfidf_model[doc]

            scores = list()
            for dict_ind, tfidf_score in tfidf:
                word = self._token_dictionary[dict_ind]

                if required_words and word in required_words:
                    scores.append(tfidf_score * self._required_field_score)
                elif self._weighted is not None and word in self._weighted:
                    scores.append(tfidf_score * (1 + weights[word]) * self._weighted[word])
                else:
                    if word in weights:
                        scores.append(tfidf_score * (1 + weights[word]))
                    else:
                        scores.append(tfidf_score)

            p_scores[p_ind] = sum(scores)
        return p_scores

    @print_class
    def _highlight_doc(self, scored_paragraphs, test_paragraphs, required_words):
        """Processing method over all paragraphs passed in. If there are required words present, we exclude all
        paragraphs that dont have those.

        Steps to processing:
            1. Lowercase all words
            2. Keep only alphabetical words
            3. Skip all english stopwords
            4. Lemmatize the word
            5. If required words are present, we keep track on the overall paragraph to determine if the paragraph
               contains at least one required word

        :param paragraphs: list() paragraphs to process into usable tokens
        :param required_words: list() words to help exclude some paragraphs we know not to be important
        :return: list() of lists
        """
        highlighted = defaultdict(list)

        for ind, score in scored_paragraphs.items():
            doc_corpus = sent_tokenize(test_paragraphs[ind])

            tokenized = self._tokenize_doc(doc_corpus, required_words=required_words)

            if tokenized:
                for i, tokens in tokenized.items():
                    highlighted[ind].append((score, doc_corpus[i]))

        return highlighted

    def main(self, file_index):
        # Read in the training data
        example_submission = pd.read_csv('./data/share_repurchase_paragraphs.csv')
        print(f"Unique key friendly names: {[name for name  in example_submission.data_key_friendly_name.unique()]}")

        # subset and pre-process the tokens to be used
        train_paragraphs = self._get_training_paragraphs(example_submission)
        train_tokenized = self._tokenize_doc(train_paragraphs)
        # add the tokens to the dictionary to keep track if what words we have found
        self._token_dictionary.add_documents(train_tokenized.values())

        weights = self._create_training_weights(tokens=train_tokenized.values())

        train_intersections = self._get_intersections(train_tokenized.values())

        field_names = ['accession_number', 'score', 'text', 'paragraph_text']

        directory = "./data/nc_training_filings/"
        contents = os.listdir(directory)

        with open('test.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()

            for c_ind, accession_number in enumerate(contents):
                print(f"\n\nRunning process for {accession_number}. T-minus {len(contents) - c_ind}")

                instances = list()

                test_paragraphs = self._read_HTML(accession_number=accession_number)

                test_tokenized = self._tokenize_doc(test_paragraphs, required_words=train_intersections)
                # add the tokens to the dictionary to keep track if what words we have found
                self._token_dictionary.add_documents(test_tokenized.values())

                corpus = self._create_corpus_from_tokens(tokens=test_tokenized)

                tfidf_model = TfidfModel(corpus=[doc[1] for doc in corpus])

                scored_paragraphs = self._score_paragraphs(tfidf_model, corpus, weights, required_words=train_intersections)

                highlighted = self._highlight_doc(scored_paragraphs, test_paragraphs, train_intersections)

                for ind, sentences in highlighted.items():
                    for score, highlight in sentences:
                        instances.append({
                            'accession_number': accession_number,
                            'score': score,
                            'text': highlight,
                            'paragraph_text': test_paragraphs[ind]
                        })

                if not highlighted:
                    instances.append({
                        'accession_number': accession_number,
                        'score': 0,
                        'text': '',
                        'paragraph_text': ''
                    })

                try:
                    writer.writerows(instances)
                except UnicodeEncodeError as e:
                    instances = [{
                        'accession_number': accession_number,
                        'score': 0,
                        'text': 'UnicodeEncodeError',
                        'paragraph_text': ''
                    }]
                    writer.writerows(instances)

                except Exception as e:
                    instances = [{
                        'accession_number': accession_number,
                        'score': 0,
                        'text': e.__class__,
                        'paragraph_text': ''
                    }]
                    writer.writerows(instances)


if __name__ == '__main__':
    t1 = time.time()
    new_constructs = NewConstructs(
        category='Share Repurchase Authorization',
        weighted=dict(
            shares=2,
            million=2.3,
            billion=2.3
        )
    )

    new_constructs.main(file_index=0)

    print(f"Done. Ran in {time.time() - t1} seconds.")

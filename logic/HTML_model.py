# -*- coding: utf-8 -*-
import os
import csv

from collections import defaultdict, OrderedDict, Counter

from bs4 import BeautifulSoup as bs4

from nltk.tokenize import sent_tokenize

from gensim.models import TfidfModel

from utils import MatchGroups

from logic.base_model import BaseModel
from logic.train_model import TrainModel


class NewConstructs(BaseModel):
    _path = '../data/nc_validation_filings/'

    _cat_group_match = {
        'Share Repurchase Authorization Date': MatchGroups.match_date,
        'Share Repurchase Count': MatchGroups.match_counts,
        'Share Repurchase Authorization': MatchGroups.match_money,
        'Share Repurchase Intention': MatchGroups.match_money,
        'Amount Spent on Share Repurchases': MatchGroups.match_money,
        'Share Repurchase Utilization': MatchGroups.match_money
    }

    """NewConstructs modeling class"""
    def __init__(self, weighted=None, required_field_score=3):
        """
        :param weighted: dict() {word: weight} pairs
        """
        super(NewConstructs, self).__init__()

        self._weighted = weighted
        self._required_field_score = required_field_score

    def _create_corpus_from_tokens(self, tokens):
        return [(key, self._token_dictionary.doc2bow(document=doc)) for key, doc in tokens.items()]

    def _read_HTML(self, accession_number):
        """read in an HTML file from the nc_training_filings directory.

        :param accession_number:
        :return:
        """
        # directory = "../data/nc_training_filings/"
        # contents = os.listdir(directory)

        with open(self._path + accession_number, 'r', encoding='utf-8') as file:
            test_text = file.read()

        soup = bs4(test_text, 'html.parser')
        soup.find('head').extract()
        t = soup.get_text()

        return [p.replace('\n', ' ').strip() for p in t.split('\n\n') if p.strip()]

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

    def _highlight_doc(self, scored_paragraphs, test_paragraphs, required_words, category):
        """Processing method over all paragraphs passed in. If there are required words present, we exclude all
        paragraphs that dont have those.

        Steps to processing:
            1. Lowercase all words
            2. Keep only alphabetical words
            3. Skip all english stopwords
            4. Lemmatize the word
            5. If required words are present, we keep track on the overall paragraph to determine if the paragraph
               contains at least one required word

        :param scored_paragraphs: list() paragraphs to process into usable tokens
        :param test_paragraphs:
        :param required_words: list() words to help exclude some paragraphs we know not to be important
        :param category:
        :return: list() of lists
        """
        highlighted = defaultdict(list)

        for ind, score in scored_paragraphs.items():
            doc_corpus = sent_tokenize(test_paragraphs[ind])

            tokenized = self._tokenize_doc(
                doc_corpus,
                tokenized_categories_required_words={category: required_words}
            )

            if tokenized:
                for cat, tokens in tokenized.items():
                    for i in tokens.keys():
                        if self._cat_group_match[category](doc_corpus[i]):
                            highlighted[ind].append((score, doc_corpus[i]))

        return highlighted

    @staticmethod
    def _write_rows_to_csv(csv_writer, highlighted, cat, accession_number, test_paragraphs):
        for ind, sentences in highlighted.items():
            for score, highlight in sentences:
                try:
                    csv_writer.writerow({
                        'category': cat,
                        'accession_number': accession_number,
                        'score': score,
                        'text': highlight,
                        'paragraph_text': test_paragraphs[ind]
                    })

                except UnicodeEncodeError as e:
                    pass

                except Exception as e:
                    pass

    def main(self):
        tm = TrainModel(
            train_path='./data/share_repurchase_paragraphs.csv',
            pkl_path='./data/share_repurchase_paragraphs.pkl'
        )

        tokens_weights = tm.read()
        if tokens_weights is None:
            tokens_weights = tm.train()

        with open('test.csv', 'w') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['accession_number', 'score', 'text', 'paragraph_text', 'category']
            )
            writer.writeheader()

            contents = os.listdir(self._path)
            for c_ind, accession_number in enumerate(contents):
                print(f"\n\nRunning process for {accession_number}. T-minus {len(contents) - c_ind}")

                test_paragraphs = self._read_HTML(accession_number=accession_number)

                test_tokenized = self._tokenize_doc(
                    test_paragraphs,
                    tokenized_categories_required_words={cat: val['intersections'] for cat, val in tokens_weights.items()}
                )
                # add the tokens to the dictionary to keep track if what words we have found
                ind_dict_count = dict()
                for cat, tokens_dict in test_tokenized.items():
                    print(cat)

                    ind_count = list()
                    for ind, tokens in tokens_dict.items():
                        if ind not in ind_dict_count:
                            ind_dict_count[ind] = 1
                            ind_count.append(ind)

                    self._token_dictionary.add_documents([tokens_dict[ind] for ind in ind_count])

                    corpus = self._create_corpus_from_tokens(tokens=tokens_dict)

                    tfidf_model = TfidfModel(corpus=[doc[1] for doc in corpus])

                    scored_paragraphs = self._score_paragraphs(
                        tfidf_model=tfidf_model,
                        corpus=corpus,
                        weights=tokens_weights[cat]['weights'],
                        required_words=tokens_weights[cat]['intersections']
                    )

                    highlighted = self._highlight_doc(
                        scored_paragraphs=scored_paragraphs,
                        test_paragraphs=test_paragraphs,
                        required_words=tokens_weights[cat]['intersections'],
                        category=cat
                    )

                    # This is a little crazy.. But it's just bringing back the top 5 results based on score
                    highlighted = dict(list(OrderedDict(sorted(highlighted.items(), key=lambda x: x[1][0][0], reverse=True)).items())[:5])

                    self._write_rows_to_csv(
                        csv_writer=writer,
                        highlighted=highlighted,
                        cat=cat,
                        accession_number=accession_number,
                        test_paragraphs=test_paragraphs
                    )


if __name__ == '__main__':
    new_constructs = NewConstructs(
        weighted=dict(
            shares=2,
            million=2.3,
            billion=2.3
        )
    )

    new_constructs.main()


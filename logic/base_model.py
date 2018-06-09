from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # pos='v'

from gensim.corpora.dictionary import Dictionary


class BaseModel(object):
    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self._token_dictionary = Dictionary()

    def _tokenize_doc(self, paragraphs, tokenized_categories_required_words=None):
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
        tokenized_per_cat = defaultdict(dict)

        for ind, doc in enumerate(paragraphs):
            lowercased = [word.lower() for word in word_tokenize(doc) if word.isalpha()]

            required_met = defaultdict(set)
            tokenized_doc = list()
            for word in lowercased:
                if word not in stopwords.words('english'):
                    lem = self._lemmatizer.lemmatize(word)

                    if tokenized_categories_required_words is not None:
                        for cat, required_words in tokenized_categories_required_words.items():
                            if lem in required_words:
                                required_met[cat].add(lem)

                    tokenized_doc.append(lem)

            if tokenized_categories_required_words is None:
                tokenized[ind] = tokenized_doc

            else:
                for key, val in required_met.items():
                    if len(val) == len(tokenized_categories_required_words[key]):
                        tokenized_per_cat[key][ind] = tokenized_doc

        if tokenized_categories_required_words:
            return tokenized_per_cat
        return tokenized

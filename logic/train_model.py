import pickle
from collections import defaultdict, Counter

import pandas as pd

from logic.base_model import BaseModel


class IncorrectExtensionError(Exception):
    pass


class UnexpectedColumnError(Exception):
    pass


class TrainModel(BaseModel):
    _TRAIN_EXTENSION = 'csv'
    _PKL_EXT = 'pkl'

    def __init__(self, train_path, pkl_path):
        super(TrainModel, self).__init__()

        self._validate_path(train_path, self._TRAIN_EXTENSION)
        self._validate_path(pkl_path, self._PKL_EXT)

        self._train_path = train_path
        self._pkl_path = pkl_path

        self._weights = None

    def __call__(self, *args, **kwargs):
        if self._pkl_path is not None:
            pass
        pass

    def _validate_path(self, path, ext):
        if not path.split('.')[-1] == ext:
            raise IncorrectExtensionError(f"{self._pkl_path} does not have correct extension. Expected {ext}")

    def read(self):
        """Read from the pkl_path. If there are contents, return it. No training needed.

        :return:
        """
        try:
            with open(self._pkl_path, "rb") as f:
                pkl = pickle.load(f)
        except IOError:  # File doesn't exist
            return None
        else:
            return pkl

    def _write(self, train):
        with open(self._pkl_path, "wb") as f:
            pickle.dump(train, f)

    def train(self):
        """Pkl path returned empty, so we need to bring in the contents from the train path, and create a new model
        to be saved off

        :return:
        """
        train_set = pd.read_csv(self._train_path)

        train_tokenized = self.get_tokens(train_set)

        tokens_weights = defaultdict(dict)

        # add the tokens to the dictionary to keep track if what words we have found
        for cat, tokens in train_tokenized:
            self._token_dictionary.add_documents(tokens.values())

            tokens_weights[cat]['weights'] = self._create_training_weights(tokens=tokens.values())

            tokens_weights[cat]['intersections'] = self._get_intersections(tokens.values())

            print(f"Intersections for {cat}: {tokens_weights[cat]['intersections']}")

        self._write(tokens_weights)

        return tokens_weights

    def get_tokens(self, train_set):
        return (
            (cat, self._tokenize_doc(self._get_training_paragraphs(train_set, cat)))
            for cat in self._get_train_categories(train_set)
        )

    @staticmethod
    def _get_training_paragraphs(train_set, category):
        """Iterates over df, and returns all paragraphs where data_key_friendly_name matches self._category

        :param train_set: pd.DataFrame()
        :return: list()
        """
        return (
            row.paragraph_text for _, row in train_set[train_set.data_key_friendly_name == category].iterrows()
        )

    @staticmethod
    def _get_train_categories(train_set):
        if 'data_key_friendly_name' not in train_set.columns:
            raise UnexpectedColumnError(
                f"'data_key_friendly_name' not found in columns, got {train_set.columns} instead")

        return (
            cat for cat in train_set.data_key_friendly_name.unique().tolist()
            if cat != 'Unknown Share Repurchase Data'
        )

    @staticmethod
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

    @staticmethod
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

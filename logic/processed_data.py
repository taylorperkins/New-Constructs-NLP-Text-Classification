import pickle

from collections import defaultdict


class DataStore(object):
    def __init__(self, path):
        self._store = path

        self.data = None

    def __call__(self, *args, **kwargs):
        try:
            self.get()
        except IOError:
            self.create()
        finally:
            return self

    def create(self):
        self.data = defaultdict(dict)

    def write(self):
        if self.data:
            with open(self._store, "wb") as f:
                pickle.dump(self.data, f)

            print("Successfully loaded data into store.")

    def get(self):
        try:
            with open(self._store, "rb") as f:
                self.data = pickle.load(f)

        except IOError:  # file doesnt exist
            raise IOError("Data Store must be created before you can reference it. Try <store>.create()")

    def get_accession_record(self, ticker, acc):
        return self.data.get(ticker, {}).get(acc)

    def update_accession(self, ticker, acc, updates):
        if updates:
            if ticker not in self.data:
                self.data[ticker] = {acc: updates}
            else:
                if acc not in self.data[ticker]:
                    self.data[ticker][acc] = updates
                else:
                    self.data[ticker][acc].update(updates)

            self.write()

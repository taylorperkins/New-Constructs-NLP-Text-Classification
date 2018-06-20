import re

from config import ALLOWED_EXTENSIONS


def time_func(func):
    """Decorator used to print the name of the function being called, and also serves to print out how long the func
     took to run

    :param func:
    :return:
    """
    import time

    def wrapper(*args, **kwargs):
        print(func.__name__)
        t1 = time.time()
        val = func(*args, **kwargs)
        print(f"\tTook {time.time() - t1} secs\n")

        return val
    return wrapper


def assert_db(f):
    """Decorator used to print the name of the function being called, and also serves to print out how long the func
     took to run

    :return:
    """
    from flask import current_app, redirect
    from functools import wraps

    @wraps(f)
    def wrapped(*args, **kwargs):
        if not hasattr(current_app, 'db'):
            return redirect('/upload/')

        return f(*args, **kwargs)
    return wrapped


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_nav_menu_options(_db):
    nav = {}
    for ticker in _db.keys():
        nav[ticker] = {acc: _db[ticker][acc].keys() for acc in _db[ticker].keys()}

    return nav


class MatchGroups:
    # Notice the `-` exclusion for years at the end
    DATE_MATCH = re.compile(r"([A-Z][a-z]+ \d{1,2},? (\b(19|20)\d{2}\b))|(\d{1,2} [A-Z][a-z]+,? (\b(19|20)\d{2}\b)|\b(19|20)\d{2}\b(?!-))", flags=re.IGNORECASE)
    MONEY_MATCH = re.compile(r"(\$[0-9.,]+\s?(per)?\s?([mb]illion|share)[s]?)", flags=re.IGNORECASE)
    # note that the dollar amount is optional
    COUNTS_MATCH = re.compile(r"(\$?[0-9.,]+\s?(per)?\s?([mb]illion|share)[s]?)", flags=re.IGNORECASE)

    @classmethod
    def match_date(cls, text):
        return True if cls.DATE_MATCH.findall(text) else False

    @classmethod
    def match_money(cls, sent):
        return True if cls.MONEY_MATCH.findall(sent) else False

    @classmethod
    def match_counts(cls, sent):
        return True if cls.COUNTS_MATCH.findall(sent) else False


CAT_GROUP_MATCH = {
    'Share Repurchase Authorization Date': {
        'regex': MatchGroups.match_date,
        'class': 'date'
    },
    'Share Repurchase Count': {
        'regex': MatchGroups.match_counts,
        'class': 'counts'
    },
    'Share Repurchase Authorization': {
        'regex': MatchGroups.match_money,
        'class': 'money'
    },
    'Share Repurchase Intention': {
        'regex': MatchGroups.match_money,
        'class': 'money'
    },
    'Amount Spent on Share Repurchases': {
        'regex': MatchGroups.match_money,
        'class': 'money'
    },
    'Share Repurchase Utilization': {
        'regex': MatchGroups.match_money,
        'class': 'money'
    }
}

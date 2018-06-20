from copy import deepcopy

from utils import MatchGroups


class AddCSSClass(object):
    _keywords = ["share", "repurchase", "authorized"]

    @staticmethod
    def highlight(sent):
        """Wraps text in highlight span"""
        return "<span class='highlight'>" + sent + "</span>"

    @staticmethod
    def replace_matches_with_class(text, _class, regex):
        """Takes some text, a css class, and a regex object and finds all matches in the text, then wraps the matches
        in a span with the appropriate css class.

        :param text: str()
        :param _class: str()
        :param regex:
        :return: str()
        """
        for group in regex.findall(text):
            group = [match.strip() for match in group if match]
            group.sort(key=len)
            if group:
                val = group.pop()
                if _class == 'counts' and '$' in val:
                    continue

                text = text.replace(val, f"<span class='{_class}'>" + val + "</span>")

        return text

    @classmethod
    def add_css_classes(cls, sent, highlight=False):
        """Iterates through each regex and finds the matches in sent. For each match, it wraps the word in the
        sentence with a span and class that's associated with the regex.

        :param sent: str()
        :param highlight: bool()
        :return: str()
        """
        sent_copy = deepcopy(sent)

        sent_copy = cls.highlight(sent_copy) if highlight else sent_copy

        for _class, regex in [
            ('date', MatchGroups.DATE_MATCH),
            ('money', MatchGroups.MONEY_MATCH),
            ('counts', MatchGroups.COUNTS_MATCH)
        ]:
            replaced = cls.replace_matches_with_class(sent_copy, _class=_class, regex=regex)
            if replaced:
                sent_copy = replaced

        return sent_copy

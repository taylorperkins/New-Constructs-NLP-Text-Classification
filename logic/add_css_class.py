from copy import deepcopy

from utils import MatchGroups


class AddCSSClass(object):
    _keywords = ["share", "repurchase", "authorized"]

    @staticmethod
    def highlight(sent):
        return "<span class='highlight'>" + sent + "</span>"

    @staticmethod
    def replace_matches_with_class(text, _class, regex):
        for group in regex.findall(text):
            group = [match.strip() for match in group if match]
            group.sort(key=len)
            if group:
                val = group.pop()
                return text.replace(val, f"<span class='{_class}'>" + val + "</span>")

    @classmethod
    def add_css_classes(cls, sent, highlight=False):
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

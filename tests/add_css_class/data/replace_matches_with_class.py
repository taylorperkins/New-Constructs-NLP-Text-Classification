from utils import MatchGroups

# Config for test data and response examples
"""Setup
test: (
    input --> {
        kwargs
    },
    output
)

"""
config = {
    "date_wraps_standard": (
        "Matches date <span class='date'>December 31, 2016</span>",
        {
            "text": "Matches date December 31, 2016",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_wraps_multiple_standard": (
        "Matches date <span class='date'>December 31, 2016</span>. Here is another <span class='date'>January 1, 2018</span>.",
        {
            "text": "Matches date December 31, 2016. Here is another January 1, 2018.",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_wraps_year_only": (
        "<span class='date'>2012</span>",
        {
            "text": "2012",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_ignores_case": (
        "<span class='date'>december 31, 2012</span>",
        {
            "text": "december 31, 2012",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_ignores_commas_1": (
        "<span class='date'>december 31 2012</span>",
        {
            "text": "december 31 2012",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_reverse_month_and_day": (
        "<span class='date'>31 december, 2012</span>",
        {
            "text": "31 december, 2012",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_ignores_commas_2": (
        "<span class='date'>31 december 2012</span>",
        {
            "text": "31 december 2012",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_does_not_match_below_20_century": (
        "31 december 1812",
        {
            "text": "31 december 1812",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "date_does_not_match_above_21_century": (
        "31 december 2112",
        {
            "text": "31 december 2112",
            "_class": "date",
            "regex": MatchGroups.DATE_MATCH
        }
    ),
    "counts_excludes_dollar_amount": (
        "$10 shares",
        {
            "text": "$10 shares",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_wraps_shares": (
        "<span class='counts'>10 shares</span>",
        {
            "text": "10 shares",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_wraps_millions": (
        "<span class='counts'>10 millions</span>",
        {
            "text": "10 millions",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_wraps_billions": (
        "<span class='counts'>10 billions</span>",
        {
            "text": "10 billions",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_matches_singularity_share": (
        "<span class='counts'>10 share</span>",
        {
            "text": "10 share",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_matches_singularity_million": (
        "<span class='counts'>10 million</span>",
        {
            "text": "10 million",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_matches_singularity_billion": (
        "<span class='counts'>10 billion</span>",
        {
            "text": "10 billion",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_ignores_case": (
        "<span class='counts'>10 BILLION</span>",
        {
            "text": "10 BILLION",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_wraps_with_commas": (
        "<span class='counts'>10,000,000 shares</span>",
        {
            "text": "10,000,000 shares",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "counts_wraps_with_periods": (
        "<span class='counts'>10.5 million</span>",
        {
            "text": "10.5 million",
            "_class": "counts",
            "regex": MatchGroups.COUNTS_MATCH
        }
    ),
    "money_includes_dollar_amount": (
        "<span class='money'>$10 shares</span>",
        {
            "text": "$10 shares",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_does_not_match_without_dollar_amount": (
        "10 shares",
        {
            "text": "10 shares",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_wraps_shares": (
        "<span class='money'>$10 shares</span>",
        {
            "text": "$10 shares",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_wraps_millions": (
        "<span class='money'>$10 millions</span>",
        {
            "text": "$10 millions",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_wraps_billions": (
        "<span class='money'>$10 billions</span>",
        {
            "text": "$10 billions",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_matches_singularity_share": (
        "<span class='money'>$10 share</span>",
        {
            "text": "$10 share",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_matches_singularity_million": (
        "<span class='money'>$10 million</span>",
        {
            "text": "$10 million",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_matches_singularity_billion": (
        "<span class='money'>$10 billion</span>",
        {
            "text": "$10 billion",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_ignores_case": (
        "<span class='money'>$10 BILLION</span>",
        {
            "text": "$10 BILLION",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_wraps_with_commas": (
        "<span class='money'>$10,000,000 shares</span>",
        {
            "text": "$10,000,000 shares",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    ),
    "money_wraps_with_periods": (
        "<span class='money'>$10.5 million</span>",
        {
            "text": "$10.5 million",
            "_class": "money",
            "regex": MatchGroups.MONEY_MATCH
        }
    )
}
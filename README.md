# Text Classification Using NLP - New Constructs Project

## Overview

This repository represents a project I had the opportunity to work on while attending the NSS data science cohort.
At the time, we were learning maching learning concepts, specifically nlp.
The company New Constructs came to us with a data question to both challenge us, and introduce us to a real-world problem that incorporates nlp.
This project is our initial attempt at solving their problem, or at least starting a process to a solution.

#### Requirements
1. For clarity, there won’t necessarily be separate paragraphs for each of those 7 data points we’ve identified (e.g., Share Repurchase Authorization Date, Share Repurchase Authorization, Share Repurchase Intention, etc.) - you could find each of those 7 data points all in a single paragraph depending on the disclosure in the filing. So if it made sense to, you could look at the problem in two stages - first, identifying those paragraphs generally that are about share repurchases, and second, classifying the data inside the paragraph into those 7 categories.
2. For the deliverable, please prepare a csv file with the data points that are included in share_repurchase_paragraphs.csv
3. For errors, we’d prefer to err on the side of a false positive. I think those will also make for easier feedback than negative cases.

#### The Data
We were given a LOT of data for this project, none of which is shown in this repository.
If you wish to know more information surrounding the data involved, please contact me personally.
At a high level, we were given SEC filings for many companies.
Within the filings, companies are disclosing information about their share repurchase intentions.
This information is generally in paragraph form in the notes of the financial statements.
Please see [this page](images/New_Constructs_Share_Repurchase_Data_Collection_Project.pdf) for a better outline on data description.

#### Our Approach

First off, I would like to say that this is a tough problem!!
There are many approaches you can take, and this is a problem that could likely be worked on for a long time.
In this project, I did not aim to give the exact answers to the questions asked, but instead provided a tool that allowed the data analysts to review the most relevant sentences and determine the answer.

The remainder of this README will be talking about the steps I took to achieve this goal.

### Steps

#### Determine Outcome

What I mean by this is.. What is my end goal??
Ultimately, I wanted a way for the user to upload a SEC filing, have the process run in the background, and save the results in the database (in this case, locally as a pkl file).

* Once the process is finished, the ticker, accession number, and categories are easily selectable to view the results.
* Selecting a category, the user should be able to see the most relevant paragraphs associated with the category with the target sentence in yellow.
* Group words should be highlighted, and score of that paragraph should be present.
* You should also be allowed the ability to select previous or future sentences relative to the relevant paragraph to provide context for the text.

Click [here](https://www.youtube.com/watch?v=HlOVcRdVRAE&feature=youtu.be) for a full demonstration on the end result.

#### Create Initial Flask App

Yes, I wanted to create an app for this! It makes sense.
There are several views that are required to make this work.
The upload screen, choosing a ticker, choosing an accession number, choosing a category, then the actual view.
Each one of these views needed html, styling, and code to handle the request.
This can all be found in [new_constructs_app.py](new_constructs_app.py) and `templates/`.

#### Create Logic for Model

This is the bread and butter right??
This particular model is all natural, homegrown, vanilla.
In a general sense, this model holds two things per category.

1. Intersections
2. Weights

The weights are just a normalized representation of word frequency in a category corpus.
Nothing really fancy there.

The interesections are also pretty simplistic..
This value is a list of the words that are in every row associated with a category.
Example being.. The words `['december', 'share']` show up in every row for the Share Repurchase Authorization Date category.

That's it!!!
_for now.._

Check [this](logic/train_model.py) out for the code.

#### Create Logic for Raw HTML

This is actually a little more complicated than training the model itself.
Steps:

* Read in HTML
* Tokenize the HTML. This includes lowercasing words, removing stopwords, lemmatizing, and making sure everything is english. This also filters out the paragraphs that don't include the intersecting words from the training model.
* Add all new words to a words dictionary
* Perform TF-IDF over the remaining paragraphs, or the corpus.
* Score the paragraphs based on the weights of the model for the category
* Pick top 3 paragraphs
* Highlight the paragraphs. This is picking the most relevant sentence from the paragraph
* Wrap the results in the appropriate css classes based on regex grouping (counts, money, or dates)
* Put it in the "datastore" (pkl file)

Base logic for this processing step can be found [here](logic/HTML_model.py)

Logic for wrapping the groups in their appropriate css classes can be found [here](logic/add_css_class.py)

Tests for those groups can be found [here](tests/add_css_class/tests/test_replace_matches_with_class.py) and [here](tests/add_css_class/data/replace_matches_with_class.py)

#### Final Notes

Aside from the processes listed.. I think that's it!
There was a lot of trial and error leading up to the end result.
Reading in the HTML, understanding the problem, determining an appropriate filter, etc.

If I were to continue with this project and work with it a bit more, I would love to incorporate a Logistic Regression model over the target values.
Perhaps a blend!
Overall, this was a fun project.
We had the chance to work with some great data, help solve a 'real-world' problem, and do some nlp!

Please message me for any questions surrounding the data or this project. Thank you!

#### Team
[Justin Rothbart](https://github.com/jroth006) AKA Redbeard Wizard
[Xander Morrison](https://github.com/jxandermorrison) AKA Greek god of regex

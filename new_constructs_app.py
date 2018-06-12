import pickle
import shutil

from flask import Flask, render_template, redirect, request, url_for, current_app

from config import UPLOAD_FOLDER

from utils import CAT_GROUP_MATCH, get_nav_menu_options

from logic.processed_data import DataStore

from logic.upload_file import UploadFileLogic
from logic.HTML_model import NewConstructs
from logic.train_model import TrainModel

app = Flask(__name__)


@app.route('/', methods=['GET'])
def get_ticker():
    if not current_app.db:
        return redirect('/upload/')

    return render_template(
        'body/get_ticker/get_ticker.html',
        tickers=current_app.db.keys()
    )


@app.route('/<ticker>/', methods=['GET'])
def get_accession_from_ticker(ticker):
    if ticker not in current_app.db.keys():
        return redirect('/')

    return render_template(
        'body/get_accession_from_ticker/get_accession_from_ticker.html',
        ticker=ticker,
        accession_numbers=current_app.db[ticker].keys()
    )


@app.route('/<ticker>/<accession_number>/', methods=['GET'])
def get_data_key_friendly_names_from_accession(ticker, accession_number):
    if ticker not in current_app.db.keys() or accession_number not in current_app.db[ticker].keys():
        return redirect('/')

    return render_template(
        'body/get_data_key_friendly_names_from_accession/get_data_key_friendly_names_from_accession.html',
        ticker=ticker,
        accession_number=accession_number,
        data_key_friendly_names=current_app.db[ticker][accession_number].keys()
    )


@app.route('/<ticker>/<accession_number>/<data_key_friendly_name>/', methods=['GET'])
def get_data_key_friendly_name_highlights(ticker, accession_number, data_key_friendly_name):
    if data_key_friendly_name not in current_app.db.get(ticker, {}).get(accession_number, {}).keys():
        return redirect('/')

    return render_template(
        'index.html',
        target=CAT_GROUP_MATCH[data_key_friendly_name]['class'],
        nav_menu=current_app.nav_menu,
        ticker=ticker,
        accession_number=accession_number,
        data_key_friendly_name=data_key_friendly_name,
        accessions={key: value.keys() for key, value in current_app.db[ticker].items()},
        paragraphs=current_app.db[ticker][accession_number][data_key_friendly_name]
    )


@app.route('/upload/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Uploading File")
        upload_file_logic = UploadFileLogic(config=current_app.config)
        redirect_to = upload_file_logic.upload(request=request)
        if redirect_to:
            return redirect(redirect_to)
        else:
            raise Exception('Error uploading html')

    return render_template(
        'body/upload_file/upload_file.html',
        dont_show_upload=True
    )


@app.route('/process_file/<ticker>/<filename>/')
def process_file(ticker, filename):
    print("Beginning the cleaning and modeling of HTML.")
    new_constructs = NewConstructs(
        data_store=DataStore(
            path='./data/processed_html_data_store.pkl'
        ),
        weighted=dict(
            shares=2,
            million=2.3,
            billion=2.3
        ),
        data_path=f"./uploads/{ticker}/"
    )

    try:
        with open('./data/share_repurchase_paragraphs.pkl', "rb") as f:
            tm = pickle.load(f)
    except IOError:  # File doesn't exist
        tm = TrainModel(
            train_path='./data/share_repurchase_paragraphs.csv',
            pkl_path='./data/share_repurchase_paragraphs.pkl'
        )

    if tm.weights is None:
        print("No current weights in training model found. Starting training process.")
        tm.train()

    new_constructs.process_HTML(tm=tm, ticker=ticker, accession_path=filename)

    with app.app_context():
        with open('./data/processed_html_data_store.pkl', "rb") as f:
            app.db = pickle.load(f)
        app.nav_menu = get_nav_menu_options(_db=app.db)

    shutil.rmtree(f"{app.config['UPLOAD_FOLDER']}/{ticker}")

    print("Finished")

    return redirect(url_for(
        "get_data_key_friendly_names_from_accession",
        ticker=ticker,
        accession_number=filename.split(".")[1]
    ))


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    with app.app_context():
        db = DataStore(path='./data/processed_html_data_store.pkl')
        app.db = db().data
        app.nav_menu = get_nav_menu_options(_db=app.db)

    app.run(debug=True, use_debugger=False, use_reloader=False)


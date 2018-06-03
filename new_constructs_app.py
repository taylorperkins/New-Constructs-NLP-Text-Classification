from flask import Flask, render_template, redirect, request, send_from_directory

from config import UPLOAD_FOLDER

from data.ticker_accession_mapping import TEST_MAPPING

from logic.upload_file import UploadFileLogic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def get_ticker():
    return render_template(
        'body/get_ticker/get_ticker.html',
        tickers=TEST_MAPPING.keys()
    )


@app.route('/<ticker>/', methods=['GET'])
def get_accession_from_ticker(ticker):
    if ticker not in TEST_MAPPING.keys():
        return redirect('/')

    return render_template(
        'body/get_accession_from_ticker/get_accession_from_ticker.html',
        ticker=ticker,
        accession_numbers=TEST_MAPPING[ticker].keys()
    )


@app.route('/<ticker>/<accession_number>/', methods=['GET'])
def get_accession_highlights(ticker, accession_number):
    if ticker not in TEST_MAPPING.keys() or accession_number not in TEST_MAPPING[ticker].keys():
        return redirect('/')

    return render_template(
        'index.html',
        ticker=ticker,
        accession_number=accession_number,
        accessions={key: value.keys() for key, value in TEST_MAPPING.items()},
        paragraphs=TEST_MAPPING[ticker][accession_number]
    )


@app.route('/upload/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        upload_file_logic = UploadFileLogic(config=app.config)
        redirect_to = upload_file_logic.upload(request=request)
        if redirect_to:
            redirect(redirect_to)
        else:
            raise Exception('Error uploading html')

    return render_template('body/upload_file/upload_file.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    # app.config['DEBUG'] = True
    # app.config['TESTING'] = True

    app.run(debug=True, use_debugger=False, use_reloader=False)

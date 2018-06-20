import os

from flask import url_for

from werkzeug.utils import secure_filename

from utils import allowed_file


class UploadFileLogic(object):
    """Logic class for taking an HTML uploaded from a flask form, and placing it in a folder"""
    def __init__(self, config):
        self._config = config

    def upload(self, request):
        # check if the post request has the file part
        if 'file' not in request.files:
            print(f"No file part. Redirecting to {request.url}")
            return request.url

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print(f"No file selected. Redirecting to {request.url}")
            return request.url

        if file and allowed_file(file.filename):
            filename_secured = secure_filename(file.filename)

            if not os.path.isdir(os.path.join(self._config['UPLOAD_FOLDER'], request.form['ticker'])):
                os.mkdir(os.path.join(self._config['UPLOAD_FOLDER'], request.form['ticker']))

            file.save(os.path.join(
                f"{self._config['UPLOAD_FOLDER']}/{request.form['ticker']}",
                filename_secured
            ))
            return url_for('process_file', ticker=request.form['ticker'], filename=filename_secured)

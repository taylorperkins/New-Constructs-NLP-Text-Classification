import os

from flask import redirect, url_for

from werkzeug.utils import secure_filename

from utils import allowed_file


class UploadFileLogic(object):
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
            filename = secure_filename(file.filename)
            file.save(os.path.join(self._config['UPLOAD_FOLDER'], '_'.join([request.form['ticker'], filename])))
            return url_for('uploaded_file', filename=filename)

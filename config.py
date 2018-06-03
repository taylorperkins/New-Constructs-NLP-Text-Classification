import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


UPLOAD_FOLDER = os.curdir + '/uploads'
ALLOWED_EXTENSIONS = ['html']

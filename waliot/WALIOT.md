sudo pacman -Syu gcc glib2 mesa cmake python3 libjpeg-turbo
pip3 install -r requirements.txt
pip3 install Flask Flask-WTF

RUN: ./app.py

REQ: ./req.http

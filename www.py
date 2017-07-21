from PIL import Image
import urllib.request
from flask import Flask, request
app = Flask(__name__)


@app.route('/', methods=['POST'])
def palm_read():
    image_url = request.form.get('image')
    im = Image.open(urllib.request.urlopen(image_url))
    # print(im.format, im.mode, im.size)
    return ' '.join([im.format, im.mode, str(im.size[0]), str(im.size[1])])

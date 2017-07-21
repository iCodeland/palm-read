from PIL import Image
import urllib.request
from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/', methods=['POST'])
def palm_read():
    image_url = request.form.get('image')
    im = Image.open(urllib.request.urlopen(image_url))

    return jsonify({
        "messages": [
            {"text": "Format: %s" % im.format},
            {"text": "Mode: %s" % im.mode},
            {"text": "Size: %d, %d" % im.size},
        ]
    })

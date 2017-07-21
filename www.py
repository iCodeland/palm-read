import numpy as np
from PIL import Image
import urllib.request
from flask import Flask, request, jsonify
from keras.models import load_model

from settings import *

app = Flask(__name__)


def rank(value):
    value = int(value)
    if value < 1 or value > 5:
        return '無法判斷'
    return '%d / 5' % value


@app.route('/', methods=['POST'])
def palm_read():
    model = load_model('model.mdl')
    image_url = request.form.get('image')
    img = Image.open(urllib.request.urlopen(image_url))
    X = np.array(
        img.resize((IMG_WIDTH, IMG_HEIGHT)).getdata(),
        np.uint8,
    ).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3) / 255
    love, job, health = [rank(pred) for pred in model.predict(X)[0]]
    return jsonify({
        "messages": [
            {"text": "愛情: %s" % love},
            {"text": "成就: %s" % job},
            {"text": "健康: %s" % health},
        ]
    })

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
    model = load_model('palm_model.mdl')
    image_url = request.form.get('image')
    img = Image.open(urllib.request.urlopen(image_url))
    X = np.array(
        img.resize((img_width, img_height)).getdata(),
        np.uint8,
    ).reshape(1, img_width, img_height, 3) / 255
    prediction = model.predict(X)[0]
    result = [rank(p) for p in prediction]
    love = result[0]
    job = result[1]
    health = result[2]
    return jsonify({
        "messages": [
            {"text": "愛情: %s" % love},
            {"text": "成就: %s" % job},
            {"text": "健康: %s" % health},
        ]
    })

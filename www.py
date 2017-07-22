import traceback
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
    return '  '.join(['\u2b50' for _ in range(value)])


@app.route('/', methods=['POST'])
def palm_read():

    user_input = request.form.get('user_input')

    # gif or picture
    if user_input.startswith('https://scontent.xx.fbcdn.net/'):
        try:
            img = Image.open(urllib.request.urlopen(user_input))
            model = load_model('model.mdl')
            X = np.array(
                img.resize((IMG_WIDTH, IMG_HEIGHT)).getdata(),
                np.uint8,
            ).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3) / 255
            ranking = tuple([rank(pred) for pred in model.predict(X)[0]])
            return jsonify({
                'messages': [
                    {'text': '您的手相算命結果為:\n愛情: %s\n成就: %s\n健康: %s' % ranking}
                ]
            })
        except Exception as e:
            with open('error.log', 'a') as f:
                f.write(traceback.format_exc())

    return jsonify({
        'messages': [
            {'text': 'Handbot無法辨識您的手相，請重新拍攝。'},
        ]
    })

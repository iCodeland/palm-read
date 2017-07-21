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

    user_input = request.form.get('user_input')

    # gif or picture
    if user_input.startswith('https://scontent.xx.fbcdn.net/v/t34.0-12/'):
        img = Image.open(urllib.request.urlopen(user_input))
        print(img.format)
        if img.format == 'JPEG':
            model = load_model('model.mdl')
            X = np.array(
                img.resize((IMG_WIDTH, IMG_HEIGHT)).getdata(),
                np.uint8,
            ).reshape(1, IMG_WIDTH, IMG_HEIGHT, 3) / 255
            love, job, health = [rank(pred) for pred in model.predict(X)[0]]
            return jsonify({
                'messages': [
                    {'text': '愛情: %s' % love},
                    {'text': '成就: %s' % job},
                    {'text': '健康: %s' % health},
                ]
            })
        return jsonify({
            'messages': [
                {'text': '照片檔案格式不支援，請使用 JPEG'},
            ]
        })

    return jsonify({
        'messages': [
            {'text': '請拍手掌照片'},
        ]
    })

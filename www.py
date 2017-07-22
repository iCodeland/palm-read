import traceback
import numpy as np
from PIL import Image
import urllib.request
from flask import Flask, request, jsonify
from keras.models import load_model

from settings import *

app = Flask(__name__)


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
            prediction = model.predict(X)[0]
            return jsonify({
                'messages': [
                    {'text': get_result(prediction)},
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


def get_result(values):
    _category = ['愛情', '成就', '健康']
    _comments = [
        [
            [
                '情路有點曲折呢～',
                '天涯何處無芳草，何必單戀一枝花～',
                '想脫魯，快前往Handbot月老廟！',
            ],
            [
                '心累嗎？給彼此一些空間吧！',
                '情感起伏有點大唷～',
                '還在苦尋你的他/她嗎？快前往Handbot月老廟吧！',
            ],
            [
                '愛情運尚可，積極出動機會就在你身邊！',
                '愛情運還可以唷，再加把勁他/她就快出現了～',
                '情路普普，快來Handbot月老廟主動出擊吧！',
            ],
            [
                '不錯唷！感情和樂捏～',
                '哎喲！情商滿高的耶～',
                '哇～內在外在再加把勁就滿分了！',
            ],
            [
                '百年難得一見理想情人！',
                '天生愛情勝利組！',
                '天菜是你！快來Handbook月老廟交朋友吧！',
            ],
        ],
        [
            [
                '不要灰心，後天努力才是成功的契機！',
                '沒關係，一起研究機器學習讓你成就飛高高～',
                '別難過惹，人生不是只有成就而已～',
            ],
            [
                '努力多一點，成就高一點！',
                '想要有更高的成就嗎？請找茱蒂命理大師開運～',
                '成就差強人意，一定可以更好！',

            ],
            [
                '成就普通，抓住機會成功就在你身邊！',
                '成就尚可，平凡順遂的人生也不錯～',
                '成就還可以喔，再加把勁成功不遠 ！',

            ],
            [
                '成就不錯喔，看好你未來的發展！',
                '成就接近滿分，HandBot月老廟的潛力股～',
                '成就相當優秀，說不定很適合創業呢！',

            ],
            [
                '百年難得一見，成就滿分的孩子啊！！！',
                '成就那麼高絕對是HandBot月老廟不可或缺的績優股～',
                '恭喜恭喜，成就滿分，任務達成！',

            ],
        ],
        [
            [
                '健康不太優秀呢，要好好愛惜身體唷！',
                '身體差強人意，多多運動更健康～',
                '想要更健康嗎？請報名HandBot健身工廠！',
            ],
            [
                '好像還可以更健康，不要太累惹～',
                '不夠健康喔，趕快睡覺補充體力！',
                '一天五蔬果，健康天天有！',
            ],
            [
                '健康情形普通，相信你可以更優秀！',
                '健康還可以喔，一天一蘋果醫生會遠離你的～',
                '吃養氣人參，讓你健康一百分！',
            ],
            [
                '身體不錯喔，精神相當飽滿呢！',
                '生命力旺盛，健康狀態良好！',
                '距離健康狀態滿分只差一步惹，加油！！！',
            ],
            [
                '身體如此優秀，趕快加入HandBot月老廟！',
                'HandBot健身工廠強烈徵求身體如此優秀的你～',
                '太優秀惹，健康狀態滿分！',
            ],
        ],
    ]
    result = []
    for i, value in enumerate(values):
        value = int(value)
        res = _category[i] + ': '
        if value < 1 or value > 5:
            res += '無法判斷\n'
        else:
            ranking = '  '.join(['\u2b50' for _ in range(value)])
            comment = np.random.choice(_comments[i][value - 1])
            res += '%s\n%s\n' % (ranking, comment)
        result.append(res)

    result = '\n'.join(result)

    return result

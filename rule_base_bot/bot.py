# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 제대로 된 챗봇을 만들려면 이 부분을 클래스로 만들어서 관리해야함
CUSTOMER_STATE = {
    'menu': None,
    'size': None,
    'pre_intent': None
}


@app.route("/bot")
def bot():
    # 사용자가 보내준 sentence 받기
    query = request.args.get('sentence')

    print('qurey : {}'.format(query))

    # 들어온 정보가 어느 intent에 속하는지 판단
    intent_id = infer_intent(query)

    if intent_id == 'hello':
        reply = '안녕하세요 무엇을 도와드릴까요?'
        CUSTOMER_STATE['menu'] = None
        CUSTOMER_STATE['size'] = None
        CUSTOMER_STATE['pre_intent'] = None

    elif intent_id == 'order':
        entities = infer_ner(query)

        if len(entities) == 0:
            reply = '주문하실 메뉴에 대해서 다시 말씀해주세요.'

        elif len(entities) == 2:
            CUSTOMER_STATE['menu'] = entities['menu']
            CUSTOMER_STATE['size'] = entities['size']

            reply = '{menu} {size} 주문해도 될까요?'.format(menu=entities['menu'],
                                                          size=entities['size'])

        elif intent_id == 'yes' and CUSTOMER_STATE['pre_intent'] == 'order':
            reply = '주문이 완료되었습니다.'

        CUSTOMER_STATE['pre_intent'] = intent_id

        result = {'result': reply}

        print('result {}'.format(result))

    return jsonify(result)


def infer_intent(msg):
    return


def infer_ner(msg):
    result = requests.get(url='http://localhost:5002/intent?sentence=' + msg)

    print('INTENT---------------------')
    print('response code : {}'.format(result.status_code))
    print('intent : {}'.format(result.text))

    if result.status_code == 500:
        return None

    response_data = result.json()

    return response_data['result']


def infer_ner(msg):
    result = requests.get(url='http://localhost:5001/ner?sentence=' + msg)

    print('NER--------------------')
    print('response code : {}'.format(result.status_code))
    print('ner : {}'.format(result.text))

    response_data = result.json()

    return response_data['result']


if __name__ == "__main__":
    app.run()

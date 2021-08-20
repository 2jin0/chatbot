# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify

import numpy as np
import argparse
import intent_infer


app = Flask(__name__)

#사용자가 직접 사용할 경우 받는 값 저장
FLAGS = None

#id들을 딕셔너리 형태로 관리
INTENTS = {0 : 'hellow',
           1 : 'order',
           2 : 'yes',
           3 : 'no'}


@app.route("/intent")
def intent():
    # 사용자가 보낸 값 받기(json 규격의 객체 - 딕셔너리형)
    query = request.args.get('sentence')    # sentence는 자연어 그대로 들어옴

    print('query : {}'.format(query))

    # 시퀀스 데이터 받기
    sequence = intent_infer.preprocess_sentence(query)
    intent_prob = intent_infer.predict(sequence)
    intent_id = np.argmax(intent_prob)  # intent의 어느 id인지 얻어냄
    intent_name = INTENTS[intent_id]    # id를 value값으로 저장

    # 반환(딕셔버리 또는 json형태로 해야함)
    result = {'result' : intent_name}

    print('result : {}'.format(result))

    return jsonify(result)


# 질의를 하고 반납받는 함수 -- 실제로 사용되지 않음
#def infer_intent(msg):
#
#    return intent_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--intent_checkpoint_file',
        type=str,
        default='./intent/training_checkpoint/service/intent_model.h5',
        help='ner predict service checkpoint path'
    )

    FLAGS, umparsed = parser.parse_known_args()

    intent_infer.model_load(checkpoint_file=FLAGS.intent_checkpoint_file)

    app.run(host='0.0.0.0', port=5002, debug=False)

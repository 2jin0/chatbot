# -*- coding: utf-8 -*-

import json
import numpy as np

from tensorflow.keras import models


NUM_SEQUENCE = None
INPUT_TOKEN_INDEX = None
MODEL_PATH = None


def model_load(checkpoint_file):

    global NUM_SEQUENCE, INPUT_TOKEN_INDEX, MODEL_PATH
    MODEL_PATH = checkpoint_file

    # 입력데이터 전처리에 필요한 속성 정보
    json_data = open('./intent/model_parameter.json').read()
    model_parameter = json.loads(json_data)
    NUM_SEQUENCE = int(model_parameter['num_sequence'])
    json_data = open('./intent/input_token_index.json').read()
    INPUT_TOKEN_INDEX = json.loads(json_data)

    print('intent_modle_loaed')



def preprocess_sentence(sentence):

    input_data = np.zeros((1, NUM_SEQUENCE), dtype='float32')

    #char단위로 토크나이즈 함
    for t, char in enumerate(sentence):
        if t < NUM_SEQUENCE:
            input_data[0, t] = INPUT_TOKEN_INDEX[char]

    return input_data

def predict(sentence):

    intent = models.load_model(MODEL_PATH)
    result = intent.predict(sentence)

    return result
# -*- coding: utf-8 -*-

import json
import numpy as np

from konlpy.tag import Twitter
from tensorflow.keras import models


NUM_SEQUENCE = None
INPUT_TOKEN_INDEX = None
REV_TOKEN_INDEX = None
REV_LABEL_INDEX = None
MODEL_PATH = None


def model_load(checkpoint_file):

    global NUM_SEQUENCE, INPUT_TOKEN_INDEX, REV_TOKEN_INDEX, REV_LABEL_INDEX, MODEL_PATH
    MODEL_PATH = checkpoint_file

    # 입력데이터 전처리에 필요한 속성 정보
    json_data = open('./ner/model_parameter.json').read()
    model_parameter = json.loads(json_data)

    NUM_SEQUENCE = int(model_parameter['num_sequence'])

    json_data = open('./ner/input_token_index.json').read()
    INPUT_TOKEN_INDEX = json.loads(json_data)

    json_data = open('./ner/output_label_index.json').read()
    OUTPUT_INDEX_LABEL = json.loads(json_data)

    REV_TOKEN_INDEX = dict(
        (i, word) for word, i in INPUT_TOKEN_INDEX.items())
    REV_LABEL_INDEX = dict(
        (i, word) for word, i in OUTPUT_INDEX_LABEL.items())

    print('ner_modle_loaed')

def preprocess_sentence(sentence):
    twitter = Twitter()

    sentence = twitter.morphs(sentence)

    input_data = np.zeros((1, NUM_SEQUENCE), dtype='float32')

    for t, word in enumerate(sentence):
        if t < NUM_SEQUENCE:
            try:
                input_data[0, t] = INPUT_TOKEN_INDEX.get(word, 1)
            except KeyError:
                input_data[0, t] = INPUT_TOKEN_INDEX['PAD']

    return input_data

def predict(sentence):

    ner = models.load_model(MODEL_PATH)
    ner_prob = ner.predict(sentence)

    result = {}

    for i, infer_prob in enumerate(ner_prob[0]):
        word = REV_TOKEN_INDEX.get(sentence[0][i], 1)
        tag_id = np.argmax(infer_prob)
        tag = REV_LABEL_INDEX[tag_id]

        print('{word} : {tag}'.format(word=word, tag=tag))


        if tag_id > 0 and tag != '-':
            result[tag] = word

    return result
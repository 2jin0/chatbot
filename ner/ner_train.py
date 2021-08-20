#-*- coding: utf-8 -*-

import numpy as np
import os
import sys
import time
import argparse
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

import ner_model
import preporcess as ps

FLAGS = None

INPUT_TOKEN_INDEX = {}
OUTPUT_LABEL_INDEX = {}
MODEL_PARAMETER = {}


def save_word_analysis_data():

    with open('./ner/input_token_index.json', 'w') as fp:
        json.dump(INPUT_TOKEN_INDEX, fp)
    with open('./ner/output_label_index.json', 'w') as fp:
        json.dump(OUTPUT_LABEL_INDEX, fp)
    with open('./ner/model_parameter.json', 'w') as fp:
        json.dump(MODEL_PARAMETER, fp)

def main(_):

    global INPUT_TOKEN_INDEX, MODEL_PARAMETER, OUTPUT_LABEL_INDEX

    train_data, train_labels, INPUT_TOKEN_INDEX, OUTPUT_LABEL_INDEX = ps.preprocess_data(
        FLAGS.data_path, FLAGS.num_sequence)

    model = ner_model.ner(len(INPUT_TOKEN_INDEX), FLAGS.embedding_dim, FLAGS.num_sequence,
                          len(OUTPUT_LABEL_INDEX), FLAGS.hidden_units)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy', Precision(), Recall(), AUC()])

    model.fit(train_data, train_labels,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              validation_split=0.2)

    if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)

    model.save(os.path.join(FLAGS.checkpoint_path, 'ner_model.h5'))

    MODEL_PARAMETER = {
        "num_input_tokens": len(INPUT_TOKEN_INDEX),
        "num_output_labels": len(OUTPUT_LABEL_INDEX),
        "embedding_dim": FLAGS.embedding_dim,
        "hidden_units": FLAGS.hidden_units,
        "num_sequence": FLAGS.num_sequence,
    }

    save_word_analysis_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default='./ner/train_data',
        help='training data path'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./ner/training_checkpoint/' + str(int(time.time())),
        help='checkpoint file path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help='전체 학습 데이터 소진 횟수'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='학습 시 한번에 사용되는 데이터의 건수'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=16,
        help='임베딩 차원수 '
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=32,
        help='은닉층의 퍼셉트론의 개수'
    )
    parser.add_argument(
        '--num_sequence',
        type=int,
        default=8,
        help='문장의 시퀀스 길이'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main([sys.argv[0]] + unparsed)
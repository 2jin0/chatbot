# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import json
import tensorflow as tf

from tensorflow.keras.optimizers import  RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import preporcess as ps
import intent_model

FLAGS = None

INPUT_TOKEN_INDEX = {}
MODEL_PARAMETER = {}

def save_word_analysis_data():

    with open('./intent/input_token_index.json', 'w') as fp:
        json.dump(INPUT_TOKEN_INDEX, fp)
    with open('./intent/model_parameter.json', 'w') as fp:
        json.dump(MODEL_PARAMETER, fp)


def main(_):

    global INPUT_TOKEN_INDEX, MODEL_PARAMETER

    train_data, train_labels, num_sequence, num_input_tokens, INPUT_TOKEN_INDEX = ps.preprocess_data(FLAGS.data_path)

    model = intent_model.Intent(num_input_tokens, FLAGS.embedding_dim, num_sequence)

    optimizer = RMSprop(lr=1e-4)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(train_data, train_labels,
              epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size,
              validation_split=0.2)

    if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)
    model.save(os.path.join(FLAGS.checkpoint_path,'intent_model.h5'))

    MODEL_PARAMETER = {
        "num_input_tokens": num_input_tokens,
        "num_sequence": num_sequence,
        "embedding_dim": FLAGS.embedding_dim
    }

    save_word_analysis_data()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default='./intent/train_data.csv',
        help='training data path'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./intent/training_checkpoint/' + str(int(time.time())),
        help='checkpoint file path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='전체 학습 데이터 소진 횟수'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='학습 시 한번에 사용되는 데이터의 건수'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=512,
        help='evaluate attention img file path'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main([sys.argv[0]] + unparsed)
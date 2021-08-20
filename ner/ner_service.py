# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, jsonify

import argparse
import ner_infer


app = Flask(__name__)
FLAGS = None

@app.route("/ner")
def ner():
    query = request.args.get('sentence')

    print('query : {}'.format(query))

    sequences = ner_infer.preprocess_sentence(query)
    result = ner_infer.predict(sequences)

    result = {'result': result}

    print('result : {}'.format(result))

    return jsonify(result)

# def infer_ner(msg):
#
#
#     return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ner_checkpoint_path',
        type=str,
        default='./ner/training_checkpoint/service/ner_model.h5',
        help='ner predict service checkpoint path'
    )

    FLAGS, umparsed = parser.parse_known_args()

    ner_infer.model_load(FLAGS.ner_checkpoint_path)

    app.run(host='0.0.0.0', port=5001, debug=False)


    app.run()
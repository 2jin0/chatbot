import re
import numpy as np

from tensorflow.keras.utils import to_categorical

# 데이터 분석에 불필요한 정보 제거
def normalizeString(s):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣 ^☆; ^a-zA-Z~.!?]+')
    match = hangul.search(s)

    result = []

    if not match:
        result = hangul.sub('', s)

    return result

def preprocess_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    input_texts = []
    train_labels = []
    input_chars = set()

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[:2500]:
        tmp_text = line.split('\t')

        if len(tmp_text) > 1:
            input_text = normalizeString(tmp_text[0])
            target_label = tmp_text[1]

        if len(input_text) > 0 and len(target_label) > 0:
            input_texts.append(input_text)
            train_labels.append(target_label)
            for char in input_text:
                if char not in input_chars:
                    input_chars.add(char)

    input_chars = sorted(list(input_chars))
    num_input_tokens = len(input_chars)
    num_sequence = max([len(txt) for txt in input_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_input_tokens)
    print('Max sequence length for inputs:', num_sequence)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_chars)])

    train_data = np.zeros((len(input_texts),
                           num_sequence),
                          dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            train_data[i, t] = input_token_index[char]

    train_labels = to_categorical(train_labels)

    return train_data, train_labels, num_sequence, num_input_tokens, input_token_index
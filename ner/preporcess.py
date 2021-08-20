import re
from collections import Counter

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def preprocess_data(data_path, num_sequece):
    vocab = Counter()
    sentences = []
    sentence = []
    ner_set = set()
    ner_to_index = {}

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue

        splits = line.split('\t')
        splits[-1] = re.sub(r'\n', '', splits[-1])
        word = splits[1]
        vocab[word] = vocab[word] + 1

        sentence.append([word, splits[-1]])

        ner_set.add(splits[-1])

    print('Number of samples:', len(sentences))
    print('Number of unique input tokens:', len(vocab))
    print('Number of unique output labels:', len(ner_set))
    print('Max sequence length for inputs:', max([len(sentence) for sentence in sentences]))

    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    word_to_index = {w: i + 2 for i, (w, n) in enumerate(vocab_sorted)}
    word_to_index['PAD'] = 0
    word_to_index['OOV'] = 1

    ner_to_index['PAD'] = 0
    i = 1

    for ner in ner_set:
        ner_to_index[ner] = i
        i = i + 1

    data_x = []

    for s in sentences:
        temp_x = []

        for w, label in s:
            try:
                temp_x.append(word_to_index.get(w, 1))
            except KeyError:
                temp_x.append(word_to_index['PAD'])

        data_x.append(temp_x)

    data_y = []

    for s in sentences:
        temp_y = []

        for w, label in s:
            temp_y.append(ner_to_index.get(label))

        data_y.append(temp_y)

    pad_x = pad_sequences(data_x, padding='post', maxlen=num_sequece)

    pad_y = pad_sequences(data_y, padding='post', maxlen=num_sequece)
    pad_y = to_categorical(pad_y)

    train_data, test_data, train_labels, test_labels = train_test_split(
        pad_x, pad_y, test_size=.2, random_state=777)

    return train_data, train_labels, word_to_index, ner_to_index
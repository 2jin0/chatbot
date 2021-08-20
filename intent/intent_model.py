from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def Intent(num_input_tokens, embedding_dim, num_sequence):
    model = Sequential()
    model.add(layers.Embedding(num_input_tokens, embedding_dim, input_length=num_sequence))
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    return model

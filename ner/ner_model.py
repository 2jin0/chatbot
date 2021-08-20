from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed

def ner(num_input_tokens, embedding_dim, num_sequence, num_labels, num_units):
    model = Sequential()
    model.add(Embedding(input_dim=num_input_tokens, output_dim=embedding_dim, input_length=num_sequence))
    model.add(Bidirectional(LSTM(num_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_labels, activation='softmax')))

    return model

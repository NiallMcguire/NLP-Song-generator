import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np



print("####################")
print("       Welcome      ")
print("####################")
print("\n")
userInput = input("1 - Train Model \n")
userData= ""
def Train(userInput, userData):
    global tokenizer, model, max_sequence_len
    if userInput == "1":
        tokenizer = Tokenizer()
        data = open('C:/Users/niall/Desktop/lyrics.txt').read()
        corpus = data.lower().split("\n")  # lower changes the case to lower, split - splits the text on denoted character or string

        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1

        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        model = Sequential()
        model.add(Embedding(total_words, 1000, input_length=max_sequence_len - 1))
        model.add(Bidirectional(LSTM(150)))
        model.add(Dense(total_words, activation='softmax'))
        adam = Adam(lr=0.02)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        history = model.fit(xs, ys, epochs=100, verbose=1)
        print(input_sequences)

        print("\nPlease Select one of the following")
        userInput = input("1 - Train Model \n2 - Generate funny text\n")
        Train(userInput, userData)

    if userInput == "2":
        userData = input("\nPlease enter seed text:\n")
        seed_text = userData
        next_words = 100

        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        print(seed_text)
        print("\nPlease Select one of the following")
        userInput = input("1 - Train Model \n2 - Generate funny text\n")
        Train(userInput, userData)


Train(userInput, userData)




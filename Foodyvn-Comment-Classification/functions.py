import numpy as np
import tensorflow as tf
from pyvi import ViTokenizer
import string
import re

lstm_layer = tf.keras.layers.LSTM

class SentimentAnalysisModel(tf.keras.Model):
    """
    Mô hình phân tích cảm xúc của câu

    Properties
    ----------
    word2vec: numpy.array
        word vectors
    lstm_layers: list
        list of lstm layers, lstm cuối cùng sẽ chỉ trả về output của lstm cuối cùng
    dropout_layers: list
        list of dropout layers
    dense_layer: Keras Dense Layer
        lớp dense layer cuối cùng nhận input từ lstm,
        đưa ra output bằng số lượng class thông qua hàm softmax
    """

    def __init__(self, word2vec, lstm_units, n_layers, num_classes, dropout_rate=0.25):
        """
        Khởi tạo mô hình

        Paramters
        ---------
        word2vec: numpy.array
            word vectors
        lstm_units: int
            số đơn vị lstm
        n_layers: int
            số layer lstm xếp chồng lên nhau
        num_classes: int
            số class đầu ra
        dropout_rate: float
            tỉ lệ dropout giữa các lớp
        """
        super().__init__(name='sentiment_analysis')

        # Khởi tạo các đặc tính của model
        self.word2vec = word2vec

        self.lstm_layers = []  # List chứa các tầng LSTM
        self.dropout_layers = []  # List chứa các tầng dropout

        for i in range(n_layers):
            new_layer = lstm_layer(units=lstm_units, name="lstm_" + str(i), return_sequences=(i < n_layers - 1))
            self.lstm_layers.append(new_layer)
            self.dropout_layers.append(tf.keras.layers.Dropout(rate=dropout_rate, name="dropout_" + str(i)))

        self.dense_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="dense_0")


    def call(self, inputs):

        # Thực hiện các bước biến đổi khi truyền thuận input qua mạng
        inputs = tf.cast(inputs, tf.int32)
        # Input hiện là indices, cần chuyển sang dạng vector
        inputs = tf.nn.embedding_lookup(self.word2vec, inputs)

        for i in range(len(self.lstm_layers)):
            inputs = self.lstm_layers[i](inputs)
            inputs = self.dropout_layers[i](inputs)
        out = self.dense_layer(inputs)
        return out


def clean_document(doc):
    # Pyvi Vitokenizer library
    doc = ViTokenizer.tokenize(doc)
    # Lower
    doc = doc.lower()
    # Split in_to words
    tokens = doc.split()
    # Remove all punctuation
    table = str.maketrans('', '', string.punctuation.replace("_", ""))
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens


strip_special_chars = re.compile("[^\w0-9 ]+")


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def get_sentence_indices(sentence, max_seq_length, _words_list, word2idx):
    """
    Hàm này dùng để lấy index cho từng từ
    trong câu (không có dấu câu, có thể in hoa)
    Parameters
    ----------
    sentence là câu cần xử lý
    max_seq_length là giới hạn số từ tối đa trong câu
    _words_list là bản sao local của words_list, được truyền vào hàm
    """
    indices = np.zeros((max_seq_length), dtype='int32')
    words = [word.lower() for word in sentence.split()]

    unk_idx = word2idx['UNK']

    for idx, word in enumerate(words):
        if idx > max_seq_length - 1:
            break
        elif word in word2idx:
            indices[idx] = word2idx[word]
        else:
            indices[idx] = unk_idx
    return indices


def predict(sentence, model, _word_list, _max_seq_length, word2idx):
    """
    Dự đoán cảm xúc của một câu

    Parameters
    ----------
    sentence: str
        câu cần dự đoán
    model: model keras
        model keras đã được train/ load trọng số vừa train
    _word_list: numpy.array
        danh sách các từ đã biết
    _max_seq_length: int
        giới hạn số từ tối đa trong mỗi câu

    Returns
    -------
    int
        0 nếu là negative, 1 nếu là positive
    """
    tokenized_sent = clean_document(sentence)
    tokenized_sent = ' '.join(tokenized_sent)
    input_data = get_sentence_indices(clean_sentences(tokenized_sent), _max_seq_length, _word_list, word2idx)
    input_data = input_data.reshape(-1, _max_seq_length)
    predictions = model(input_data)
    predictions = tf.argmax(predictions, 1)[0].numpy().astype(np.int32)

    return predictions

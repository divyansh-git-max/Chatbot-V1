# for testing purpose only 

data =[
    ('Hi, How are you?',"greeting"),
    ('Hello!',"greeting"),
    ('I want to buy 10 shares of AAPL stock',"buy_order"),
    ('I want to short 10 shares of NVIDIA',"sell_order"),
    ("I want to buy the 10 shares of APPL","buy_order"),
    ('How much money I have in my wallet',"wallet_fund_query"),
]



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Activation,SimpleRNN,Embedding,LSTM,GRU


texts,labels=zip(*data)


label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(labels) # encoding the labels in the training data using label encoder

tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
X=pad_sequences(sequences,padding='post')
print(X)
print(tokenizer.word_index)


onehot=tf.one_hot(encoded_labels,depth=len(set(encoded_labels)))
y=onehot.numpy()

# model=Sequential([
#     Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=100,input_length=X.shape[1]),
#     LSTM(150),
#     Dense(len(set(encoded_labels)),activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)

# model.fit(X,y,epochs=200,verbose=1)
# model.fit(X_train,y_train,epochs=200,verbose=1,validation_data=(X_val,y_val))


# def predict_intent(query):
#     seq=tokenizer.texts_to_sequences([query])
#     padded=pad_sequences(seq,maxlen=X.shape[1])
#     pred=model.predict(padded)[0]
#     intent=label_encoder.inverse_transform([np.argmax(pred)])
#     return intent[0]
# DEEP RNN , DEEP LSTM , DEEP GRU LSTM





model=Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=100,input_length=X.shape[1]),
    LSTM(150,return_sequences=True),
    LSTM(150),
    Dense(len(set(encoded_labels)),activation='softmax')
])



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=200,verbose=1,validation_data=(X_val,y_val))


def predict_intent_deep_rnn(query):
    seq=tokenizer.texts_to_sequences([query])
    padded=pad_sequences(seq,maxlen=X.shape[1])
    pred=model.predict(padded)[0]
    intent=label_encoder.inverse_transform([np.argmax(pred)])
    return intent[0]



# print(predict_intent('I want to buy 10 shares of NVIDIA'))
print(predict_intent_deep_rnn('I want to buy 10 shares of NVIDIA'))
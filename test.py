# for testing purpose only 

data =[
    ('Hi, How are you?', "greeting"),
    ('Hello!', "greeting"),
    ('Good morning!', "greeting"),
    ('Hey there!', "greeting"),
    ('Hi!', "greeting"),
    ('Good afternoon', "greeting"),
    ('Greetings!', "greeting"),
    ('What’s up?', "greeting"),
    ('Hello chatbot!', "greeting"),

    # Buy Orders
    ('I want to buy 10 shares of AAPL stock', "buy_order"),
    ('Buy 50 units of Tesla', "buy_order"),
    ("I want to buy the 10 shares of APPL", "buy_order"),
    ('Can I purchase 20 Google stocks?', "buy_order"),
    ('Order 5 shares of Amazon', "buy_order"),
    ('Buy 100 units of Reliance', "buy_order"),
    ('Purchase 30 Microsoft stocks', "buy_order"),
    ('I’d like to buy 12 shares of Netflix', "buy_order"),
    ('Please buy 40 units of TCS', "buy_order"),
    ('Get me 25 shares of Infosys', "buy_order"),

    # Sell Orders
    ('I want to short 10 shares of NVIDIA', "sell_order"),
    ('Sell 15 shares of Google', "sell_order"),
    ('Please sell 20 Tesla stocks', "sell_order"),
    ('I want to sell 50 units of Amazon', "sell_order"),
    ('Short 10 shares of Microsoft', "sell_order"),
    ('Dump 5 shares of Netflix', "sell_order"),
    ('Sell all my holdings of TCS', "sell_order"),
    ('Can you sell 30 units of Reliance?', "sell_order"),
    ('Get rid of 10 Infosys shares', "sell_order"),

    # Wallet Fund Queries
    ('How much money I have in my wallet', "wallet_fund_query"),
    ('Do I have enough cash?', "wallet_fund_query"),
    ('Check available balance', "wallet_fund_query"),
    ('What’s my account balance?', "wallet_fund_query"),
    ('How much funds do I have?', "wallet_fund_query"),
    ('Is there any money left?', "wallet_fund_query"),
    ('Can you tell me my wallet balance?', "wallet_fund_query"),
    ('I want to know how much cash I have', "wallet_fund_query"),
    ('Show me my available funds', "wallet_fund_query"),
    ('Am I low on balance?', "wallet_fund_query"),

    ('Hi, How are you?', "greeting"),
    ('Hello!', "greeting"),
    ('Good morning!', "greeting"),
    ('Hey there!', "greeting"),
    ("What's up?", "greeting"),

    ('I want to buy 10 shares of AAPL stock', "buy_order"),
    ('I would like to purchase some stocks', "buy_order"),
    ("Buy 50 units of Tesla", "buy_order"),
    ("Get me 25 shares of MSFT", "buy_order"),
    ("I want to buy the 10 shares of APPL", "buy_order"),

    ('I want to short 10 shares of NVIDIA', "sell_order"),
    ('Sell 15 shares of Google', "sell_order"),
    ('I want to sell my Meta stocks', "sell_order"),
    ('Short 20 units of AMZN', "sell_order"),
    ('Can you short Apple stock?', "sell_order"),

    ('How much money I have in my wallet', "wallet_fund_query"),
    ('Show me my funds', "wallet_fund_query"),
    ('Do I have enough cash?', "wallet_fund_query"),
    ("What's my account balance?", "wallet_fund_query"),
    ("Check available balance", "wallet_fund_query"),
      ("Hey, what's going on?", "greeting"),
    ("Yo!", "greeting"),
    ("Hi there, assistant", "greeting"),
    ("Good to see you!", "greeting"),
    ("How's it going?", "greeting"),
    ("Wassup!", "greeting"),

    # Buy Orders
    ("Can you buy 60 shares of Apple?", "buy_order"),
    ("Place an order for 100 units of Infosys", "buy_order"),
    ("Need to buy some Tesla stocks", "buy_order"),
    ("I am interested in buying 70 Google shares", "buy_order"),
    ("Buy me 90 shares of Facebook", "buy_order"),
    ("Please go ahead and buy 25 Amazon shares", "buy_order"),
    ("Execute a buy for 10 units of Zomato", "buy_order"),

    # Sell Orders
    ("I want to sell 80 shares of Apple", "sell_order"),
    ("Put 100 Tesla units for sale", "sell_order"),
    ("Please sell off my Microsoft stocks", "sell_order"),
    ("I need to liquidate 45 Facebook shares", "sell_order"),
    ("Initiate a sell order for 60 Reliance", "sell_order"),
    ("Sell those 10 shares of Amazon I bought", "sell_order"),
    ("Dispose of 5 Netflix units", "sell_order"),

    # Wallet Fund Queries
    ("Tell me how much money is in my account", "wallet_fund_query"),
    ("Check my current balance", "wallet_fund_query"),
    ("Do I have sufficient funds?", "wallet_fund_query"),
    ("Let me know if I have enough balance", "wallet_fund_query"),
    ("Can you check if my wallet has money?", "wallet_fund_query"),
    ("Show my wallet amount", "wallet_fund_query"),
    ("Is there enough in my account to buy stocks?", "wallet_fund_query"),
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
    Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=200,input_length=X.shape[1]),
    LSTM(200,return_sequences=True),
    LSTM(200),
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
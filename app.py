from flask import Flask,request,jsonify
import spacy
import re
import numpy as np
from bs4 import BeautifulSoup
from Models.ner_model import NERModel
from Models.intent_model import IntentClassifier
import chromadb




nlp = spacy.load("custom_ner_model") 

app=Flask(__name__)
trained_nlp=spacy.load('custom_ner_model')


# chroma_client = chromadb.Client()
# collection = chroma_client.create_collection(name="chatbot_faq")


# collection.add(
#     documents=[
#         "How to cancel the order?",
#         "I want the refund"
#         "I want to know the price of META stock"
#         "I want to know the stock price of the apple company",
#         "This document is about the oranges",
#         "Give my P/L statement for this month"
#     ],
#     ids=["id1", "id2","id3","id4","id5",'id6']
# )

# preprocess the user input => predict the entity =>  print the entity => generate the response


# def preprocess(q): # regex part
#     q=str(q).lower().strip()
#     q=q.replace('%','percent')
#     q=q.replace('$','dollar')
#     q=q.replace('₹','rupee')
#     q=q.replace('@','at')
#     q=q.replace('€','euro')
#     q = q.replace(',000,000,000 ', 'b ')
#     q = q.replace(',000,000 ', 'm ')
#     q = q.replace(',000 ', 'k ')
#     q = re.sub(r'([0-9]+)000000000', r'\1b', q)
#     q = re.sub(r'([0-9]+)000000', r'\1m', q)
#     q = re.sub(r'([0-9]+)000', r'\1k', q)
#     return q



# def extract_entities(input): # it will extract the entities from the user input
#     doc = trained_nlp(input)
#     entities = {ent.label_: ent.text for ent in doc.ents}
#     return entities




# @app.route('/product_entity',methods=['POST']) # testing if the routes were working or not
# def predict_entity():
#     data=request.get_json()
#     text=data.get('text')
#     if not text:
#         return jsonify({
#             "error":"some user input is required!"
#         }),400
#     test_text=[text]
#     print(test_text)
#     try:
#         final_list=[]
#         for test in test_text:
#             doc=trained_nlp(test)
#             for ent in doc.ents:
#                 final=(ent.text,ent.label_)
#                 final_list.append(final)
#         return jsonify({
#             "message":final_list
#         })
#     except ValueError as e:
#         print(e)
#     return jsonify({
#         "message":"Yoo bro"
#     }),200



# def generate_response(user_input): # entities=[]
#     for text in user_input:
#         doc=trained_nlp(text)



# @app.route('/chat',methods=["GET","POST"]) # final O/P
# def chat():
#     data=request.get_json()
#     user_input=[data.get('text')]
#     response=generate_response(user_input)
#     return jsonify({
#         "message":response
#     }),200

 
@app.route('/ner_testing_route',methods=['POST',"GET"]) # route to extract the enitites only
def ner_testing_route(): 
    data=request.get_json()
    user_input=data.get('text')
    ner=NERModel()
    final_list=ner.extract_entities(user_input)
    print(final_list)
    if final_list != []:
        return jsonify({
            "message":final_list
        })
    else:
        return jsonify({
            "message":"Sorry i Could get you can you rephrase your query please"
        })
    



@app.route('/intent_classify',methods=['GET',"POST"]) # route to test intent
def intent_classify():
    data=request.get_json()
    user_input=data.get('text')
    if not user_input:
        return jsonify({
            "message":"Sorry you have to type the query"
        }),400
    intent_classifer=IntentClassifier()
    user_intent=intent_classifer.predict(user_input)
    if isinstance(user_intent,np.ndarray):
        user_intent=user_intent.tolist()
    if user_intent:
        return jsonify({
            "message":user_intent[0]
        }),200
    else:
        return jsonify({
            "message":"can you rephrase your query?"
        }),400


# @app.route('/chromadb',methods=['GET',"POST"])
# def chromadb():

#     return 



@app.route('/preprocess_query',methods=["GET","POST"]) 
def preprocess_query():
    data=request.get_json()
    user_input=data.get('text')
    if not user_input:
        return jsonify({
            "message":"Sorry you have to type the query"
        }),400
    ner=NERModel()
    final_dict=dict()
    intent_classifer=IntentClassifier()
    extracted_entity=ner.extract_entities(user_input)
    user_intent=intent_classifer.predict(user_input)
    if isinstance(user_intent,np.ndarray):
        user_intent=user_intent.tolist()
    if user_intent:
        if "intent" not in final_dict:
            final_dict['intent']=user_intent
        if extracted_entity != []:
            if 'entities' not in final_dict:
                final_dict['entities']=extracted_entity
        
        return jsonify({
            "message":final_dict
        }),200
    return jsonify({
        "message":"Sorry couldn't get you please rephrase the query and ask again"
    }),400

if __name__ =="__main__":
    app.run(debug=True)



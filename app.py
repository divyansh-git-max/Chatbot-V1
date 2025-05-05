from flask import Flask,request,jsonify
import spacy
import re 
from bs4 import BeautifulSoup
nlp = spacy.load("custom_ner_model") # our custom ner model trained with dataset
app=Flask(__name__)

trained_nlp=spacy.load('custom_ner_model')

# preprocess the user input => predict the entity =>  print the entity => generate the response


def preprocess(q):
    q=str(q).lower().strip()
    q=q.replace('%','percent')
    q=q.replace('$','dollar')
    q=q.replace('₹','rupee')
    q=q.replace('@','at')
    q=q.replace('€','euro')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    return q



def extract_entities(input): # it will extract the entities from the user input
    doc = trained_nlp(input)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities




@app.route('/product_entity',methods=['POST']) # testing if the routes were working or not
def predict_entity():
    data=request.get_json()
    text=data.get('text')
    if not text:
        return jsonify({
            "error":"some user input is required!"
        }),400
    test_text=[text]
    print(test_text)
    try:
        final_list=[]
        for test in test_text:
            doc=trained_nlp(test)
            for ent in doc.ents:
                final=(ent.text,ent.label_)
                final_list.append(final)
        return jsonify({
            "message":final_list
        })
    except ValueError as e:
        print(e)
    return jsonify({
        "message":"Yoo bro"
    }),200







def generate_response(user_input): # entities=[]
    for text in user_input:
        doc=trained_nlp(text)








@app.route('/chat',methods=["GET","POST"]) # final O/P
def chat():
    data=request.get_json()
    user_input=[data.get('text')]
    response=generate_response(user_input)
    return jsonify({
        "message":response
    }),200
















if __name__ =="__main__":
    app.run(debug=True)
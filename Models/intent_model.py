# intent classification using the logistic regression 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import joblib
from flask import jsonify

class IntentClassifier():
    def __init__(self,model_path='Intent_model'):
        self.model = joblib.load(model_path)
        print("[INFO] Model loaded from disk.")
    
    def regex_search(self,user_query):
        if re.search(r'\bhi\b|\bhello!\b|\bhey\b|\bhello\b|\bmorning\b|\bevening\b|\bevening\b',user_query):
            return "greeting"
        elif re.search(r'\bbuy\b.*\bshare\b',user_query):
            return 'buy_order'
        elif re.search(r'\bsell\b.*\bshare\b',user_query):
            return 'sell_order'
        elif re.search(r'\bcheck\b.*\bbalance\b',user_query):
            return 'wallet_fund_query'
        elif re.search(r'\bshow\b.*\bwallet\b.*\bfund\b',user_query):
            return 'wallet_fund_query'


    def predict(self,user_query):
        rule_based_intent=self.regex_search(user_query)
        if rule_based_intent:
            return rule_based_intent
        final_intent = self.model.predict([user_query])
        if final_intent:
            return final_intent
        return jsonify({
            "message":"intent not found"
        }),400
    



# FY(\d{4} Q[1-4])[^\$]+ \$([0-9\.]+)
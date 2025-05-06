# intent classification using the logistic regression 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import joblib




class IntentClassifier():
    def __init__(self,model_path='intent_model'):
        self.model=None
        self.vectorizer=None
        try:
            self.model = joblib.load(self.model_path)
            print("[INFO] Model loaded from disk.")
        except FileNotFoundError:
            print("[INFO] No saved model found. Train a new one.")

    def regex_search(self,data):
        text,labels=zip(*data)
        text=text.lower()
        if re.search(r'\bhi\b|\bhello!\b|\bhey\b|\bhello\b|\bmorning\b|\bevening\b|\bevening\b',text):
            return "greeting"
        elif re.search(r'\bbuy\b.*\bshare\b',text):
            return 'buy_order'
        elif re.search(r'\bsell\b.*\bshare\b',text):
            return 'sell_order'
        elif re.search(r'\bcheck\b.*\bbalance\b',text):
            return 'wallet_fund_query'
        elif re.search(r'\bshow\b.*\bwallet\b.*\bfund\b',text):
            return 'wallet_fund_query'

    def model_training(self,data):
        texts,labels=zip(*data)
        X_train,X_val,y_train,y_val=train_test_split(texts,labels,test_size=0.2,random_state=42)
        self.model=Pipeline([
            ('tfidf',TfidfVectorizer()),
            ('LG',LogisticRegression(max_iter=10000))
        ])
        self.model.fit(X_train,y_train)
        print("Training complete. Evaluation on test set:")
        joblib.dump(self.model, self.model_path)
        y_pred=self.model.predict(X_val)
        print('classifiction_report',classification_report(y_val,y_pred))

    def predict(self,data):
        rule_based_intent=self.regex_search(data)
        text,labels=zip(*data)
        text=text.lower()
        if rule_based_intent:
            return rule_based_intent
        return self.model_training.predict(text)[0]
    



# FY(\d{4} Q[1-4])[^\$]+ \$([0-9\.]+)
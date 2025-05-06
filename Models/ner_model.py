import spacy

class NERModel():
    def __init__(self, model="custom_ner_model"):
        self.nlp=spacy.load(model)

    def extract_entities(self,text):
        doc=self.nlp(text)
        return [(ent.text,ent.label_) for ent in doc.ents]

    def print_entities(self,text):
        entities=self.extract_entities(text)
        for label,text in entities:
            print(f"{text}-> {label}")
    
    def extract_batches(self,text):
        results=[]
        docs=self.nlp.pipe(text)
        for doc in docs:
            res=[(ent.text,ent.label_) for ent in doc.ents]
            results.append(res)
        return results
    
    def print_batch(self,text):
        results=self.extract_batches(text)
        print(results)
        for i,entities in enumerate(results):
            # print(i,entities)
            for text,label in entities:
                print(f"{text}-> {label}")
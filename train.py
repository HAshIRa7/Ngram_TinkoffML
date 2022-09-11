import argparse  
import pickle as pcl 
from os import listdir
import re 
import random  
import numpy as np  

class N_gram_model: 
    
    def __init__(self, n = 2): 
        self.n = n 
        self.model = {} # mapping prev words to target word and its probability 

    def fit(self, input_dir: str):   
        text = ''
        d = listdir(input_dir) 
        for f in d:
            with open(input_dir + "/" + f, 'r') as file:
                text += file.read()  
        self.tokenize(text)  
        for context in self.model.keys(): 
            uniq_val, count = np.unique(self.model[context], return_counts=True) 
            prob = count / len(self.model[context])
            self.model[context] = np.column_stack((uniq_val, prob))

    def tokenize(self, text): 
        text = text.lower() 
        text = re.sub(r'[^\w\s]', '', text) 
        tokens = text.split() 
        n = self.n   
        ngrams = [(tuple([tokens[i-t-1] for t in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))] 
        for ngram in ngrams: 
            prev, target = ngram 
            if prev in self.model: 
                self.model[prev].append(target) 
            else: 
                self.model[prev] = [target]

    def save(self, model_path): 
        with open (model_path, 'wb') as file: 
            pcl.dump(self.model, file)  

    def load(self, model_path): 
        with open(model_path, 'rb') as file: 
            self.model.clear() 
            self.model = pcl.load(file) 
            self.n = len(list(self.model.keys())[0]) 

    def generate(self, length, seed): 
        np.random.seed(seed) 
        result = [] 
        cur_words = []
        index = np.random.choice(len(self.model))
        cur_text = list(self.model.keys())[index]

        for word in cur_text:
            result.append(word)  
            cur_words.append(word)

        i = self.n 
        while i < length: 
            words = self.model[tuple(cur_words)][:, 0] 
            prob = self.model[tuple(cur_words)][:, 1].astype(float) 
            next_word = np.random.choice(words, p = prob) 
            result.append(next_word) 
            cur_words.pop(0) 
            cur_words.append(next_word) 
            i += 1 

            if tuple(cur_words) not in self.model: 
                index = np.random.choice(len(self.model))  
                cur_text = list(self.model.keys())[index] 
                cur_words = []
                for word in cur_text:
                    result.append(word) 
                    cur_words.append(word) 

                i += self.n

        return ' '.join(result)



def create_model(n, input_dir, model_path): 
    model = N_gram_model(n)
    model.fit(input_dir) 
    model.save(model_path)
    return model

if __name__ =="__main__":  
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', type=str, default='', help='Path to data for training model') 
    parser.add_argument('--model', type=str, default='', help='Path to model')  
    args = parser.parse_args() 
    model = create_model(5, args.input_dir, args.model) 







 

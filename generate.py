from train import N_gram_model  
import argparse 

if __name__== "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', type=str, default='', help ='There model is saved') 
    parser.add_argument('--length', type=int, default='', help='Generated text length') 
    args = parser.parse_args() 

    n_gram_model = N_gram_model()
    n_gram_model.load(args.model) 
    seed = 230429
    print(n_gram_model.generate(args.length, seed))
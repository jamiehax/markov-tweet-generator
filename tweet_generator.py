import numpy as np
import csv
import argparse
import string
from nltk import word_tokenize
from nltk import pos_tag
from tqdm import tqdm


class V3:
    """
        Model:
            states: part of speech
            transitions: probabilities of transitioning between parts of speech
            alphabet: all seen words
            emissions: probability of emitting a word at state k given p.o.s. k and emission k - 1
    """

    MAX_LENGTH = 20
    WEIGHT_FACTOR = 2
    
    def __init__(self):

        # number of example tweets
        self.num_examples = 0

        # dict of {word: occurrences} accross all examples
        self.words = {}

        # dict of {word k: {word k + 1: word k + 1 occurrences}}
        # i.e. self.given_words[word_k][word_k+1] gives the num of occurrences of seeing the string "word_k word_k+1"
        # the strings stored in here may also contain two words (word could be "I am") for second order Markov
        self.given_words = {}

        # dict of {pos: [words]} i.e. all seen words of that pos
        self.pos_words = {}

        # dict of {pos: occurrences} accross all examples
        self.pos = {}

        # dict of {pos k: {pos k + 1: pos k + 1 occurrences}}
        # i.e. self.given_pos[pos_k][pos_k+1] gives the num of occurrences of seeing two words labled "pos_k pos_k+1"
        self.given_pos = {}
    

    def process_data(self, path: str, col: str, length: int):
        
        tweets = []
        with open(path, 'r') as csvfile:
            tweet_reader = csv.reader(csvfile)
            next(tweet_reader)
            print("processing data")
            for row in tqdm(tweet_reader, total=length):

                # get tweet content
                tweet_content = row[col]

                # normalize each word in tweet
                tweet_content = tweet_content.lower()
                punc = string.punctuation + "‘’“”…—"
                punc = punc.replace("@", '')
                punc = punc.replace("#", '')
                tweet_content_no_punc = "".join([char for char in tweet_content if char not in punc])

                # tokenize words
                tweet_words = word_tokenize(tweet_content_no_punc)

                # remove links and pictures
                tweet_words = self.clean_words(tweet_words)

                # POS tag words
                tweet_words = pos_tag(tweet_words)
                
                # ignore tweets of just a picture or link
                if len(tweet_words) > 2:
                    # add a null value with POS "-1" at end of list to denote end of tweet
                    tweet_words.append((None, "-1"))
                    tweets.append(tweet_words)
        
        return tweets


    def clean_words(self, tweet_words):
        """ Remove links and pictures, and join @ / # with their original account / hashtag """

        cleaned_words = []
        tweet_iter = iter(enumerate(tweet_words))
        for index, word in tweet_iter:
            
            # remove links and pictures
            if "http" in word or "pictwittercom" in word:
                continue

            # replace & code with 'and'
            elif "amp" in word:
                cleaned_words.append("and")

            # add @ / # to the original account / hashtag
            elif ('@' in word or '#' in word) and (index + 1) < len(tweet_words) - 1:
                cleaned_words.append(''.join(tweet_words[index:index+2]))
                next(tweet_iter)

            else:
                cleaned_words.append(word)
    
        return cleaned_words


    def build_model(self, path: str, col: str, length: int):
        """ Add necesary counts to calculate word probabilities """

        # list of [(tweet_id, tweet_content)]
        data = self.process_data(path, col, length)
        self.num_examples = len(data)

        print("building model")
        # count word and pos occurrences
        for tweet in tqdm(data):
            for k, (word, pos) in enumerate(tweet):

                # add word to self.words count dict
                if word in self.words:
                    self.words[word] += 1
                else:
                    self.words[word] = 1
                
                # add POS to self.pos count dict
                if pos in self.pos:
                    self.pos[pos] += 1
                else:
                    self.pos[pos] = 1

                # add word to POS word list
                if pos in self.pos_words:
                    self.pos_words[pos].append(word)
                else:
                    self.pos_words[pos] = [word]

                # first order
                # add "word_k word_k+1" occurrence to self.given_words
                if (k + 1) < len(tweet):
                    if tweet[k][0] in self.given_words:
                        if tweet[k + 1][0] in self.given_words[tweet[k][0]]:
                            self.given_words[tweet[k][0]][tweet[k + 1][0]] += 1
                        else:
                            self.given_words[tweet[k][0]][tweet[k + 1][0]] = 1
                    else:
                        self.given_words[tweet[k][0]] = {tweet[k + 1][0]: 1}
                
                # second order
                # add "word_k-1 word_k word_k+1" occurrence to self.given_words and self.words
                if (k - 1) >= 0 and (k + 1) < len(tweet):
                    two_words = ' '.join([tweet[k - 1][0], tweet[k][0]])

                    # add word to self.words count dict
                    if two_words in self.words:
                        self.words[two_words] += 1
                    else:
                        self.words[two_words] = 1

                    if two_words in self.given_words:
                        if tweet[k + 1][0] in self.given_words[two_words]:
                            self.given_words[two_words][tweet[k + 1][0]] += 1
                        else:
                            self.given_words[two_words][tweet[k + 1][0]] = 1
                    else:
                        self.given_words[two_words] = {tweet[k + 1][0]: 1}

                # add "pos_k pos_k+1" occurrence to self.given_pos
                if (k + 1) < len(tweet):
                    if tweet[k][1] in self.given_pos:
                        if tweet[k + 1][1] in self.given_pos[tweet[k][1]]:
                            self.given_pos[tweet[k][1]][tweet[k + 1][1]] += 1
                        else:
                            self.given_pos[tweet[k][1]][tweet[k + 1][1]] = 1
                    else:
                        self.given_pos[tweet[k][1]] = {tweet[k + 1][1]: 1}


    def generate(self, prompt: str) -> str:
        """ Generate a tweet with the given word probabilities given the prompt """

        # normalize prompt, POS tag and tokenize
        prompt = prompt.lower()
        punc = string.punctuation + "‘’“”…—"
        punc = punc.replace("@", '')
        punc = punc.replace("#", '')
        prompt_no_punc = "".join([char for char in prompt if char not in punc])
        prompt = word_tokenize(prompt_no_punc) 
        prompt = self.clean_words(prompt) 
        prompt = pos_tag(prompt)        

        # sample next word
        word = prompt[-1][0]
        two_words = ' '.join([prompt[-2][0], prompt[-1][0]])
        pos = prompt[-1][1]
        generated_words = [w[0] for w in prompt]
        while len(generated_words) < self.MAX_LENGTH:
            word = self.sample(word, two_words, pos)
            if word is not None:
                generated_words.append(word)
                two_words = ' '.join([generated_words[-2], generated_words[-1]])
                pos = pos_tag(generated_words)[-1][1]
            else:
                break

        predicted_tweet = ' '.join(generated_words)
        return predicted_tweet

        
    def sample(self, word: str, two_words: str, pos: str) -> str:
        """ Return a sample of word k + 1 given word k and pos k """

        # calculate next pos probabilities
        next_pos_probs = {}
        for next_pos in self.given_pos[pos]:
            next_pos_probs[next_pos] = self.given_pos[pos][next_pos] / self.pos[pos]

        # sample next pos
        pos_prob_dist = list(next_pos_probs.values())
        p = np.random.choice(np.arange(0, len(next_pos_probs)), p=pos_prob_dist)
        next_pos = list(next_pos_probs)[p]

        # sample next word given pos and previous word
        next_word_probs = {}
        next_two_words_probs = {}
        if word in self.given_words:

            # only sample from words with correct POS and have followed given word
            possible_words = (w for w in self.given_words[word] if w in self.pos_words[next_pos])
            
            # add probabilities of the next_word given word
            for next_word in possible_words:
                next_word_probs[next_word] = self.given_words[word][next_word] / self.words[word]
            
            if two_words in self.given_words:
                # only sample from words with correct POS and have followed given word and word - 1
                possible_words = (w for w in self.given_words[two_words] if w in self.pos_words[next_pos])

                # add probabilities of the next_word given word and word - 1
                for next_word in possible_words:
                    next_two_words_probs[next_word] = self.given_words[two_words][next_word] / self.words[two_words]


            # create weighted prob distribution for 1 word and 2 word occurrences
            prob_dist = list(next_word_probs.values()) + [(self.WEIGHT_FACTOR * p) for p in list(next_two_words_probs.values())]

            # normalize probability distribution
            prob_dist_sum = sum(prob_dist)
            word_prob_dist = [float(i) / prob_dist_sum for i in prob_dist]

            # sample and return word
            if word_prob_dist:
                w = np.random.choice(np.arange(0, len(next_word_probs) + len(next_two_words_probs)), p=word_prob_dist)
                return_word = list(list(next_word_probs) + list(next_two_words_probs))[w]
                return return_word
            else:
                return None
        else:
            return None


class V2:
    """
        Model:
            states: state k represents the current word
            transitions: probability of seeing word k+1 given seeing word k, k-1, and k-2
            alphabet: all words seen in the set of tweets
    """

    # max number of words to generate in tweet
    MAX_LENGTH = 20

    # factor to weight second order markov probabilities with
    WEIGHT_FACTOR_2 = 2

    # factor to weight third order markov probabilities with
    WEIGHT_FACTOR_3 = 3
    
    def __init__(self):

        # number of example tweets
        self.num_examples = 0

        # dict of {word: occurrences} accross all examples
        self.words = {}

        # dict of {word k: {word k + 1: word k + 1 occurrences}}
        # i.e. self.given_words[word_k][word_k+1] gives the num of occurrences of seeing the string "word_k word_k+1"
        # the strings stored in here may also contain two or three words (word could be "I am" / "I am the") for second / third order Markov
        self.given_words = {}
    

    def process_data(self, path: str, col: str, length: int):
        """ Normalize the tweets by making all words lowercase and removeing punctuation, links, and pictures """
        
        tweets = []
        with open(path, 'r') as csvfile:
            tweet_reader = csv.reader(csvfile)
            next(tweet_reader)
            print("processing data")
            for row in tqdm(tweet_reader, total=length):

                # get tweet content
                tweet_content = row[col]

                # normalize each word in tweet
                tweet_content = tweet_content.lower()
                punc = string.punctuation + "‘’“”…—"
                punc = punc.replace("@", '')
                punc = punc.replace("#", '')
                tweet_content_no_punc = "".join([char for char in tweet_content if char not in punc])

                # tokenize words
                tweet_words = word_tokenize(tweet_content_no_punc)

                # clean up tweet
                tweet_words = self.clean_words(tweet_words)
                
                # ignore tweets of just a picture or link
                if len(tweet_words) > 2:
                    # add a null value to denote end of tweet
                    tweet_words.append(None)
                    tweets.append(tweet_words)
        
        return tweets


    def clean_words(self, tweet_words):
        """ Remove links and pictures, and join @ / # with their original account / hashtag """

        cleaned_words = []
        tweet_iter = iter(enumerate(tweet_words))
        for index, word in tweet_iter:
            
            # remove links and pictures
            if "http" in word or "pictwittercom" in word:
                continue

            # replace & code with 'and'
            elif "amp" in word:
                cleaned_words.append("and")

            # add @ / # to the original account / hashtag
            elif ('@' in word or '#' in word) and (index + 1) < len(tweet_words) - 1:
                cleaned_words.append(''.join(tweet_words[index:index+2]))
                next(tweet_iter)

            else:
                cleaned_words.append(word)
    
        return cleaned_words


    def build_model(self, path: str, col: str, length: int):
        """ Add necesary counts to calculate word probabilities """

        # list of [(tweet_id, tweet_content)]
        data = self.process_data(path, col, length)
        self.num_examples = len(data)

        print("building model")
        # count word and pos occurrences
        for tweet in tqdm(data):
            for k, word in enumerate(tweet):

                # add word to self.words count dict
                if word in self.words:
                    self.words[word] += 1
                else:
                    self.words[word] = 1

                # first order markov probabilities
                # add "word_k word_k+1" occurrence to self.given_words
                if (k + 1) < len(tweet):
                    if tweet[k] in self.given_words:
                        if tweet[k + 1] in self.given_words[tweet[k]]:
                            self.given_words[tweet[k]][tweet[k + 1]] += 1
                        else:
                            self.given_words[tweet[k]][tweet[k + 1]] = 1
                    else:
                        self.given_words[tweet[k]] = {tweet[k + 1]: 1}
                
                # second order markov probabilities
                # add "word_k-1 word_k word_k+1" occurrence to self.given_words and self.words
                if (k - 1) >= 0 and (k + 1) < len(tweet):
                    two_words = ' '.join([tweet[k - 1], tweet[k]])

                    # add word to self.words count dict
                    if two_words in self.words:
                        self.words[two_words] += 1
                    else:
                        self.words[two_words] = 1

                    if two_words in self.given_words:
                        if tweet[k + 1] in self.given_words[two_words]:
                            self.given_words[two_words][tweet[k + 1]] += 1
                        else:
                            self.given_words[two_words][tweet[k + 1]] = 1
                    else:
                        self.given_words[two_words] = {tweet[k + 1]: 1}
                
                # third order markov probabilities
                # add "word_k-2 word_k-1 word_k word_k+1" occurrence to self.given_words and self.words
                if (k - 2) >= 0 and (k + 1) < len(tweet):
                    three_words = ' '.join([tweet[k - 2], tweet[k]])

                    # add word to self.words count dict
                    if three_words in self.words:
                        self.words[three_words] += 1
                    else:
                        self.words[three_words] = 1

                    if three_words in self.given_words:
                        if tweet[k + 1] in self.given_words[three_words]:
                            self.given_words[three_words][tweet[k + 1]] += 1
                        else:
                            self.given_words[three_words][tweet[k + 1]] = 1
                    else:
                        self.given_words[three_words] = {tweet[k + 1]: 1}


    def generate(self, prompt: str) -> str:
        """ Generate a tweet based on word probabilities from the prompt """

        # normalize prompt tokenize
        prompt = prompt.lower()
        punc = string.punctuation + "‘’“”…—"
        punc = punc.replace("@", '')
        punc = punc.replace("#", '')
        prompt_no_punc = "".join([char for char in prompt if char not in punc]) 
        prompt = word_tokenize(prompt_no_punc)
        prompt = self.clean_words(prompt)  

        # sample next word
        word = prompt[-1]
        two_words = ' '.join([prompt[-2], prompt[-1]])
        three_words = ' '.join([prompt[-3], prompt[-1]])
        generated_words = [w for w in prompt]
        while len(generated_words) < self.MAX_LENGTH:
            word = self.sample(word, two_words, three_words)
            if word is not None:
                generated_words.append(word)
                two_words = ' '.join([generated_words[-2], generated_words[-1]])
                three_words = ' '.join([generated_words[-3], generated_words[-2], generated_words[-1]])
            else:
                break

        predicted_tweet = ' '.join(generated_words)
        return predicted_tweet

        
    def sample(self, word: str, two_words: str, three_words: str) -> str:
        """ Return a sample of word k + 1 given word k, k-1, and k-2 """

        # sample next word given pos and previous word
        next_word_probs = {}
        next_2_words_probs = {}
        next_3_words_probs = {}
        if word in self.given_words:
            
            # add probabilities of the next_word given word
            for next_word in self.given_words[word]:
                next_word_probs[next_word] = self.given_words[word][next_word] / self.words[word]
            
            if two_words in self.given_words:

                # add probabilities of the next_word given word and word - 1
                for next_word in self.given_words[two_words]:
                    next_2_words_probs[next_word] = self.given_words[two_words][next_word] / self.words[two_words]
            
            if three_words in self.given_words:

                # add probabilities of the next_word given word and word - 1
                for next_word in self.given_words[three_words]:
                    next_3_words_probs[next_word] = self.given_words[three_words][next_word] / self.words[three_words]

            # create weighted prob distribution for first, second, and third order markov
            prob_dist = list(next_word_probs.values()) + [(self.WEIGHT_FACTOR_2 * p) for p in list(next_2_words_probs.values())] + [(self.WEIGHT_FACTOR_3 * p) for p in list(next_3_words_probs.values())]

            # normalize probability distribution
            prob_dist_sum = sum(prob_dist)
            word_prob_dist = [float(i) / prob_dist_sum for i in prob_dist]

            # sample and return word
            if word_prob_dist:
                w = np.random.choice(np.arange(0, 
                                    len(next_word_probs) + len(next_2_words_probs) + len(next_3_words_probs)),    p=word_prob_dist)
                sample_word = list(list(next_word_probs) + list(next_2_words_probs) + list(next_3_words_probs))[w]
                return sample_word
            else:
                return None
        else:
            return None


class V1:
    """
        Model:
            states: state k represents the current word
            transitions: probability of seeing word k+1 given seeing word k
            alphabet: all words seen in the set of tweets
    """

    # max number of words to generate in tweet
    MAX_LENGTH = 20
    
    def __init__(self):

        # number of example tweets
        self.num_examples = 0

        # dict of {word: occurrences} accross all examples
        self.words = {}

        # dict of {word k: {word k + 1: word k + 1 occurrences}}
        # i.e. self.given_words[word_k][word_k+1] gives the num of occurrences of seeing the string "word_k word_k+1"
        self.given_words = {}


    def process_data(self, path: str, col: str, length: int):
        """ Normalize and tokenize the tweets """
        
        tweets = []
        with open(path, 'r') as csvfile:
            tweet_reader = csv.reader(csvfile)
            next(tweet_reader)
            print("processing data")
            for row in tqdm(tweet_reader, total=length):

                # get tweet content
                tweet_content = row[col]

                # normalize each word in tweet
                tweet_content = tweet_content.lower()
                punc = string.punctuation + "‘’“”…—"
                punc = punc.replace("@", '')
                punc = punc.replace("#", '')
                tweet_content_no_punc = "".join([char for char in tweet_content if char not in punc])

                # tokenize words
                tweet_words = word_tokenize(tweet_content_no_punc)

                # remove links and pictures
                tweet_words = self.clean_words(tweet_words)

                # ignore tweets of just a picture or link
                if len(tweet_words) > 2:
                    # add a null value at end of list to denote end of tweet
                    tweet_words.append(None)
                    tweets.append(tweet_words)
        
        return tweets


    def clean_words(self, tweet_words):
        """ Remove links and pictures, and join @ / # with their original account / hashtag """

        cleaned_words = []
        tweet_iter = iter(enumerate(tweet_words))
        for index, word in tweet_iter:
            
            # remove links and pictures
            if "http" in word or "pictwittercom" in word:
                continue

            # replace & code with 'and'
            elif "amp" in word:
                cleaned_words.append("and")

            # add @ / # to the original account / hashtag
            elif ('@' in word or '#' in word) and (index + 1) < len(tweet_words) - 1:
                cleaned_words.append(''.join(tweet_words[index:index+2]))
                next(tweet_iter)

            else:
                cleaned_words.append(word)
    
        return cleaned_words


    def build_model(self, path: str, col: str, length: int):
        """ Add necesary counts to calculate word probabilities """

        # list of [(tweet_id, tweet_content)]
        data = self.process_data(path, col, length)
        self.num_examples = len(data)

        print("building model")
        # count word and pos occurrences
        for tweet in tqdm(data):
            for k, word in enumerate(tweet):

                # add word to self.words count dict
                if word in self.words:
                    self.words[word] += 1
                else:
                    self.words[word] = 1

                # add "word_k word_k+1" occurrence to self.given_words
                if (k + 1) < len(tweet):
                    if tweet[k] in self.given_words:
                        if tweet[k + 1] in self.given_words[tweet[k]]:
                            self.given_words[tweet[k]][tweet[k + 1]] += 1
                        else:
                            self.given_words[tweet[k]][tweet[k + 1]] = 1
                    else:
                        self.given_words[tweet[k]] = {tweet[k + 1]: 1}
                        
    
    def generate(self, prompt: str) -> str:
        """ Generate a tweet based on word probabilities from the prompt """

       # normalize prompt and tokenize
        prompt = prompt.lower()
        punc = string.punctuation + "‘’“”…—"
        punc = punc.replace("@", '')
        punc = punc.replace("#", '')
        prompt_no_punc = "".join([char for char in prompt if char not in punc]) 
        prompt = word_tokenize(prompt_no_punc)
        prompt = self.clean_words(prompt)   

        # sample next word
        word = prompt[-1]
        while len(prompt) < self.MAX_LENGTH:
            word = self.sample(word)
            if word is not None:
                prompt.append(word)
            else:
                break


        predicted_tweet = ' '.join(prompt)
        return predicted_tweet
        

    def sample(self, word: str) -> str:
        """ Return a sample of word k + 1 given word k """

        next_word_probs = {}
        if word in self.given_words:

            # compute probabilities of the next_word given word
            for next_word in self.given_words[word]:
                next_word_probs[next_word] = self.given_words[word][next_word] / self.words[word]

            # sample next word
            prob_dist = list(next_word_probs.values())
            w = np.random.choice(np.arange(0, len(next_word_probs)), p=prob_dist)
            sampled_word = list(next_word_probs)[w]
            return sampled_word
        else:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Markov Model to generate tweets")
    parser.add_argument(
        "-d", "--data",
        default="biden",
        help="Twitter data to train model on. One of: 'trump' (Donald Trump), 'biden' (Joe Biden), 'musk' (Elon Musk), 'dem' (2020 Election), 'rep' (2020 Election)",
    )
    parser.add_argument(
        "-m", "--model",
        default="v2",
        help="Model to use. One of: 'v1', 'v2', or 'v3"
    )
    args = parser.parse_args()

    data = {
        'trump': {'file': "data/trump.csv",
                  'link': "https://www.kaggle.com/datasets/austinreese/trump-tweets",
                  'size': 43352,
                  'column': 2
        },

        'biden': {'file': "data/biden.csv",
                  'link': "https://www.kaggle.com/datasets/rohanrao/joe-biden-tweets",
                  'size': 6064,
                  'column': 3
        },

        'musk': {'file': "data/musk.csv",
                 'link': "https://www.kaggle.com/datasets/ayhmrba/elon-musk-tweets-2010-2021",
                 'size': 34880,
                 'column': 7
        },

        'dem': {'file': "data/democrat.csv",
                'link': "https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets",
                'size': 42068,
                'column': 2
        },

        'rep': {'file': "data/republican.csv",
                'link': "https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets",
                'size': 44392,
                'column': 2
        }
    }

    if args.model == "v1":
        model = V1()
    elif args.model == 'v2':
        model = V2()
    else:
        model = V3()

    file_path = data[args.data]['file']
    column = data[args.data]['column']

    if 'size' in data[args.data]: 
        size = data[args.data]['size']
    else:
        size = 1
    
    if 'link' in data[args.data]: 
        link = data[args.data]['link']
    else:
        link = ""

    model.build_model(file_path, column, size)

    print(args.model, "model built with", args.data, "tweets adapted from:")
    print(link)

    while True:
        prompt = input("enter a prompt, or 'q' to quit: ")

        if prompt == 'q':
            break
        elif args.model == 'v1' and len(prompt.split()) < 1:
            print("***********************************************")
            print("this model takes a minimum of one word as input")
            print("***********************************************")
        elif args.model == 'v2' and len(prompt.split()) < 3:
            print("**************************************************")
            print("this model takes a minimum of three words as input")
            print("**************************************************")
        elif args.model == 'v3' and len(prompt.split()) < 2:
            print("************************************************")
            print("this model takes a minimum of two words as input")
            print("************************************************")
        else:
            generated_tweet = model.generate(prompt)
            print(generated_tweet)

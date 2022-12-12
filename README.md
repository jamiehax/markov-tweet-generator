# markov-tweet-generator
Provides 3 trainable models for generating tweets.

### V1
This is the simplest of the three models. It implements a first order markov chain - where each state represents a word - to generate text by sampling from the conditional probability distribution.

### V2
This model performs the best of the 3, and implements a third order markov chain, where again each state represents a word. This model generates text by sampling from a weighted distribution of the first, second, and third order probabilities.

### V3
While I had high hopes for this model, it unforuntaely failed to live up to its potential. It implements a second order markov chain with P.O.S. tagging. In this model, the states are parts of speech which depend only on the previous state and are transitioned between based on probabilities learned from the training data. Each state emits a word, which depends on the previous two words.

## Running tweet_generator.py
Run the program from the command line with ```python3 tweet_generator.py```. You can specify the model to train with the ```-m``` argument, which takes one of ```v1```, ```v2```, ```v3```. You can specify the data to train the model on with the ```-d``` argument, which takes one of ```trump``` (Donald Trump tweets), ```biden``` (Joe Biden tweets), ```musk``` (Elon Musk tweets), ```dem``` (tweets from Democrats about the 2020 election), ```rep``` (tweets from Republicans about the 2020 election). For example, to train the V2 model on Joe Biden tweets, run

```python3 tweet_generator.py -m v2 -d biden```

Finally, the -h argument will print out the help menu for the arguments.

Once the model is built, you will be prompted with

```enter a prompt, or 'q' to quit:```

at which point you can enter a prompt for the model to generate a tweet from. 

## Using your own data
If you want to train a model on your own data, then:

1. Make sure it is in a CSV file format
2. Add it to the 'data' directory
3. Add the pathname to the file to the ```data_files``` dict in the main method:
4. Add the index of the column containing the actual tweet content to the ```data_columns``` dict in the ```process_data``` method of whichever model you want to train

'''
Uses a clickbait headline dataset from Kaggle.com
(https://www.kaggle.com/amananandrai/clickbait-dataset) to train a
naive bayes classifier.
Baseline accuracy: 98.734375% (only tokenising)
'''
import numpy as np
import pandas as pd
import nltk, string, operator, itertools
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

class ClickbaitDetector:
    def __init__(self, data=None, lower=False, stop_list=False, punct_removal=False, hide_confusion_matrix=False):
        self.data = data
        self.lower = lower
        self.stop_list = stop_list
        self.stop_words = set(stopwords.words('english'))
        self.punct_removal = punct_removal
        self.hide_confusion_matrix = hide_confusion_matrix
        self.prior = None
        self.classes = [0, 1]
        self.vocabulary = None
        self.likelihood = None

    def __pre_process(self):
        # Run before tokenisation to catch adjacent punctuation
        if self.punct_removal:
            self.data['headline'] = self.data['headline'].apply(lambda row: ''.join([w for w in row if not w in string.punctuation]))

        # Tokenise
        self.data['headline'] = self.data['headline'].apply(lambda row: nltk.word_tokenize(row))

        # Set all characters to be lower case
        if self.lower:
            self.data['headline'] = self.data['headline'].apply(lambda row: [w.lower() for w in row])

        # TODO make punctuation removal actually work
        # stop_list removes words and punctuation where punct_removal only removes punctuation
        if self.stop_list:
            self.data['headline'] = self.data['headline'].apply(lambda row: [w for w in row if not w in self.stop_words])

    def __pre_process_headline(self, headline):
        # Run before tokenisation to catch adjacent punctuation
        if self.punct_removal:
            headline = ''.join([w for w in headline if not w in string.punctuation])
        
        headline = nltk.word_tokenize(headline)

        # Convert all characters to lower case
        if self.lower:
            headline = [w.lower() for w in headline]

        # Remove stop words
        if self.stop_list:
            headline = [w for w in headline if w not in self.stop_words]

        return headline

    def train(self):
        self.__pre_process()

        num_docs     = len(self.data.index)
        self.prior   = self.data['clickbait'].value_counts() / num_docs

        # Split dataframe into classes
        clickbait = self.data.loc[self.data['clickbait'] == 1]
        not_clickbait = self.data.loc[self.data['clickbait'] == 0]

        # Get overall vocabulary and vocabulary distribution
        vocab_count = Counter()
        self.data['headline'].apply(lambda row: [w for w in row]).apply(vocab_count.update)
        self.vocabulary = set(Counter({k: c for k, c in vocab_count.items()}).keys())

        # Get class specific vocabulary distributions
        clickbait_counter = Counter()
        clickbait['headline'].apply(lambda row: [w for w in row]).apply(clickbait_counter.update)
        not_clickbait_counter = Counter()
        not_clickbait['headline'].apply(lambda row: [w for w in row]).apply(not_clickbait_counter.update)
        counts = {0: not_clickbait_counter, 1: clickbait_counter}

        # Find likelihood of each token being part of each class
        self.likelihood = defaultdict(dict)
        for w in self.vocabulary:
            for c in self.classes:
                # Laplace smoothing to prevent 0 values
                self.likelihood[w][c] = (counts[c][w]+1) / (sum([counts[i][w] for i in self.classes]) + len(self.classes))

    def test(self, test_df):
        confusion_matrix = np.zeros((2, 2))
        correct = 0
        for headline, clickbait in test_df.itertuples(index=False):
            classification = self.classify(headline, print_result=False)[0]
            confusion_matrix[clickbait][classification] += 1
            if classification == clickbait:
                correct += 1

        accuracy = 100*(correct / len(test_df.index))
        print(f'Accuracy: {accuracy}')
        if not self.hide_confusion_matrix:
            sns.heatmap(confusion_matrix, annot=True, fmt='', cmap='YlGnBu').set_title('Confusion Matrix')
            plt.show()

        return confusion_matrix, accuracy

    def classify(self, headline, print_result=False, verbose_response=False):
        headline = self.__pre_process_headline(headline)
        class_products = {}
        for c in self.classes:
            class_products[c] = self.prior[c]
            for w in headline:
                if w in self.vocabulary:
                    class_products[c] *= self.likelihood[w][c]

        classification = max(class_products.items(), key=operator.itemgetter(1))
        if print_result:
            print(f'Clickbait: {False if not classification else True}')
        if verbose_response:
            return classification, class_products
        return classification

    def classify_batch(self, headlines, verbose_response=False):
        return [(h, self.classify(h, verbose_response=verbose_response)) for h in headlines]



if __name__ == '__main__':
    # Standard running example
    '''cd = ClickbaitDetector(data=pd.read_csv('training.csv'), lower=False, stop_list=False, punct_removal=False)
    cd.train()
    results = cd.test(pd.read_csv('validation.csv'))'''

    accuracies = {}
    for l in [True, False]:
        for s in [True, False]:
            for p in [True, False]:
                settings = ('l' if l else '') + ('s' if s else '') + ('p' if p else '')
                print(f'Settings: {settings}')
                cd = ClickbaitDetector(data=pd.read_csv('training.csv'), lower=l, stop_list=s, punct_removal=p, hide_confusion_matrix=True)
                cd.train()
                results = cd.test(pd.read_csv('validation.csv'))
                accuracies[settings] = results[1] # Results 1 is accuracy without confusion matrix
                
    print(accuracies)

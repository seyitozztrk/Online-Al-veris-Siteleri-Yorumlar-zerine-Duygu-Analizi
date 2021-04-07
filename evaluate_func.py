


class evaluate_and_initiliaze:

    def __init__(self):
        import pandas 
        print('init')
        #fetch data from csv files
        pos = pandas.read_csv('pos.txt', delimiter="\t", header=None)
        neg = pandas.read_csv('neg_reviews.txt', delimiter="\t", header=None)
        #positive samples
        pos['target'] = 1
        pos.columns=['text', 'target']
        #negative samples
        neg['target'] = 0
        neg.columns=['text', 'target']

        self.df = pandas.concat([neg, pos], ignore_index=True)
        self.df.reset_index()

    def normalization_data(self):
        import nltk
        nltk.download('stopwords')
        import re 
        import numpy as np 

        WPT = nltk.WordPunctTokenizer()
        stop_word_list = nltk.corpus.stopwords.words('turkish')
        from snowballstemmer import stemmer

        
        from TurkishStemmer import TurkishStemmer
        stemmer = TurkishStemmer()
        
        
        yorumlar = []
        for i in range(0, len(self.df)):

            yorum = re.sub("[^AaBbCcÇçDdEeFfGgĞğHhİiIıJjKkLlMmNnOoÖöPpRrSsŞşTtUuÜüVvYyZz']", ' ', self.df['text'][i]) #drop things that without letters
            yorum = re.sub("[']", '', yorum) #drop things that without letters
            yorum = yorum.lower()
            yorum = yorum.strip()
            yorum = yorum.split()

            yorum = [stemmer.stem(word) for word in yorum if word not in stop_word_list]
            yorum = ' '.join(yorum)

            yorumlar.append(yorum)
        # print(yorumlar)ds
        return yorumlar
        
    def create_confusion_matrix(self, true_value, predicted_value):
        from sklearn.metrics import confusion_matrix
        # from sklearn.metrics import plot_confusion_matrix
        confusion_m = confusion_matrix(true_value, predicted_value)

        import matplotlib.pyplot as plt
        from mlxtend.plotting import plot_confusion_matrix
        import numpy as np

        fig, ax = plot_confusion_matrix(conf_mat=confusion_m, figsize=(2, 2), cmap="OrRd")
        plt.show()

            
    def scores(self, true_value, predicted_value):
        from sklearn.metrics import precision_recall_fscore_support as score
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        precision, recall, fscore, support = score(true_value, predicted_value)
        #Precision () is defined as the number of true positives () over the number of true positives plus the number of false positives (Fp).
        # tp / (tp + fp)
        # print('precision: {}'.format(precision))

        print('accuracy: {}'.format(accuracy_score(true_value, predicted_value)))
        print('precision: {}'.format(precision_score(true_value, predicted_value)))

        # Recall () is defined as the number of true positives () over the number of true positives plus the number of false negatives (Fn).
        # tp / (tp + fn)
        print('recall: {}'.format(recall_score(true_value, predicted_value)))
        # These quantities are also related to the () score, which is defined as the harmonic mean of precision and recall.
        print('fscore: {}'.format(f1_score(true_value, predicted_value)))
        # The support is the number of occurrences of each class in y_true.
        print('support: {}'.format(support))
  

    


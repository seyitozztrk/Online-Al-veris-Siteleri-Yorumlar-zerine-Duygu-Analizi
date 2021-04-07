from evaluate_func import evaluate_and_initiliaze

class basic_operation(evaluate_and_initiliaze):
    def __init__(self):
        super().__init__()


    def vectorization_Xtrain(self, Xtrain, Xtest ):
        from sklearn.feature_extraction.text import TfidfVectorizer
      
        self.vectorizer = TfidfVectorizer(max_features=1000, analyzer='word')
        print(self.vectorizer)
        print('slm')

        Xtra = self.vectorizer.fit_transform(Xtrain)
        Xte = self.vectorizer.transform(Xtest)
        self.feature_names = self.vectorizer.get_feature_names()
        return Xtra, Xte
        
    def train_test_split_dataset(self, model):
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        import numpy as np #1
        import pickle
        import pandas as pd 
        from sklearn.model_selection import train_test_split


         # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
        import gensim
        # model = None
        model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)
       
        try:
            file = open('normalization_X_data.pickle', 'rb')
            self.X = pickle.load(file)
        except:
            self.X = self.normalization_data()
            pickle.dump(self.X, open("normalization_X_data.pickle", "wb"))

        
        self.y = self.df.iloc[:,1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.01,test_size=0.005, stratify=self.y,random_state=42)
        
        print(len(self.X_train))
        print(len(self.X_test))
        print(len(self.y_train))
        print(len(self.y_test))

        # print('->>',len(self.y_train[self.y_train == 0] ) )
        # print('->',len(self.y_train[self.y_train == 1] ) )
        
        # print('->>',len(self.y_test[self.y_test == 0] ) )
        # print('->',len(self.y_test[self.y_test == 1] ) )

        print('------------------------')
        
        self.X_train, self.X_test  = self.vectorization_Xtrain(self.X_train, self.X_test )
        print((self.X_train).shape)
        print((self.X_test).shape)
        print('***********************')
        self.X_train = self.embedding_process(self.X_train, model)
        self.X_test  = self.embedding_process(self.X_test, model)


        print(len(self.X_train))
        print(self.y_train.shape)
        print(len(self.X_test))
        print(self.y_test.shape)

        print('********************{} SVM CLASSIFIER')
        self.svmRun()
        
        # if model is ('svm'):
        #     
        # elif model is ('bayes'):
        #     print('********************{} BAYES CLASSIFIER')
        #     self.bayesRun()
        # elif model is ('decision'):
        #     print('********************{} DECISION TREE CLASSIFIER')
        #     self.decisionTreeRun()

        
    def embedding_process(self, arr,model):
        from sklearn.model_selection import train_test_split
        from gensim.models.word2vec import Word2Vec
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np 
        import pandas as pd
        print('embbedding+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
       
        
        vect = self.vectorizer
        feature_names = self.feature_names
        # print('hi hw re u ? {}  <<<-->>>\n{}'.format(vect, feature_names))
        
        train_array = []
        print(type(arr))
        df1 = pd.DataFrame(arr.toarray(), columns=vect.get_feature_names())
        print('->>>',df1.shape)
        # break #careful
        for i in range(df1.shape[0]):
            b = []
            for j in range(df1.shape[1]):
                if df1.iloc[i,j] != 0:
                    try:
                        b.append(model[df1.columns[j]].tolist())
                    except:
                        b.append(np.zeros(100))
                else:
                  b.append(np.zeros(100))

            train_array.append(b)
            # print(train_array[i])
            # print('\n')
        return train_array


    def svmRun(self):
        import numpy as np 
        print('hello svmRun')
        print(type(self.X_train))

        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)

        print(type(self.X_train))
        print(type(self.X_test))
        print('///')
        print(self.X_train.shape)
        print(self.X_test.shape)


        nsamples, nx, ny = self.X_train.shape
        self.X_train = self.X_train.reshape((nsamples,nx*ny))
        
        nsamples2, nx2, ny2 = self.X_test.shape
        self.X_test = self.X_test.reshape((nsamples2,nx2*ny2))

        print(self.X_train.shape)
        print(self.X_test.shape)

        # return

        from sklearn import svm
        linear_svm = svm.LinearSVC()
        print(self.X_train)
        linear_svm.fit(self.X_train, self.y_train)
        y_pred = linear_svm.predict(self.X_test)
        self.create_confusion_matrix(self.y_test, y_pred)
        self.scores(self.y_test, y_pred)
        
    def bayesRun(self):
        from sklearn.naive_bayes import MultinomialNB
        sentiment_model = MultinomialNB().fit(self.X_train, self.y_train)

        y_pred = sentiment_model.predict(self.X_test)
        self.create_confusion_matrix(self.y_test, y_pred)
        self.scores(self.y_test, y_pred)

    def decisionTreeRun(self):
        from sklearn import tree
        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.y_train)

        y_pred = decision_tree.predict(self.X_test)

        self.create_confusion_matrix(self.y_test, y_pred)
        self.scores(self.y_test, y_pred)


        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
        print('finish split, goto vectorization')


        
        
        
        
        
        
        
        









        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # # enumerate the splits and summarize the distributions
        # k_fold = 0 
        # for train_ix, test_ix in kfold.split(self.X, self.y):
        #     k_fold+=1
        #     # select rows
        #     self.X_train, self.X_test = self.X[train_ix], self.X[test_ix]
        #     self.y_train, self.y_test = self.y[train_ix], self.y[test_ix]

        #     self.X_train = self.X_train[:1000]
        #     self.y_train = self.y_train[:1000]
            
        #     self.X_test = self.X_test[:1000]
        #     self.y_test = self.y_test[:1000]

        #     self.X_train, self.X_test = self.embedding_process()

        #     # print(self.X_train.shape, '  ' , self.X_test.shape)
            
        #     # summarize train and test composition
        #     train_0, train_1 = len(self.y_train[self.y_train==0]), len(self.y_train[self.y_train==1])
        #     test_0, test_1 = len(self.y_test[self.y_test==0]), len(self.y_test[self.y_test==1])
        #     print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

        #     if model is ('svm'):
        #         print('********************{} SVM CLASSIFIER'.format(k_fold))
        #         self.svmRun()
        #     elif model is ('bayes'):
        #         print('********************{} BAYES CLASSIFIER'.format(k_fold))
        #         self.bayesRun()
        #     elif model is ('decision'):
        #         print('********************{} DECISION TREE CLASSIFIER'.format(k_fold))
        #         self.decisionTreeRun()
        #     break
          
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    



  

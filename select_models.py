from basic_abc import basic_operation

class models(basic_operation):

    def __init__(self):

        super().__init__()
    
    
    def svmModel(self):
#         self.embedding_process()
        self.train_test_split_dataset('svm')

    def NaiveBayesModel(self):
        
        self.train_test_split_dataset('bayes')
  
    def DecisionTreeModel(self):
       
        self.train_test_split_dataset('decision')
 
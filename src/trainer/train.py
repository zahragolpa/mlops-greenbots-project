from trainer.trainers import RandomForestTrainer, SVMTrainer, LogisticRegressionTrainer
from prefect import flow

@flow(name="Green Bots ML Pipeline")
def train_model(path_data, path_model):
    rf_trainer = RandomForestTrainer()
    rf_trainer.train_random_forest(path_data,path_model)

    # svm_trainer = SVMTrainer()
    # svm_trainer.train_svm('../data/processed/clean_data_20240207_114159.csv', 'models/svm.pkl')
    #
    # lr_trainer = LogisticRegressionTrainer()
    # lr_trainer.train_logistic_regression('../../data/processed/clean_data_20240207_114159.csv', '../models/logistic_regression.pkl')


if __name__ == '__main__':
    path_data = '../../data/processed/clean_data_20240207_114159.csv'
    path_model = '../models/random_forest.pkl'
    
    train_model(path_data,path_model)
import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):

        try:

            logging.info("Splitting training and test input data")

            # Split train and test arrays
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {

                "Logistic Regression": LogisticRegression(),

                "Random Forest Classifier": RandomForestClassifier(),

                "Decision Tree Classifier": DecisionTreeClassifier(),

                "KNeighbors Classifier": KNeighborsClassifier()

            }
            params = {

                "Logistic Regression": {

                    "C": [0.01, 0.1, 1, 10],

                    "solver": ["liblinear"]

                },

                "Random Forest Classifier": {

                    "n_estimators": [50, 100],

                    "max_depth": [5, 10, None]

                },

                "Decision Tree Classifier": {

                    "criterion": ["gini", "entropy"],

                    "max_depth": [5, 10, None]

                },

                "KNeighbors Classifier": {

                    "n_neighbors": [3, 5, 7]

                }
            }
            

            
            

            # Evaluate models
            model_report,best_models = evaluate_model(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                params=params

            )

            print(model_report)

            # Best model score
            best_model_score = max(sorted(model_report.values()))

            # Best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Best model object
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No best model found")
                    

            logging.info("Best model found on both training and testing dataset")

            

            # Save model
            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            # Prediction
            predicted = best_model.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
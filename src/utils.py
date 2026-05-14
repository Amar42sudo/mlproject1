import os
import sys
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            pickle.dump(obj, file_obj)

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_model(

    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params

):

    try:

        report = {}

        best_models = {}

        for model_name, model in models.items():

            param = params[model_name]

            gs = GridSearchCV(

                estimator=model,

                param_grid=param,

                cv=3,

                scoring='accuracy',

                n_jobs=-1

            )

            gs.fit(X_train, y_train)

            tuned_model = gs.best_estimator_

            # Save tuned model
            best_models[model_name] = tuned_model

            # Prediction
            y_test_pred = tuned_model.predict(X_test)

            # Accuracy
            test_model_score = accuracy_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

            print("=" * 50)

            print(f"Model: {model_name}")

            print(f"Accuracy: {test_model_score}")

            print(f"Best Parameters: {gs.best_params_}")

        return report, best_models

    except Exception as e:

        raise CustomException(e, sys)
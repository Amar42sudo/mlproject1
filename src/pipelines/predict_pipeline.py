import sys 
import os 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # Load model
            model = load_object(
                file_path=model_path
            )

            # Load preprocessor
            preprocessor = load_object(
                file_path=preprocessor_path
            )

            # Transform input data
            data_scaled = preprocessor.transform(
                features
            )

            # Prediction
            preds = model.predict(
                data_scaled
            )

            return preds

        except Exception as e:

            raise CustomException(e, sys)

class CustomData:
    def __init__(

        self,
        age,
        gender,
        daily_social_media_hours,
        platform_usage,
        sleep_hours,
        screen_time_before_sleep,
        academic_performance,
        physical_activity,
        social_interaction_level,
        stress_level,
        anxiety_level,
        addiction_level):

        self.age = age

        self.gender = gender

        self.daily_social_media_hours = (
                daily_social_media_hours
            )

        self.platform_usage = platform_usage

        self.sleep_hours = sleep_hours

        self.screen_time_before_sleep = (
                screen_time_before_sleep
            )

        self.academic_performance = (
                academic_performance
            )

        self.physical_activity = (
                physical_activity
            )

        self.social_interaction_level = (
                social_interaction_level
            )

        self.stress_level = stress_level

        self.anxiety_level = anxiety_level

        self.addiction_level = addiction_level

    def get_data_as_frame(self):
        try:

            custom_data_input_dict = {

                "age": [self.age],

                "gender": [self.gender],

                "daily_social_media_hours": [
                    self.daily_social_media_hours
                ],

                "platform_usage": [
                    self.platform_usage
                ],

                "sleep_hours": [
                    self.sleep_hours
                ],

                "screen_time_before_sleep": [
                    self.screen_time_before_sleep
                ],

                "academic_performance": [
                    self.academic_performance
                ],

                "physical_activity": [
                    self.physical_activity
                ],

                "social_interaction_level": [
                    self.social_interaction_level
                ],

                "stress_level": [
                    self.stress_level
                ],

                "anxiety_level": [
                    self.anxiety_level
                ],

                "addiction_level": [
                    self.addiction_level
                ]

            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:
            raise CustomException(e, sys)

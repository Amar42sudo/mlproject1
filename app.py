from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(

            age=float(request.form.get('age')),

            gender=request.form.get('gender'),

            daily_social_media_hours=float(
                request.form.get(
                    'daily_social_media_hours'
                )
            ),

            platform_usage=request.form.get(
                'platform_usage'
            ),

            sleep_hours=float(
                request.form.get(
                    'sleep_hours'
                )
            ),

            screen_time_before_sleep=float(
                request.form.get(
                    'screen_time_before_sleep'
                )
            ),

            academic_performance=request.form.get(
                'academic_performance'
            ),

            physical_activity=request.form.get(
                'physical_activity'
            ),

            social_interaction_level=request.form.get(
                'social_interaction_level'
            ),

            stress_level=request.form.get(
                'stress_level'
            ),

            anxiety_level=request.form.get(
                'anxiety_level'
            ),

            addiction_level=request.form.get(
                'addiction_level'
            )

        )

        pred_df = data.get_data_as_frame()

        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(
            pred_df
        )

        return render_template(

            'home.html',

            results=results[0]

        )


if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
        )
        


import pickle

import numpy as np
import sklearn
from flask import Flask, render_template, request


app = Flask(__name__)
model = pickle.load(open('utils/model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html',) 

@app.route('/predict', methods=['POST'])
def predict():

    feature_array = []
    error_flag = False

    for key in request.form.keys():
        try:
            value = float(request.form[key])
            feature_array.append(value)
        except:
            error_flag = True
            break

    if error_flag:
        return render_template(
            'prediction.html',
            text = "One or more missing or bad value found."
        )

    input_arr  = [np.array(feature_array)]
    prediction = model.predict(input_arr)

    if prediction >= 0.5:
        text = "Please take care of yourelf. You are diabetic."
    else:
        text = "You are not diabetic. Maintain the same. : )"

    return render_template(
        'prediction.html',
        text=text
    )


if __name__ == '__main__':
    app.run()

from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('tic-tac-toe.pkl')

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        data = []
        for key in request.form:
            data.append(int(request.form[key]))

        # Create a DataFrame
        columns = ['top_left_square', 'top_middle_square', 'top_right_square',
                   'middle_left_square', 'middle_middle_square', 'middle_right_square',
                   'bottom_left_square', 'bottom_middle_square', 'bottom_right_square']
        input_df = pd.DataFrame([data], columns=columns)

        # Make a prediction using the model
        prediction = model.predict(input_df).round().astype(int)+1
        prediction = np.where(prediction == 1, 2, np.where(prediction == 2, 1, prediction))

        return render_template('index.html', prediction=prediction[0])
    except:
        return render_template('index.html', error_message="An error occurred. Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)


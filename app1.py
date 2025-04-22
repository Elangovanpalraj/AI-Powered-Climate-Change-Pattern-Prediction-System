from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model1.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        hour = int(request.form['hour'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        weekday = int(request.form['weekday'])

        # Create a DataFrame with these values
        input_df = pd.DataFrame([{
            'Hour': hour,
            'Day': day,
            'Month': month,
            'Weekday': weekday
        }])

        # Handle missing columns due to one-hot encoding
        input_df = pd.get_dummies(input_df)
        expected_cols = ['Hour', 'Day', 'Month', 'Weekday']
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

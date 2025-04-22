from flask import Flask, render_template, request, redirect
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = joblib.load('linear_regression_model.pkl')
target_column = 'Air Temperature (OC)'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and process the uploaded data
        df = pd.read_csv(file_path, parse_dates=["Date/Time"])
        df.columns = df.columns.str.strip()
        df["Hour"] = df["Date/Time"].dt.hour
        df["Day"] = df["Date/Time"].dt.day
        df["Month"] = df["Date/Time"].dt.month
        df["Weekday"] = df["Date/Time"].dt.dayofweek

        X = df.drop(columns=["Date/Time", target_column], errors='ignore')
        X = pd.get_dummies(X)
        X = X.ffill().bfill().fillna(0)

        X = X.tail(10)
        predictions = model.predict(X)

        # Save predictions with timestamp
        output_df = df[["Date/Time"]].tail(10).copy()
        output_df["Predicted_Air_Temperature"] = predictions
        output_csv = os.path.join("predicted_temperatures.csv")
        output_df.to_csv(output_csv, index=False)

        return render_template("index.html", tables=[output_df.to_html(classes='data', header="true")])

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

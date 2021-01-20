import json
import os

from flask import Flask, request, jsonify
from statsmodels.tsa.arima_model import ARIMAResults
import pandas as pd

app = Flask(__name__)


def load_model():
    return ARIMAResults.load('./model.pkl')


@app.route('/predict')
def predict():
    arima = load_model()
    arima = arima.model.fit()
    pred = arima.forecast(steps=7)
    pd.Series(pred)
    return {"predictions": pd.Series(pred).to_json()}


def main():
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()


if __name__ == '__main__':
    main()

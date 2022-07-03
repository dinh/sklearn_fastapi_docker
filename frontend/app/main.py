from flask import Flask, render_template

app = Flask(__name__)

CHURN_API_ROOT = "http://127.0.0.1:8000"


@app.route('/')
def home():
    return render_template('index.html', churn_api_endpoint=f'{CHURN_API_ROOT}/predict')


if __name__ == "__main__":
    app.run(debug=True)

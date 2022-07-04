from flask import Flask, render_template

app = Flask(__name__)

CHURN_API_ROOT = "http://127.0.0.1:8000"


@app.route('/')
def home():
    return render_template('index.html', churn_api_root=f'{CHURN_API_ROOT}')


@app.route('/batch-predict')
def batch_predic():
    return render_template('batch-predict.html', churn_api_root=f'{CHURN_API_ROOT}')


if __name__ == "__main__":
    app.run(debug=True)

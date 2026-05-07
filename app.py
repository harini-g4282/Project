from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    text = news.lower()

    fake_keywords = [
        "dead", "secretly", "aliens", "miracle cure",
        "cat as finance minister", "sun will not rise",
        "ban internet", "free petrol forever"
    ]

    # obvious suspicious claim check
    for word in fake_keywords:
        if word in text:
            return render_template(
                'index.html',
                prediction_text="❌ Fake News"
            )

    # ML prediction for all other inputs
    vector = tfidf.transform([news])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "✅ Real News"
    else:
        result = "❌ Fake News"

    return render_template('index.html', prediction_text=result)
if __name__ == '__main__':
    app.run(debug=True)

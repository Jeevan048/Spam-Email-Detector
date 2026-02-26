from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', accuracy=round(accuracy*100,2))

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = "🚨 Spam Message!"
    else:
        result = "✅ Not Spam Message!"

    return render_template(
        'index.html',
        prediction_text=result,
        confidence=confidence,
        accuracy=round(accuracy*100,2)
    )

if __name__ == "__main__":
    app.run(debug=True)
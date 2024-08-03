from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)
    output = prediction[0][1] * 100  # Convert to percentage

    if output > 25:
        precaution = "Your risk is high. Please consider consulting a doctor and follow these precautionary measures: ..."
    else:
        precaution = "Your risk is low. Maintain a healthy lifestyle to keep it that way."

    return render_template('index.html', prediction_text=f'Your diabetes risk is {output:.2f}%. {precaution}')

if __name__ == "__main__":
    app.run(debug=True)

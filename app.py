from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form values to integers
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = prediction[0]
        if output == 0:
            return render_template('index.html', prediction_text='Prediction: No Heart Disease')
        else:
            return render_template('index.html', prediction_text='Prediction: Heart Disease')
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)

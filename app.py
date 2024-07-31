import pickle
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
import pickle

# Example of saving a model to a pickle file
model = ...  # your model here
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
from joblib import dump, load

# Save the model
dump(model, 'model.joblib')

# Load the model
model = load('model.joblib')


# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except pickle.UnpicklingError as e:
    print(f"Error loading pickle file: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        try:
            # Predict using the loaded model
            result = model.predict([to_predict_list])[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            result = None
        
        if result == 1:
            prediction = 'Given transaction is fraudulent'
        else:
            prediction = 'Given transaction is NOT fraudulent'
        
        return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

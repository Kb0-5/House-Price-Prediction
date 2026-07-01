import os
from flask import Flask, render_template, request
from house_price_predict import load_model, predict_house_price

app = Flask(__name__)

MODEL_PATH = 'house_price_model.pkl'

if not os.path.exists(MODEL_PATH):
    print("Model file not found. Running training pipeline on Render server...")
    import house_price_train  

model = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_data = {
            'POSTED_BY': request.form.get('POSTED_BY'),
            'UNDER_CONSTRUCTION': int(request.form.get('UNDER_CONSTRUCTION')),
            'RERA': int(request.form.get('RERA')),
            'BHK_NO.': int(request.form.get('BHK_NO.')),
            'BHK_OR_RK': request.form.get('BHK_OR_RK'),
            'SQUARE_FT': float(request.form.get('SQUARE_FT')),
            'READY_TO_MOVE': int(request.form.get('READY_TO_MOVE')),
            'RESALE': int(request.form.get('RESALE')),
            'LONGITUDE': float(request.form.get('LONGITUDE')),
            'LATITUDE': float(request.form.get('LATITUDE'))
        }
        prediction = predict_house_price(input_data, model)
        prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import joblib

def load_model(model_path='house_price_model.pkl'):
    return joblib.load(model_path)

def predict_house_price(input_data, model):
    input_df = pd.DataFrame([input_data])
    
    # Generate prediction
    prediction = model.predict(input_df)
    return float(prediction[0])

if __name__ == "__main__":
    model = load_model()
    sample_input = {
        'POSTED_BY': 'Owner',
        'UNDER_CONSTRUCTION': 0,
        'RERA': 0,
        'BHK_NO.': 2,
        'BHK_OR_RK': 'BHK',
        'READY_TO_MOVE': 1,
        'RESALE': 1,
        'LONGITUDE': 22.5922,
        'LATITUDE': 88.4849
    }
    result = predict_house_price(sample_input, model)
    print(f"Sample Prediction: {result}")
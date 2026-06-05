import joblib
import xgboost as xgb
import os

model_path = r"c:\Users\ayush\Downloads\fianl mera project\models\voice_xgb_model.pkl"
try:
    model = joblib.load(model_path)
    print("Model type:", type(model))
    
    # Check if it's an XGBClassifier
    if hasattr(model, 'set_params'):
        print("Old params:", model.get_params().get('tree_method'))
        model.set_params(tree_method='hist', predictor='cpu_predictor')
        joblib.dump(model, model_path.replace('.pkl', '_fixed.pkl'))
        print("Saved CPU-fixed model using set_params")
    elif isinstance(model, xgb.Booster):
        model.set_param({'tree_method': 'hist', 'predictor': 'cpu_predictor'})
        joblib.dump(model, model_path.replace('.pkl', '_fixed.pkl'))
        print("Saved CPU-fixed booster")
except Exception as e:
    print("Error:", e)

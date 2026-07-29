# app.py
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
import os

app = Flask(__name__)
CORS(app)

# Load model and preprocessors
print("="*60)
print("🚀 Starting Churn Prediction API")
print("="*60)

try:
    model = joblib.load("saved_models/churn_model.pkl")
    print("✓ Model loaded successfully")
    
    # Load feature names
    X_train = pd.read_csv("Dataset/X_train.csv")
    feature_names = X_train.columns.tolist()
    print(f"✓ Features: {len(feature_names)}")
    
    # Load scaler if exists
    try:
        scaler = joblib.load("saved_models/scaler.pkl")
        print("✓ Scaler loaded")
    except:
        scaler = None
        print("⚠ No scaler found")
        
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

# ============================================
# API ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'service': 'Customer Churn Prediction API',
        'version': '1.0',
        'endpoints': {
            '/': 'GET - API Information',
            '/health': 'GET - Health Check',
            '/predict': 'POST - Predict single customer',
            '/predict_batch': 'POST - Predict multiple customers',
            '/features': 'GET - List all features'
        },
        'model_info': {
            'type': type(model).__name__,
            'features': len(feature_names)
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': len(feature_names) > 0
    })

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        'total_features': len(feature_names),
        'features': feature_names[:20],  # First 20 features
        'sample': X_train.head(2).to_dict('records') if 'X_train' in locals() else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a single customer
    Expected JSON format:
    {
        "features": {
            "tenure": 12,
            "monthly_charges": 70.5,
            "total_charges": 840.0,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
        if 'features' in data:
            customer_data = data['features']
        else:
            customer_data = data
        
        # Convert to DataFrame
        input_df = pd.DataFrame([customer_data])
        
        # Ensure all features exist
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[feature_names]
        
        # Scale if needed (for Logistic Regression)
        if scaler and type(model).__name__ == 'LogisticRegression':
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
        else:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
        
        # Prepare response
        churn_prob = float(probability[0][1])
        churn_class = int(prediction[0])
        
        response = {
            'prediction': churn_class,
            'churn_probability': round(churn_prob * 100, 2),
            'churn_risk': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low',
            'confidence': round(max(probability[0]) * 100, 2),
            'customer_id': data.get('customer_id', 'N/A')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict churn for multiple customers
    Expected JSON format:
    {
        "customers": [
            {"customer_id": 1, "tenure": 12, ...},
            {"customer_id": 2, "tenure": 24, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({'error': 'Missing "customers" key'}), 400
        
        customers = data['customers']
        results = []
        
        for customer in customers:
            # Convert to DataFrame
            input_df = pd.DataFrame([customer])
            
            # Ensure all features exist
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns
            input_df = input_df[feature_names]
            
            # Predict
            if scaler and type(model).__name__ == 'LogisticRegression':
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)
            else:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)
            
            churn_prob = float(probability[0][1])
            
            results.append({
                'customer_id': customer.get('customer_id', 'N/A'),
                'prediction': int(prediction[0]),
                'churn_probability': round(churn_prob * 100, 2),
                'churn_risk': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
            })
        
        return jsonify({
            'total_customers': len(results),
            'high_risk_customers': sum(1 for r in results if r['churn_risk'] == 'High'),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
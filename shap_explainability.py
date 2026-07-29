# shap_explainability.py - FIXED VERSION
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*60)

# Load model and data
model = joblib.load("saved_models/churn_model.pkl")
X_test = pd.read_csv("Dataset/X_test.csv")
X_train = pd.read_csv("Dataset/X_train.csv")

# Check model type
model_type = type(model).__name__
print(f"\nModel Type: {model_type}")

# Get feature names
feature_names = X_test.columns.tolist()
print(f"Features: {len(feature_names)}")

# ========================================
# CREATE APPROPRIATE EXPLAINER
# ========================================
print("\nCreating SHAP explainer...")

try:
    # Try TreeExplainer first (for tree-based models)
    if 'Tree' in model_type or 'XGB' in model_type or 'LGBM' in model_type or 'CatBoost' in model_type:
        print("Using TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value
    else:
        # Use KernelExplainer for Logistic Regression and other models
        print("Using KernelExplainer (works for all models)...")
        
        # Use a subset of data for faster computation
        background_data = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
        
        # For Logistic Regression, we need a prediction function
        def predict_proba(X):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return np.column_stack([1 - model.predict(X), model.predict(X)])
        
        explainer = shap.KernelExplainer(
            model.predict_proba, 
            background_data,
            link="logit"
        )
        
        # Use subset for faster computation
        test_sample = X_test.sample(min(50, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(test_sample, nsamples=100)
        expected_value = explainer.expected_value
        
        # If explainer returns list of values (for both classes)
        if isinstance(shap_values, list):
            # Use class 1 (churn) SHAP values
            shap_values = shap_values[1]
            expected_value = expected_value[1] if isinstance(expected_value, list) else expected_value
            X_test = test_sample
        else:
            X_test = test_sample
            
except Exception as e:
    print(f"⚠ Error with explainer: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: Use KernelExplainer with simplified approach
    from sklearn.linear_model import LogisticRegression
    
    # Create a simple prediction wrapper
    def f(X):
        return model.predict_proba(X)[:, 1]
    
    # Use background data
    background_data = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(f, background_data)
    
    # Sample test data
    test_sample = X_test.sample(min(50, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(test_sample, nsamples=100)
    expected_value = explainer.expected_value
    X_test = test_sample

print(f"✓ SHAP explainer created successfully!")

# ========================================
# 1. SUMMARY PLOT - Top Features
# ========================================
print("\n[1] Generating SHAP Summary Plot...")

try:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      show=False, max_display=15)
    plt.title("SHAP Feature Importance - Top 15 Features", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/shap_summary_plot.png")
except Exception as e:
    print(f"⚠ Could not generate summary plot: {e}")

# ========================================
# 2. BAR PLOT - Mean Absolute SHAP
# ========================================
print("\n[2] Generating SHAP Bar Plot...")

try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance - Mean |SHAP Value|", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/shap_bar_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/shap_bar_plot.png")
except Exception as e:
    print(f"⚠ Could not generate bar plot: {e}")

# ========================================
# 3. WATERFALL PLOT - Individual Customer
# ========================================
print("\n[3] Analyzing Individual Customer...")

try:
    # Pick a high-risk customer
    probabilities = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(probabilities)
    
    print(f"High-risk customer index: {high_risk_idx}")
    print(f"Churn probability: {probabilities[high_risk_idx]:.3f}")

    # For KernelExplainer, create Explanation object
    if hasattr(explainer, 'expected_value'):
        expected_val = expected_value
    else:
        expected_val = 0  # Fallback

    # Create waterfall plot
    plt.figure(figsize=(12, 8))
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values_single = shap_values[0][high_risk_idx] if len(shap_values) > 0 else shap_values[high_risk_idx]
    else:
        shap_values_single = shap_values[high_risk_idx]
    
    # Create explanation object
    exp = shap.Explanation(
        values=shap_values_single,
        base_values=expected_val if not isinstance(expected_val, list) else expected_val[0],
        data=X_test.iloc[high_risk_idx].values,
        feature_names=feature_names
    )
    
    shap.waterfall_plot(exp, show=False, max_display=15)
    plt.title(f"Customer {high_risk_idx} - Churn Explanation", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/shap_waterfall_customer.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/shap_waterfall_customer.png")
except Exception as e:
    print(f"⚠ Could not generate waterfall plot: {e}")

# ========================================
# 4. FORCE PLOT - Interactive HTML
# ========================================
print("\n[4] Creating SHAP Force Plot (HTML)...")

try:
    # For KernelExplainer
    if 'KernelExplainer' in str(type(explainer)):
        # Simplified force plot
        shap.initjs()
        
        # Create force plot for single prediction
        force_plot = shap.force_plot(
            expected_value if not isinstance(expected_value, list) else expected_value[0],
            shap_values[high_risk_idx] if isinstance(shap_values, list) else shap_values[high_risk_idx],
            X_test.iloc[high_risk_idx],
            feature_names=feature_names,
            matplotlib=False
        )
        
        # Save as HTML
        shap.save_html("results/shap_force_plot.html", force_plot)
        print("✓ Saved: results/shap_force_plot.html")
    else:
        # For TreeExplainer
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[high_risk_idx],
            X_test.iloc[high_risk_idx],
            feature_names=feature_names,
            matplotlib=False
        )
        shap.save_html("results/shap_force_plot.html", force_plot)
        print("✓ Saved: results/shap_force_plot.html")
except Exception as e:
    print(f"⚠ Could not create force plot: {e}")

# ========================================
# 5. FEATURE IMPORTANCE RANKING
# ========================================
print("\n[5] Feature Importance Ranking...")

try:
    # Calculate mean absolute SHAP values
    if isinstance(shap_values, list):
        shap_abs = np.abs(shap_values[0]).mean(axis=0)
    else:
        shap_abs = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': shap_abs
    }).sort_values('SHAP_Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save to CSV
    importance_df.to_csv("results/shap_feature_importance.csv", index=False)
    print("\n✓ Saved: results/shap_feature_importance.csv")
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features['Feature'], top_features['SHAP_Importance'])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 15 Feature Importances (SHAP)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/shap_feature_importance_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/shap_feature_importance_ranking.png")
    
except Exception as e:
    print(f"⚠ Could not generate feature importance ranking: {e}")

print("\n" + "="*60)
print("✅ SHAP ANALYSIS COMPLETED!")
print("📁 Results saved in 'results/' folder")
print("="*60)
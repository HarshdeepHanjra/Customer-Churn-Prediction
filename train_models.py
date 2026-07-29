# train_models.py
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import sys
from datetime import datetime
import matplotlib
from xgboost import XGBClassifier
matplotlib.use('Agg')  # For non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ========================================
# CREATE DIRECTORIES
# ========================================
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("\n" + "="*60)
print("MODEL TRAINING PIPELINE")
print("="*60)

# ========================================
# 1. LOAD DATA WITH ERROR HANDLING
# ========================================
print("\n[1] LOADING DATA...")
print("-"*60)

try:
    # Check if files exist
    required_files = [
        "Dataset/X_train.csv",
        "Dataset/X_test.csv", 
        "Dataset/y_train.csv",
        "Dataset/y_test.csv"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")
    
    X_train = pd.read_csv("Dataset/X_train.csv")
    X_test = pd.read_csv("Dataset/X_test.csv")
    y_train = pd.read_csv("Dataset/y_train.csv").values.ravel()
    y_test = pd.read_csv("Dataset/y_test.csv").values.ravel()
    
    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_test: {X_test.shape}")
    print(f"✓ y_train: {y_train.shape}")
    print(f"✓ y_test: {y_test.shape}")
    print(f"✓ Training churn rate: {y_train.mean()*100:.2f}%")
    print(f"✓ Test churn rate: {y_test.mean()*100:.2f}%")
    
except FileNotFoundError as e:
    print(f"\n✗ ERROR: {e}")
    print("\nPlease run 'train_test_split.py' first!")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERROR loading data: {e}")
    sys.exit(1)

# ========================================
# 2. HANDLE IMBALANCE WITH SMOTE
# ========================================
print("\n[2] HANDLING IMBALANCE WITH SMOTE...")
print("-"*60)

try:
    # Check if SMOTE is needed
    if y_train.mean() < 0.3:  # If churn rate < 30%
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"✓ SMOTE applied successfully!")
        print(f"  Original: {X_train.shape[0]} samples")
        print(f"  Resampled: {X_train_resampled.shape[0]} samples")
        print(f"  Original churn: {y_train.mean()*100:.2f}%")
        print(f"  New churn: {y_train_resampled.mean()*100:.2f}%")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        print(f"⚠ Churn rate {y_train.mean()*100:.2f}% > 30%, SMOTE not applied")
except Exception as e:
    print(f"⚠ SMOTE failed: {e}")
    print("  Using original data...")
    X_train_resampled, y_train_resampled = X_train, y_train

# ========================================
# 3. PREPARE FEATURES
# ========================================
print("\n[3] PREPARING FEATURES...")
print("-"*60)

# Store feature names
feature_names = X_train.columns.tolist()
print(f"✓ Features: {len(feature_names)}")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "saved_models/scaler.pkl")
print(f"✓ Scaler saved to 'saved_models/scaler.pkl'")

# ========================================
# 4. DEFINE MODELS
# ========================================
print("\n[4] DEFINING MODELS...")
print("-"*60)

# Calculate class weights for XGBoost
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"Scale pos weight: {scale_pos_weight:.3f}")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        class_weights=[1, scale_pos_weight],
        random_seed=42,
        verbose=False
    )
}

print(f"✓ Models defined: {len(models)}")

# ========================================
# 5. TRAIN AND EVALUATE
# ========================================
print("\n[5] TRAINING MODELS...")
print("="*60)

results = []
best_models = {}
failed_models = []

for idx, (name, model) in enumerate(models.items(), 1):
    print(f"\n[{idx}/{len(models)}] Training {name}...")
    print("-"*40)
    
    try:
        start_time = datetime.now()
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
            'Train_Time': round(train_time, 2)
        })
        
        # Save model
        model_filename = f"saved_models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        
        # Store for later use
        best_models[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'train_time': train_time
        }
        
        print(f"✓ {name} completed in {train_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
    except Exception as e:
        print(f"✗ {name} failed: {str(e)[:100]}")
        failed_models.append(name)

# ========================================
# 6. CHECK RESULTS
# ========================================
if len(results) == 0:
    print("\n" + "="*60)
    print("✗ ERROR: No models trained successfully!")
    print("="*60)
    sys.exit(1)

# ========================================
# 7. CREATE COMPARISON TABLE
# ========================================
print("\n[6] COMPARISON RESULTS")
print("="*60)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1-Score', ascending=False)

print("\nMODEL COMPARISON TABLE:")
print(df_results.to_string(index=False))

# Save comparison table
df_results.to_csv("results/model_comparison.csv", index=False)
print(f"\n✓ Saved to 'results/model_comparison.csv'")

# ========================================
# 8. IDENTIFY BEST MODEL
# ========================================
print("\n[7] BEST MODEL SELECTION")
print("="*60)

best_model_name = df_results.iloc[0]['Model']
best_f1 = df_results.iloc[0]['F1-Score']
best_recall = df_results.iloc[0]['Recall']

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Recall: {best_recall:.4f}")
print(f"   Accuracy: {df_results.iloc[0]['Accuracy']:.4f}")

# Save best model
best_model = best_models[best_model_name]['model']
joblib.dump(best_model, "saved_models/churn_model.pkl")
print(f"✓ Best model saved as 'saved_models/churn_model.pkl'")

# ========================================
# 9. CONFUSION MATRICES
# ========================================
print("\n[8] GENERATING CONFUSION MATRICES...")
print("-"*60)

try:
    # Create subplots
    n_models = len(best_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, data) in enumerate(best_models.items()):
        if idx < len(axes):
            cm = confusion_matrix(y_test, data['y_pred'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            axes[idx].set_title(f'{name}\nF1: {df_results[df_results["Model"]==name]["F1-Score"].values[0]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(len(best_models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved to 'results/confusion_matrices.png'")
    
except Exception as e:
    print(f"⚠ Could not save confusion matrices: {e}")

# ========================================
# 10. FEATURE IMPORTANCE
# ========================================
print("\n[9] FEATURE IMPORTANCE ANALYSIS...")
print("-"*60)

try:
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_imp_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', 
                   ax=ax, palette='viridis')
        ax.set_title(f'Top 15 Feature Importances - {best_model_name}')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance saved to 'results/feature_importance.png'")
    else:
        print("⚠ Model doesn't support feature importance")
        
except Exception as e:
    print(f"⚠ Could not generate feature importance: {e}")

# ========================================
# 11. ROC CURVES
# ========================================
print("\n[10] GENERATING ROC CURVES...")
print("-"*60)

try:
    plt.figure(figsize=(10, 8))
    
    for name, data in best_models.items():
        if len(np.unique(y_test)) > 1:  # Both classes present
            fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
            roc_auc = roc_auc_score(y_test, data['y_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved to 'results/roc_curves.png'")
    
except Exception as e:
    print(f"⚠ Could not generate ROC curves: {e}")

# ========================================
# 12. CLASSIFICATION REPORT
# ========================================
print("\n[11] CLASSIFICATION REPORT")
print("="*60)

try:
    report = classification_report(
        y_test, 
        best_models[best_model_name]['y_pred'],
        target_names=['No Churn', 'Churn']
    )
    print(report)
    
    # Save report
    with open('results/classification_report.txt', 'w') as f:
        f.write(f"Classification Report - {best_model_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
    print("✓ Report saved to 'results/classification_report.txt'")
    
except Exception as e:
    print(f"⚠ Could not generate classification report: {e}")

# ========================================
# 13. BUSINESS METRICS
# ========================================
print("\n[12] BUSINESS METRICS")
print("="*60)

try:
    # Average customer LTV (can be adjusted)
    avg_ltv = 500
    
    # Churn metrics
    actual_churn_rate = y_test.mean()
    predicted_churn_rate = best_models[best_model_name]['y_pred'].mean()
    
    total_customers = len(y_test)
    actual_churners = int(actual_churn_rate * total_customers)
    predicted_churners = int(predicted_churn_rate * total_customers)
    
    # Revenue calculations
    revenue_at_risk = predicted_churners * avg_ltv
    potential_savings = revenue_at_risk * 0.30  # 30% retention rate
    
    print(f"📊 Total Customers: {total_customers:,}")
    print(f"📊 Actual Churn Rate: {actual_churn_rate*100:.2f}% ({actual_churners} customers)")
    print(f"📊 Predicted Churn Rate: {predicted_churn_rate*100:.2f}% ({predicted_churners} customers)")
    print(f"\n💰 Revenue at Risk (Avg LTV ${avg_ltv:,}): ${revenue_at_risk:,.2f}")
    print(f"💰 Potential Revenue Saved (30% retention): ${potential_savings:,.2f}")
    
    # Business metrics for future reference
    business_metrics = {
        'Total_Customers': total_customers,
        'Actual_Churn_Rate': actual_churn_rate,
        'Predicted_Churn_Rate': predicted_churn_rate,
        'Actual_Churners': actual_churners,
        'Predicted_Churners': predicted_churners,
        'Revenue_at_Risk': revenue_at_risk,
        'Potential_Revenue_Saved': potential_savings,
        'Retention_Rate_Assumed': 0.30
    }
    
    pd.DataFrame([business_metrics]).to_csv("results/business_metrics.csv", index=False)
    print("\n✓ Business metrics saved to 'results/business_metrics.csv'")
    
except Exception as e:
    print(f"⚠ Could not calculate business metrics: {e}")

# ========================================
# 14. SUMMARY
# ========================================
print("\n" + "="*60)
print("✅ MODEL BUILDING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📁 Results saved to 'results/' folder:")
print(f"   - model_comparison.csv")
print(f"   - confusion_matrices.png")
print(f"   - roc_curves.png")
print(f"   - feature_importance.png")
print(f"   - classification_report.txt")
print(f"   - business_metrics.csv")
print(f"\n📁 Models saved to 'saved_models/' folder:")
print(f"   - churn_model.pkl (Best Model)")
print(f"   - scaler.pkl")
print(f"   - {best_model_name.lower().replace(' ', '_')}.pkl")
print("="*60)
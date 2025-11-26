import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

# local imports
from .utils import handle_error



def evaluate_model(model, X_test, y_test, model_name='Model'):
    try:
        print(f'Evaluating model: {model_name}...')
        
        # make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Yes')
        recall = recall_score(y_test, y_pred, pos_label='Yes')
        f1 = f1_score(y_test, y_pred, pos_label='Yes')
        
        # calculate AUC if probability predictions are available
        auc = None
        if y_pred_proba is not None:
            # convert target to binary for AUC calculation
            y_test_binary = (y_test == 'Yes').astype(int)
            auc = roc_auc_score(y_test_binary, y_pred_proba)
        
        # print results
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        if auc is not None:
            print(f'AUC-ROC: {auc:.4f}')
        
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
        
        print('\nConfusion Matrix:')
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        return metrics
    except Exception as e:
        handle_error(e, def_name='evaluate_model', msg=f'Model: {model_name}')
        return None


def compare_models(models_dict, X_test, y_test):
    try:
        print('Comparing multiple models...')
        results = []
        for model_name, model in models_dict.items():
            metrics = evaluate_model(model, X_test, y_test, model_name)
            results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc']
            })
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print('\n=== Model Comparison ===')
        print(results_df.to_string(index=False, float_format='%.4f'))
        return results_df
    except Exception as e:
        handle_error(e, def_name='compare_models')
        return None


def analyze_predictions(y_test, y_pred, sample_size=10):
    try:
        print('Analyzing predictions...')
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Correct': y_test == y_pred
        })
        sample_df = comparison_df.sample(n=min(sample_size, len(comparison_df)), random_state=42)
        print(f'\n=== Sample Predictions ({len(sample_df)} examples) ===')
        print(sample_df.to_string(index=False))
        
        # Accuracy breakdown
        accuracy_by_class = comparison_df.groupby('Actual')['Correct'].mean()
        print(f'\n=== Accuracy by Class ===')
        for actual_class, accuracy in accuracy_by_class.items():
            print(f'{actual_class}: {accuracy:.4f}')
        
        return comparison_df
    except Exception as e:
        handle_error(e, def_name='analyze_predictions')
        return None


def calculate_business_metrics(y_test, y_pred, cost_fn=100, cost_fp=50):
    try:
        print('Calculating business metrics...')
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        business_metrics = {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'cost_false_negatives': fn * cost_fn,
            'cost_false_positives': fp * cost_fp,
            'total_cost': total_cost,
            'cost_per_prediction': total_cost / len(y_test)
        }
        print(f'\n=== Business Metrics ===')
        print(f'True Negatives: {tn}')
        print(f'False Positives: {fp} (Cost: ${fp * cost_fp})')
        print(f'False Negatives: {fn} (Cost: ${fn * cost_fn})')
        print(f'True Positives: {tp}')
        print(f'Total Cost: ${total_cost}')
        print(f'Cost per Prediction: ${total_cost / len(y_test):.2f}')
        return business_metrics
    except Exception as e:
        handle_error(e, def_name='calculate_business_metrics')
        return None
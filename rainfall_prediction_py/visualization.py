import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# local imports
from .utils import handle_error


def setup_plot_style():
    try:
        print('Setting up plot style...')
        plt.style.use('default')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    except Exception as e:
        handle_error(e, def_name='setup_plot_style')


def plot_target_distribution(y, title='Target Distribution'):
    try:
        print(f'Plotting target distribution: {title}')
        setup_plot_style()
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # count plot
        y.value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title(f'{title} - Counts')
        ax1.set_xlabel('Rain Tomorrow')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # pie chart
        y.value_counts().plot(
            kind='pie', ax=ax2, autopct='%1.1f%%', 
            colors=['skyblue', 'salmon']
        )
        ax2.set_title(f'{title} - Proportions')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_target_distribution')


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    try:
        print(f'Plotting confusion matrix for {model_name}...')
        setup_plot_style()
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Rain', 'Rain'],
                    yticklabels=['No Rain', 'Rain'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_confusion_matrix')


def plot_feature_importance(importance_df, title='Feature Importance', top_n=10):
    try:
        print('Plotting feature importance...')
        setup_plot_style()
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'{title} (Top {top_n})')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_feature_importance')


def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    try:
        print(f'Plotting ROC curve for {model_name}...')
        setup_plot_style()
        if isinstance(y_true.iloc[0], str):     # convert target to binary if needed
            y_true_binary = (y_true == 'Yes').astype(int)
        else:
            y_true_binary = y_true
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_roc_curve')


def plot_model_comparison(results_df):
    try:
        print('Plotting model comparison...')
        setup_plot_style()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        _, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i], palette='Set2')
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # add value labels on bars
                for j, v in enumerate(results_df[metric]):
                    if pd.notna(v):
                        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_model_comparison')


def plot_seasonal_patterns(df):
    try:
        print('Plotting seasonal patterns...')
        setup_plot_style()
        if 'Season' in df.columns and 'RainToday' in df.columns:
            seasonal_rain = df.groupby(['Season', 'RainToday']).size().unstack(fill_value=0)
            seasonal_rain_pct = seasonal_rain.div(seasonal_rain.sum(axis=1), axis=0) * 100
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # count plot
            seasonal_rain.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
            ax1.set_title('Rainfall Count by Season')
            ax1.set_xlabel('Season')
            ax1.set_ylabel('Count')
            ax1.legend(['No Rain', 'Rain'])
            ax1.tick_params(axis='x', rotation=45)
            
            # percentage plot
            seasonal_rain_pct.plot(kind='bar', ax=ax2, color=['skyblue', 'salmon'])
            ax2.set_title('Rainfall Percentage by Season')
            ax2.set_xlabel('Season')
            ax2.set_ylabel('Percentage')
            ax2.legend(['No Rain', 'Rain'])
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_seasonal_patterns')


def plot_location_patterns(df):
    try:
        print('Plotting location patterns...')
        setup_plot_style()
        if 'Location' in df.columns and 'RainToday' in df.columns:
            # create location analysis
            location_rain = df.groupby(['Location', 'RainToday']).size().unstack(fill_value=0)
            location_rain_pct = location_rain.div(location_rain.sum(axis=1), axis=0) * 100
            
            plt.figure(figsize=(12, 8))
            location_rain_pct['Yes'].plot(kind='bar', color='salmon')
            plt.title('Rainfall Percentage by Location')
            plt.xlabel('Location')
            plt.ylabel('Rainfall Percentage (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_location_patterns')


def plot_correlation_heatmap(df, figsize=(12, 10)):
    try:
        print('Plotting correlation heatmap...')
        setup_plot_style()
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            plt.figure(figsize=figsize)
            correlation_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                    cmap='coolwarm', center=0, fmt='.2f',
                    square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        handle_error(e, def_name='plot_correlation_heatmap')

import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd

# local imports
from .utils import handle_error


def create_preprocessor(numeric_features, categorical_features):
    try:
        print('Creating preprocessing pipeline...')
        # define preprocessing steps
        numeric_transformer = StandardScaler()
        
        # create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', 'passthrough', categorical_features)
            ]
        )
        
        return preprocessor
    except Exception as e:
        handle_error(e, def_name='create_preprocessor')
        return None


def encode_categorical_features(X_train, X_test, categorical_features):
    try:
        print('Encoding categorical features...')
        encoders = {}
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for feature in categorical_features:
            encoder = LabelEncoder()
            X_train_encoded[feature] = encoder.fit_transform(X_train[feature])
            X_test_encoded[feature] = encoder.transform(X_test[feature])
            encoders[feature] = encoder
        
        return X_train_encoded, X_test_encoded, encoders
    except Exception as e:
        handle_error(e, def_name='encode_categorical_features')
        return X_train, X_test, {}


def create_random_forest_pipeline(preprocessor, random_state=42):
    try:
        print('Creating Random Forest pipeline...')
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
        return pipeline
    except Exception as e:
        handle_error(e, def_name='create_random_forest_pipeline')
        return None


def create_logistic_regression_pipeline(preprocessor, random_state=42):
    try:
        print('Creating Logistic Regression pipeline...')
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
        return pipeline
    except Exception as e:
        handle_error(e, def_name='create_logistic_regression_pipeline')
        return None


def train_random_forest(X_train, y_train, preprocessor, random_state=42, cv_folds=5, n_jobs=-1):
    try:
        print('Training Random Forest...')
        rf_pipeline = create_random_forest_pipeline(preprocessor, random_state=random_state)
        rf_param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        rf_grid = GridSearchCV(
            rf_pipeline,
            rf_param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        rf_grid.fit(X_train, y_train)
        print(f'Best Random Forest parameters: {rf_grid.best_params_}')
        print(f'Best Random Forest CV score: {rf_grid.best_score_:.4f}')
        return rf_grid
    except Exception as e:
        handle_error(e, def_name='train_random_forest')
        return None


def train_logistic_regression(X_train, y_train, preprocessor, random_state=42, cv_folds=5, n_jobs=-1):
    try:
        print('Training Logistic Regression...')
        lr_pipeline = create_logistic_regression_pipeline(preprocessor, random_state=random_state)
        lr_param_grid = {
            'classifier__solver': ['liblinear'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight': [None, 'balanced']
        }
        lr_grid = GridSearchCV(
            lr_pipeline,
            lr_param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        lr_grid.fit(X_train, y_train)
        print(f'Best Logistic Regression parameters: {lr_grid.best_params_}')
        print(f'Best Logistic Regression CV score: {lr_grid.best_score_:.4f}')
        return lr_grid
    except Exception as e:
        handle_error(e, def_name='train_logistic_regression')
        return None


def get_feature_importance(model, feature_names, top_n=10):
    try:
        print('Getting feature importance...')
        importance = model.best_estimator_.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance_df.head(top_n)
    except Exception as e:
        handle_error(e, def_name='get_feature_importance')
        return pd.DataFrame()

import pandas as pd
from sklearn.model_selection import train_test_split

# local imports
from .utils import handle_error


def date_to_season(date):
    try:
        month = date.month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        elif month in [9, 10, 11]:
            return 'Spring'
        else:
            return 'Invalid month, try again'
    except Exception as e:
        handle_error(e, def_name='date_to_season', msg=f'Date: {date}')
        return None


def load_and_clean_data(url, handle_na='drop'):
    try:
        print('Loading data...')
        df = pd.read_csv(url)
        print(f'Initial dataset shape: {df.shape}')
        print(f'Missing values before cleaning: {df.isnull().sum().sum()}')
        if handle_na not in ['drop', 'fbfill', 'mean', 'mode']:
            raise ValueError('Invalid handle_na option. Use "drop", "ffill", "bfill", "mean", or "mode".')
        if handle_na == 'drop':
            df = df.dropna()
        elif handle_na == 'fbfill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif handle_na == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif handle_na == 'mode':
            df = df.fillna(df.mode().iloc[0])
        else:
            df = df.fillna(method=handle_na)
        print(f'Dataset shape after handling missing values: {df.shape}')
        return df
    except Exception as e:
        handle_error(e, def_name='load_and_clean_data', msg=f'URL: {url}, handle_na: {handle_na}')
        return None


def preprocess_data(df, target_locations):
    try:
        print('Preprocessing data...')
        # rename columns to avoid data leakage
        df = df.rename(columns={
            'RainToday': 'RainYesterday',
            'RainTomorrow': 'RainToday'
        })
        
        # filter to target locations
        df = df[df.Location.isin(target_locations)]
        print(f'Dataset shape after location filtering: {df.shape}')
        
        # convert Date to datetime and extract season
        df['Date'] = pd.to_datetime(df['Date'])
        df['Season'] = df['Date'].apply(date_to_season)
        
        # drop the Date column
        df = df.drop(columns=['Date'])
        
        return df
    except Exception as e:
        handle_error(e, def_name='preprocess_data')
        return None


def split_features_target(df, target_column='RainToday'):
    try:
        print('Splitting features and target...')
        X = df.drop(columns=target_column, axis=1)
        y = df[target_column]
        
        print(f'Features shape: {X.shape}')
        print(f'Target distribution:\n{y.value_counts()}')
        print(f'Target proportion:\n{y.value_counts(normalize=True)}')
        
        return X, y
    except Exception as e:
        handle_error(e, def_name='split_features_target', msg=f'Target column: {target_column}')
        return None, None


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    try:
        print('Splitting data into training and test sets...')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        print(f'Training set size: {X_train.shape[0]}')
        print(f'Test set size: {X_test.shape[0]}')
        print(f'Training target distribution:\n{y_train.value_counts()}')
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        handle_error(e, def_name='train_test_split_data', msg=f'test_size: {test_size}, random_state: {random_state}')
        return None, None, None, None


def get_feature_types(X_train):
    try:
        print('Identifying feature types...')
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f'Numeric features ({len(numeric_features)}): {numeric_features}')
        print(f'Categorical features ({len(categorical_features)}): {categorical_features}')
        
        return numeric_features, categorical_features
    except Exception as e:
        handle_error(e, def_name='get_feature_types')
        return [], []
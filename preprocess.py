import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Drop unnecessary ID columns
    df = df.drop(columns=['nameOrig', 'nameDest'])

    # Encode transaction type
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    # Features & Target
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    return X, y

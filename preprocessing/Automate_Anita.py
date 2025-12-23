import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(input_path, output_path):
    
    df = pd.read_csv(input_path)
    
    categorical_cols = [
        'gender',
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    df['average_score'] = (
        df['math score'] +
        df['reading score'] +
        df['writing score']
    ) / 3
    
    df['target'] = df['average_score'].apply(lambda x: 1 if x >= 70 else 0)
    
    numerical_cols = ['math score', 'reading score', 'writing score']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    input_path = "siswa_raw/StudentsPerformance.csv"
    output_path = "preprocessing/siswa_preprocessing.csv"

    preprocess_data(input_path, output_path)
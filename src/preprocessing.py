from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    categorical_features = [
        'mainroad','guestroom','basement',
        'hotwaterheating','airconditioning',
        'prefarea','furnishingstatus'
    ]
    numeric_features = ['area','bedrooms','bathrooms','stories','parking']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ]
    )
    return preprocessor
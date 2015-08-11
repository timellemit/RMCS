from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.preprocessing.label import LabelEncoder
from sklearn.preprocessing.data import MinMaxScaler
from scipy.io.arff.arffread import loadarff
import pandas as pd

def preprocess(dataset_path):
    vectorizer = DictVectorizer(sparse=False)
    label_encoder = LabelEncoder()
    normalizer = MinMaxScaler()
    data, _ = loadarff(dataset_path)
    df = pd.DataFrame.from_records(data).rename(columns=str.lower)
    # build classes
    y = label_encoder.fit_transform(df['class'])
#         print "y", y
    # build features
    x = df.drop('class', axis=1).to_dict('records')
#         print "x",x
    # transform nominal features
    x = vectorizer.fit_transform(x)

    # scaling
    x = normalizer.fit_transform(x)
    return x, y
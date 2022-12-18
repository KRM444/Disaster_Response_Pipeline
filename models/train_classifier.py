# import libraries
import sys
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import re
import pandas as pd
import numpy as np
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')


def load_data(database_filepath):
    """Load processed data

    Args:
        database_filepath {str} -- The filepath of SQLite database

    Returns:
        Dataframe -- Pandas Dataframe of processed data
    """
    # Load data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM df", engine)
    # Split data to feature and target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, list(Y.columns.values)


def tokenize(text):
    # Find and replace URL's with a placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # Normalize, tokenize, and remove punctuation
    # word_tokenize simply splits up a sentence string into a list containing split elements as the module sees fit
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove Stopwords
    # stopwords in nltk are the most common words in data. They are words that you do not want to use to describe the topic of your content.
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatization
    # Lemmatization is the process of replacing a word with its root or head word called lemma
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def build_model():
    """Build a machine learning pipeline

    Returns:
        object -- Grid search object
    """
    # Build pipline
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(
                            RandomForestClassifier(verbose=False)))
                        ])

    # define parameters for GridSearchCV
    parameters = {'vect__min_df': [5],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [30],
                  'clf__estimator__min_samples_split': [2, 5]
                  }

    # create gridsearch object and return as final model pipeline
    cv_rf = GridSearchCV(pipeline, param_grid=parameters,
                         scoring='f1_weighted')

    model = cv_rf
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Calculate and display the evaluation of ML model

    Arguments:
        model {object} -- Grid search object
        X_test {dataframe} -- Pandas Dataframe for features
        Y_test {dataframe} -- Pandas Dataframe for targets
        category_names {list} -- list of column names of targets
    """
    y_hat = model.predict(X_test)
    y_true = np.array(Y_test)
    print(classification_report(y_true=y_true,
                                y_pred=y_hat,
                                target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:  # does this need to equal 4 since we have 4 functions?

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

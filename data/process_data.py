import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and Merge messages_df and categories_df
    ------------------------------------------------------------
    Args:
        messages_filepath   {str} : CSV filepath of message data.
        categories_filepath {str} : CSV filepath of categories data.

    Returns:
        Dataframe -- Merged Pandas Dataframe 
    ------------------------------------------------------------
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merged data
    df = messages.merge(categories, how='inner', on='id')

    return df


def clean_data(df):
    """Data Cleaning
    ----------------------------------------------------------------------------------------------------------------
    Args:
        df   {pandas dataframe} : merged df of 'messages' & 'categories'

    Returns:
        df   {pandas dataframe} : Cleaned dataframe 

    1. Clean the 'categories' column by:
        a) Split the values in the categories column on the ; character so that each value becomes a separate column.
        b) Use the first row of categories dataframe to create column names for the categories data. 
        c) Rename columns of categories with new column names.
        d) Convert category values to just numbers 0 or 1 *(dtype = int)*.
        e) Replace categories column in df with new category columns.

    2. Drop unnecessary features and duplicates:
        a) Drop 'child_alone' column because it only has 1 value, therefore it's meaningless
        b) Drop 'related=2' values because it constitutes for a small majority of the datasets and is misleading
        c) Drop all further duplicates in the dataset
    ----------------------------------------------------------------------------------------------------------------
    """
    # 1a
    categories = df['categories'].str.split(';', expand=True)

    # 1b
    row = categories.iloc[0, :]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()

    # 1c
    categories.columns = category_colnames

    # 1d
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    # For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        categories[column] = categories[column].astype(int)

    # 1e
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # 2a,b,c
    df.drop('child_alone', axis=1, inplace=True)
    df = df[df['related'] != 2]
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database

    Args:
        df   {pandas dataframe} : Cleaned pandas dataframe 
        database_filename {str} : Table name <df>
    """
    database_filename
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

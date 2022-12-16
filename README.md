# Building a Disaster Response Pipeline
###### An ETL application with python, sqlalchemy, NLP, and machine learning
![AIRBNB Seattle](https://i.ytimg.com/vi/oRiLLd2hX0E/maxresdefault.jpg)

In this journey, we will analyze disaster data from [Appen](https://appen.com/) to build a model for an API that classifies disaster messages. A machine learning pipeline will be created to categorize these events such that we can send the messages to an appropriate disaster relief agency. A web app will be developed where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

This project will adhere to the **Cr**oss **I**ndustry **S**tandard **P**rocess for **D**ata **M**ining (*CRISP-DM*) below:

![CRISP-DM Process](http://optimumsportsperformance.com/blog/wp-content/uploads/2019/12/Screen-Shot-2019-12-14-at-10.12.35-PM-768x423.png)

*- Image provided courtesy of [Patrick Ward, Ph.D.](http://optimumsportsperformance.com/blog/data-analysis-template-in-r-markdown-jupyter-notebook/).* 

 
## 1) 👨‍💼🤝👨‍💼Business Understanding
There were 323 recorded natural disasters worldwide in 2022. Although most people only hear the Amber Alerts go off 🔊, there are disaster response teams in place, and they need to respond fast! I think we can agree that that's a difficult job. Now, imagine how difficult that must be with a vast amount of data coming through several channels, and one has to decipher which data is actionable quickly. That job just became more difficult.

Creating solutions that help disaster response teams make the right decisions fast would prove immensely helpful to our society in such events. Therefore, introductory projects such as this one can be scaled to near perfection (as they may be with [Palantir's](https://www.palantir.com/) many decision-making offerings). Open-source connoisseurs can contribute to similar projects and help protect people worldwide, one disaster at a time.

Caveat: this project is a base case for learning purposes. It can be scaled for actual usability with superior: NLP, statistical applications, and data warehouses.

## 2) 👨‍💻Data Understanding/Preparation: ETL Pipeline 
![ETL](https://rivery.io/wp-content/uploads/2020/05/ETL-Process-for-linkedin3-1024x535.png)

Altogether, there are 2 datasets: 
- **messages.csv**: contains id, message that was sent, and the classified genre of the message (news, direct, social)
 
- **categories.csv**: contains id and categories of the message (related;request;offer;aid_related; etc.) 
    - Values in the 'categories' column were split on the ';' character such that each value became a separate column
    - The first row of the categories data frame was used to create column names for the categories data
    - Category values will be converted to binary, 0 or 1, and the datatype will be converted to int.

Before:
![categories1](https://user-images.githubusercontent.com/35614192/208192745-d29a895d-12f7-4580-98a5-d9264e3e4f6b.jpg)


After Cleaning 🧹🧹🧹
![categories2](https://user-images.githubusercontent.com/35614192/208192762-8ac54705-2478-4d21-b75c-6a4e71c87297.jpg)


- Duplicate entries were dropped
- Several features are `HIGHLY imbalanced`, there are a few recipes to cure this, such as: **Under from majority class and Over Samplings from minority class, etc.** However, I will skip this due to time constraints (the models take over 7 hours to train and test on my local machine🙃).
- `related=2` was dropped because it represents a minority of the feature.
- There is only one class for `child_alone`, so this column is meaningless and thus dropped.

Both of the cleaned datasets were merged on the 'id' column and loaded to an `sqlite database`
```
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite:///processed_data.db')
df.to_sql('df', engine, index=False)
```

## 3)🤖👷‍♂️ Modeling: ML Pipeline 
Next, the `processed_data.db` was loaded from the sqlalchemy database and prepared to leverage NLP foundations such as tokenization, removing stopwords/punctuation, lemmatization, and normalizing the text before applying it on the data and syncing it within the Machine Learning Pipeline:

```
engine = create_engine('sqlite:///processed_data.db')
df = pd.read_sql("SELECT * FROM df", engine)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    # Find and replace URL's with a placeholder 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # Normalize, tokenize, and remove punctuation
    # word_tokenize simply splits up a sentence string into a list containing          split elements as the module sees fit
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # Remove Stopwords
    # stopwords in nltk are the most common words in data. They are words that you do not want to use to describe the topic of your content.  
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Lemmatization (truncates word to root word, if it exists)
    # Lemmatization is the process of replacing a word with its root or head word called lemma 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens
```

Following is an example of the `token` function:

![tokenization_example](https://user-images.githubusercontent.com/35614192/208193053-614e3bcb-be2e-49c5-871d-bb14e5246151.jpg)


Notice how it:
* `Normalizes` the text: "IKEA" becomes "ikea", and the period "." is removed
* Removes common `stopwords`: "I", "will", "be", "to", etc.
* Performs `lemmatization`: truncates "things" to "thing"


Next, we will implement the ML Pipeline and fit it on the training data:
```
pipeline = Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer = tokenize)),
                    ('tfidf_transformer', TfidfTransformer()),
                    ('MultiOutput_Random_Forest_Classifier', MultiOutputClassifier(RandomForestClassifier()))
                    ])
                    
pipeline.fit(X_train, Y_train)
```

#### A mini breakdown of what's happening:

As you may know, although humans can understand the meaning behind text, machines are not as capable. Therefore, we have to convert the text data to numerical and then feed it through a pipeline consisting of `CountVectorizer, TfidTransformer, and an ML algorithm`.  

##### CountVectorizer
> Converts a collection of text documents to a matrix of token counts, and stores the elements in Compressed Sparse Row format. - sklearn
* In English, it's a transformer that extracts each word in the provided text/document and creates a column for each word (feature). 
* Next, we can apply <transformer.toarray()> to get values for the matrix where each row represents the frequency of the word/feature/column occurrence.
* Here's a simple visual depiction:

![CountVector_Example](https://user-images.githubusercontent.com/35614192/208193011-8eb3a7be-19d4-4222-bd30-61120fb97c31.jpg)


##### TfidTransformer
* Tf-idf = Term Frequency — Inverse Document Frequency
* [Here](https://towardsdatascience.com/tf-term-frequency-https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558) is well put together explanation by Yassine Hamdaoui if you're interested.

#### Multi Output Random Forest Classifier 
> A multi-output problem is a supervised learning problem with several outputs to predict - sklearn
* Our goal here is to apply a classification rule that can map to multiple outputs (in math terms, that translates to: $\hat{v}_1 = {\hat{y}_1 ∈ Y_1,  \hat{y}_2 ∈ Y_2,..., \hat{y}_n ∈ Y_n}$)
>The random forest is a classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.



## 3) Evaluation
We will use the `weighted avg F1 Score` to measure the model's performance. It is calculated by taking the mean of all per-class F1 scores while considering each class's support. Consider the following for a single variable **multiplied by the variables support** because the support will "balance" the imbalanced dataset to a degree:

![F1_Score](https://miro.medium.com/max/720/1*9uo7HN1pdMlMwTbNSdyO3A.webp)

Where

![Precision](https://miro.medium.com/max/640/1*VXnUvOEdf3IiYVCD6Wd2vg.webp)

and

![Recall](https://miro.medium.com/max/640/1*Aj3aYW4vwYAoJqyL36PVtQ.webp)

#### Quick notes on the F1 Score:
- **Penalizes incorrectly classified cases, advantageous when the data is imbalanced, in which case the 'Accuracy' metric is useless.**
- It's a weighted harmonic mean of precision and recall. The closer to 1, the better the model.
- F1 Score becomes 1 only when precision and recall are both 1.
- F1 score becomes high only when both precision and recall are high.

The weighted average F1 Score for the model was 0.57. Not very impressive, but it's a start.

To improve the Score, I implemented the **AdaBoost ensemble method** and performed hyperparameter tuning via GridSearchCV (best_params_).

This ***boosted*** the Weighted F1 Score to 0.61, a ~7% increase. So, we can save this model to a pickle file for future reference.

## 4) 🚀 Deployment: Flask app
Lastly, the file structure of the project:

```
- 📂app
| - template
| |- master.html              # main page of web app
| |- go.html                  # classification result page of web app
|- run.py                     # Flask file that runs app

- 📂data 
|- disaster_categories.csv    # data to process 
|- disaster_messages.csv      # data to process
|- process_data.py            # ETL to clean data and output to 'processed_data.db'
|- processed_data.db          # database to save clean data to

- 📂models 
|- train_classifier.py        # ML 
|- classifier.pkl             # saved model 

- README.md 📚
```

To access the Flask web app:
1. Open the terminal and navigate to the 📂Project folder
    - **Enter:** python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/processed_data.db
    - **Enter:** python models/train_classifier.py data/processed_data.db models/classifier.pkl

2. Navigate to the 📂app folder:
    - **Enter:**: cd app
    - **Enter:**: python run.py

3. Open a web browser and go to http://loclhost:3001

4. Experiment!


#### Sample:
https://user-images.githubusercontent.com/35614192/208192912-f3c64138-71f3-4f6e-9b66-e6e8b2247af5.mov



## Closing Remarks:

In this journey, we discussed developing ETL, ML Pipeline, and Model Evaluation.



Although the model score is not as high as I would like it to be, it may be improved through several avenues of training, such as:



- Dealing with imbalanced data via under-sampling the minority class

- Dropping certain features to improve model speed and score

- Applying additional NLP techniques and evaluating performance

- etc.



That’s all for now, folks, until next time. Thanks for reading.

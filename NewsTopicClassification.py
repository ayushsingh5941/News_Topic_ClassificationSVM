# Importing data sets
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
# Extracting features using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
# Importing svc
# Linear SVC as its faster
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
# import pipeline to tweak both SVC and tf-idf
from sklearn.pipeline import Pipeline
import timeit
import re
# sklearn svm internally handles multi class classifier problems we just need to add or remove the categories

categories = None
data_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories,random_state=42)
lemmatizer = WordNetLemmatizer()
all_names = set(names.words())

def letters_only(astr):
    for c in astr:
        if not c.isalpha():
            return False
    return True


def clean_text(docs):
    """Function to clean text removing names and non letters"""
    documents = []
    for sen in range(0, len(docs)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(docs[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [lemmatizer.lemmatize(word) for word in document if letters_only(word) and word not in all_names]
        document = ' '.join(document)

        documents.append(document)
    return documents


x_cleaned_test = clean_text(data_test.data)
y_test = data_test.target
x_cleaned_train = clean_text(data_train.data)
y_train = data_train.target
from collections import Counter
Counter(y_train)
# Cross validation for checking optimal C score in svc
# Instead of using above written code for cross validation we will use GridSearchCv from sci-kit learn
# We will find out best estimator for both tf-idf and C in SVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svc', LinearSVC()),
])

parameters_pipeline = {
    'tfidf__max_df': (0.25, 0.5),
    'tfidf__max_features': (40000, 50000),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'svc__C': (0.1, 1, 10, 100),
}

grid_search = GridSearchCV(pipeline, parameters_pipeline, n_jobs=-1, cv=3)

start_time = timeit.default_timer()
grid_search.fit(x_cleaned_train, y_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

print(grid_search.best_params_)
print(grid_search.best_score_)
pipeline_best = grid_search.best_estimator_
accuracy = pipeline_best.score(x_cleaned_test, y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))

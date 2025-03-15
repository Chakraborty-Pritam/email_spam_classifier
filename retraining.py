import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set the NLTK data path (if necessary)
nltk.data.path.append("C:/Users/pritam/AppData/Roaming/nltk_data")

# Initialize PorterStemmer
ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters and numbers
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# Step 1: Load the dataset
df = pd.read_csv('C:/Users/pritam/Downloads/spam.csv', encoding='latin-1')

# Keep only the relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Check the distribution of spam and ham
print("Original Dataset:")
print(df['label'].value_counts())

# Step 2: Balance the dataset
spam = df[df['label'] == 'spam']
ham = df[df['label'] == 'ham']

# Undersample ham to match the number of spam examples
ham_undersampled = resample(ham, replace=False, n_samples=len(spam), random_state=42)

# Combine the undersampled ham and spam
balanced_df = pd.concat([ham_undersampled, spam])

# Check the new distribution
print("\nBalanced Dataset:")
print(balanced_df['label'].value_counts())

# Step 3: Preprocess the data
balanced_df['transformed_message'] = balanced_df['message'].apply(transform_text)

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(balanced_df['transformed_message'], balanced_df['label'],
                                                    test_size=0.2, random_state=42)

# Step 5: Train the TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 6: Train and evaluate multiple models
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True)
}

best_accuracy = 0
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Check if this model is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")

# Step 7: Save the best model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nSaved the best model ({best_model_name}) and vectorizer to disk.")
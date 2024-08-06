import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('Reviews.csv')

# Ensure there are no missing values
data.dropna(inplace=True)

# Define text preprocessing function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to the data
X = data['Review'].apply(clean_text)
y = data['Liked']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer with trigrams and more features
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)

# Fit and transform the training data, and transform the test data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the vectorizer for future use
joblib.dump(vectorizer, 'vectorizer.pkl')

# Initialize SVM model
svm_model = SVC(kernel='linear')

# Grid Search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_vectorized, y_train)

# Use the best estimator from grid search
best_svm_model = grid.best_estimator_

# Predict on the test set
y_pred = best_svm_model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Improved Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Save the model for future use
joblib.dump(best_svm_model, 'svm_model.pkl')

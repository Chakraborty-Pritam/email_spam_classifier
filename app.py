from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
import mysql.connector
from scipy.sparse import csr_matrix
import warnings

# Suppress Scikit-Learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.secret_key = 'pc67676565464678ypc'  # Replace with a secure secret key

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the TfidfVectorizer and MultinomialNB model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Function to classify an email
def classify_email(email_text):
    # Preprocess the email text
    transformed_text = transform_text(email_text)

    # Transform the text using the TfidfVectorizer
    vector_input = tfidf.transform([transformed_text])

    # Make a prediction using the MultinomialNB model
    prediction = model.predict(vector_input)[0]

    return "Spam" if prediction == "spam" else "Not Spam"

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')


# Function to preprocess text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters and numbers
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if
            word not in nltk.corpus.stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)


# Define your database connection
db = mysql.connector.connect(
    host="localhost",
    user="pritampch",  # Replace with your MySQL username
    password="root",  # Replace with your MySQL password
    database="smc"  # Replace with your database name
)


# Home route
@app.route('/')
def home():
    return render_template('home.html')


# About route
@app.route('/about')
def about():
    return render_template('about.html')


# Index route (dashboard)
@app.route('/index')
def index():
    # Check if the user is logged in
    if 'user' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('signin'))  # Redirect to sign-in if not logged in


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('signin'))

    # Get the email text from the form
    input_sms = request.form.get('message')

    # Preprocess the email text
    transformed_sms = transform_text(input_sms)

    # Transform the text using the TfidfVectorizer
    vector_input = tfidf.transform([transformed_sms])

    # Make a prediction using the MultinomialNB model
    result = model.predict(vector_input)[0]
    prediction = "Spam" if result == 1 else "Not Spam"

    # Save the prediction to the database
    cur = db.cursor()
    try:
        user_id = session['user'][0]  # Assuming user ID is stored in the session
        cur.execute("INSERT INTO classifications (user_id, email_text, classification) VALUES (%s, %s, %s)",
                    (user_id, input_sms, prediction))
        db.commit()
    except mysql.connector.Error as err:
        return f"Database error: {err}"
    finally:
        cur.close()

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)


# Sign-in route
@app.route('/signin')
def signin():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('signin.html')


# Sign-up route
@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


# Register route
@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Debugging: Print form data
        print(f"Full Name: {full_name}, Username: {username}, Email: {email}, Phone: {phone}, Password: {password}")

        # Check if passwords match
        if password != confirm_password:
            return "Password and Confirm Password do not match."

        # Insert data into MySQL
        cur = db.cursor()
        try:
            cur.execute("INSERT INTO users (full_name, username, email, phone, password) VALUES (%s, %s, %s, %s, %s)",
                        (full_name, username, email, phone, password))
            db.commit()
            flash('Registration successful', 'success')
            return redirect(url_for('signin'))
        except mysql.connector.Error as err:
            return f"Database error: {err}"
        finally:
            cur.close()

    return "Invalid request method"


# Login route
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember_me = request.form.get('remember_me')  # Get the 'remember_me' checkbox value

        # Query the database to check if the email and password match
        cur = db.cursor()
        try:
            cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
            user = cur.fetchone()
            cur.close()

            if user:
                session['user'] = user  # Store user data in the session
                if remember_me:
                    session.permanent = True  # Make the session permanent if 'remember_me' is checked
                return redirect(url_for('index'))
            else:
                return "Login failed. Check your email and password."
        except mysql.connector.Error as err:
            return f"Database error: {err}"

    return "Invalid request method"


# Logout route
@app.route('/logout')
def logout():
    # Clear the user session to log out
    session.pop('user', None)
    return redirect(url_for('home'))  # Redirect to the home page after logging out


if __name__ == '__main__':
    app.run(debug=True)
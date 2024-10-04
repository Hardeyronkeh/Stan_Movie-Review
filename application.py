from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved Logistic Regression model and vectorizer
lr_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  # Default no result
    if request.method == 'POST':
        # Get review input from the form
        review = request.form['review']
        
        # Preprocess the review (TF-IDF transformation)
        review_tfidf = vectorizer.transform([review])
        
        # Predict sentiment using Logistic Regression model
        prediction = lr_model.predict(review_tfidf)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
    
    # Render the HTML page and pass the result if available
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

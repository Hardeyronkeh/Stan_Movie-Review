This project is a machine learning-based sentiment analysis system that classifies movie reviews as either positive or negative. The sentiment classifier is built using a Logistic Regression model, trained on a dataset of movie reviews, and is deployed as a Flask web application for real-time predictions.

Table of Contents
Project Overview
Model Training
Technologies Used
Installation and Setup
How to Run
How to Use the App
Project Structure
Future Enhancements
License
Project Overview
The project uses a Logistic Regression model for binary sentiment classification of movie reviews. Each review is preprocessed, tokenized, transformed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, and fed into the Logistic Regression model to predict whether the sentiment is positive (1) or negative (0).

The trained model and vectorizer are saved and loaded during deployment using Flask. The user interface allows a user to input a movie review and receive an analysis of whether the sentiment is positive or negative.

Model Training
The model is trained using the following steps:

Data Loading: The dataset of movie reviews is loaded and separated into training and testing sets.
Text Preprocessing:
Convert text to lowercase.
Remove punctuation.
Tokenize text into words.
Remove stopwords.
TF-IDF Vectorization: The processed text data is transformed using TF-IDF vectorization, converting text into numerical values that are fed into the model.
Model Building: The Logistic Regression model is trained on the vectorized text data.Model Evaluation: The model is evaluated using accuracy metrics.
Technologies Used
Python (3.x)
Flask - For deploying the web application.
scikit-learn - For model training and text vectorization.
joblib - For saving and loading the model and vectorizer.
HTML/CSS - For the front-end user interface.
Jinja2 - For template rendering in Flask.
Installation and Setup
Prerequisites
Python 3.x
Virtual environment tool (optional but recommended)
pip (Python package manager)

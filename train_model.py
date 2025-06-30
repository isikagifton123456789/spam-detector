"""
Email Spam Detection Model Training Script

This script demonstrates how to train a spam detection model and save the required .pkl files.
You'll need to replace the sample data with your actual dataset.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
import re

def preprocess_text(text):
    """
    Enhanced preprocess text for better spam detection
    """
    # Convert to lowercase
    text = text.lower()
    
    # Keep certain patterns that are important for spam detection
    # Replace URLs but keep the fact that there was a URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL_LINK ', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' WEBSITE_LINK ', text)
    
    # Replace phone numbers but keep the pattern
    text = re.sub(r'\b\d{1}-?\d{3}-?\d{3}-?\d{4}\b', ' PHONE_NUMBER ', text)
    text = re.sub(r'\b\d{10,}\b', ' PHONE_NUMBER ', text)
    
    # Replace money amounts but keep the pattern
    text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', ' MONEY_AMOUNT ', text)
    text = re.sub(r'£\d+(?:,\d{3})*(?:\.\d{2})?', ' MONEY_AMOUNT ', text)
    
    # Replace excessive punctuation but keep some
    text = re.sub(r'!{2,}', ' EXCLAMATION ', text)
    text = re.sub(r'\?{2,}', ' QUESTION ', text)
    
    # Keep important spam indicator words before removing punctuation
    spam_indicators = ['free', 'win', 'winner', 'congratulations', 'urgent', 'click', 'call now', 
                      'limited time', 'act now', 'guarantee', 'risk free', 'no cost', 'prize']
    
    # Remove most punctuation but keep apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_spam_data():
    """
    Load the spam.csv dataset
    """
    try:
        # Try different possible locations for the spam.csv file
        possible_paths = [
            "spam.csv",  # Current directory
            "../archive (1)/spam.csv",  # Archive folder
            "f:/email spam detector/archive (1)/spam.csv"  # Full path
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError("spam.csv not found in any expected location")
        
        print(f"Loading dataset from: {csv_path}")
        
        # Load the CSV file with proper column names
        df = pd.read_csv(csv_path, encoding='latin1')
        
        # Check if the file has headers or not
        if 'Category' in df.columns and 'Message' in df.columns:
            # File has headers
            df = df.rename(columns={'Category': 'label', 'Message': 'message'})
        elif df.shape[1] >= 2:
            # File doesn't have headers, use first two columns
            df.columns = ['label', 'message'] + [f'col_{i}' for i in range(2, len(df.columns))]
        else:
            raise ValueError("CSV file doesn't have the expected structure")
        
        # Clean and process labels
        print("Original label counts:")
        print(df['label'].value_counts())
        
        # Convert labels to binary (1 for spam, 0 for ham)
        # Handle case variations
        df['label'] = df['label'].str.lower().str.strip()
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Remove any rows with NaN labels (unmapped values)
        original_size = len(df)
        df = df.dropna(subset=['label'])
        dropped_rows = original_size - len(df)
        
        if dropped_rows > 0:
            print(f"Warning: Dropped {dropped_rows} rows with unmapped labels")
        
        # Also remove any rows with missing messages
        df = df.dropna(subset=['message'])
        
        print(f"\nDataset loaded successfully!")
        print(f"Total messages: {len(df)}")
        print(f"Spam messages: {int(df['label'].sum())}")
        print(f"Ham messages: {int(len(df) - df['label'].sum())}")
        
        return df['message'].tolist(), df['label'].tolist()
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure spam.csv is in one of these locations:")
        for path in possible_paths:
            print(f"- {path}")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

def train_spam_detector():
    """
    Train the spam detection model and save the required files
    """
    print("Loading and preprocessing data...")
    
    # Load your actual spam dataset
    emails, labels = load_spam_data()
    
    if emails is None or labels is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Preprocess the email texts
    processed_emails = [preprocess_text(email) for email in emails]
    
    print(f"Dataset size: {len(emails)} emails")
    print(f"Spam emails: {sum(labels)}")
    print(f"Ham emails: {len(labels) - sum(labels)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_emails, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("\nTraining Count Vectorizer with enhanced features...")
    # Create and train Count vectorizer with better parameters for spam detection
    vectorizer = CountVectorizer(
        max_features=10000,  # Increased features for better detection
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        binary=False,  # Use term frequency instead of just binary
        lowercase=True,
        token_pattern=r'\b\w+\b'  # Better token pattern
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model with optimized parameters...")
    # Train the logistic regression model with better parameters
    model = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        C=1.0,  # Regularization parameter
        class_weight='balanced',  # Handle imbalanced dataset
        solver='liblinear'  # Good for small datasets
    )
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save the trained model
    print("\nSaving trained model...")
    joblib.dump(model, 'logistic_model.pkl')
    
    print("Saving Count vectorizer...")
    joblib.dump(vectorizer, 'count_vectorizer.pkl')
    
    print("\nModel training completed successfully!")
    print("Files created:")
    print("- logistic_model.pkl")
    print("- count_vectorizer.pkl")
    
    # Test the saved model
    print("\nTesting saved model...")
    test_model()

def test_model():
    """
    Test the saved model with sample emails
    """
    try:
        # Load the saved model and vectorizer
        model = joblib.load('logistic_model.pkl')
        vectorizer = joblib.load('count_vectorizer.pkl')
        
        # Test emails - Enhanced spam examples based on user feedback
        test_emails = [
            # SPAM Examples - Dating/Romance
            "Lonely? Chat with beautiful women in your area! They're waiting for you right now!",
            "Dating site: Find your perfect match tonight! Local women want to meet you!",
            "Hello Beautiful, I saw your profile and I think you're amazing. I'm a US soldier stationed overseas. Can we chat?",
            "Hi Gorgeous! I'm a wealthy businessman looking for someone special. I'd love to spoil you. Message me back!",
            "Your secret admirer is waiting to meet you! Find out who has been watching you. Click here now!",
            
            # SPAM Examples - Financial/Account
            "ALERT: Unusual activity detected on your PayPal account. Click here to verify: paypal-security-check.net",
            "Your Netflix subscription will be cancelled unless you update your payment info. Click here now.",
            "Your iCloud storage is full! Upgrade now or lose your photos! Click to purchase more storage.",
            
            # SPAM Examples - Pharmacy/Medical
            "Cheap prescription drugs without prescription! Viagra, Cialis, Xanax available now! 90% discount!",
            "Lose 30 pounds in 30 days! Revolutionary new weight loss pill approved by doctors! Order now!",
            "STOP paying high prices for medications! Canadian pharmacy with 80% savings! No prescription needed!",
            "Male enhancement pills that really work! Increase size naturally! Discreet shipping worldwide!",
            
            # SPAM Examples - Tech Support
            "WARNING: Your computer is infected with 5 viruses! Download our security software immediately!",
            "Microsoft Security Alert: Your Windows license has expired! Renew now to avoid system shutdown!",
            "URGENT: Your iPhone has been hacked! Install our security app now to protect your data!",
            "Your computer is running slow! Download PC Cleaner Pro to speed up your system by 300%!",
            
            # SPAM Examples - Investment
            "Stock tip alert! XYZ company about to explode! Buy now before it's too late! 500% returns guaranteed!",
            
            # SPAM Examples - Inheritance/Nigerian Prince
            "Greetings! I am Mrs. Sarah Johnson, widow of late oil executive. I need trustworthy person to help claim inheritance.",
            "Good day! I am attorney handling estate of wealthy client who died without heirs. You have been selected to inherit.",
            
            # SPAM Examples - Travel/Vacation
            "Last minute cruise deal! 90% off luxury Caribbean cruise! Book now before it's too late!",
            
            # SPAM Examples - Charity/Religious
            "Pastor John needs your prayers and support! Send donation to continue his mission work!",
            "Hurricane victims need your help! Donate now to our emergency relief fund!",
            
            # SPAM Examples - Debt Relief
            "Debt forgiveness program! Reduce your debt by 80%! Call now before program ends!",
            
            # SPAM Examples - Prize/Money
            "Congratulations! You've won a free iPhone! Click here now!",
            "FREE MONEY! Claim your $1000 cash prize now! Text WIN to 12345. Limited time offer!",
            "WINNER! You've won £5000 in our lottery! Send your bank details to claim your prize.",
            "Your mobile number has won a £2000 prize! Call 09061234567 to claim. Standard rates apply.",
            
            # HAM Examples (legitimate messages)
            "The meeting has been rescheduled to 3 PM tomorrow.",
            "Hi, how was your day? Let me know if you want to grab dinner tonight.",
            "Your Amazon order has been shipped and will arrive by Thursday.",
            "Reminder: Your dentist appointment is scheduled for tomorrow at 2 PM.",
            "Thanks for the great presentation today. The client was very impressed.",
            "Can you please send me the quarterly report when you have a chance?",
            "Happy birthday! Hope you have a wonderful day celebrating.",
            "The weather forecast shows rain tomorrow. Don't forget your umbrella!"
        ]
        
        print("\nTesting with sample emails:")
        print("=" * 60)
        
        spam_count = 0
        ham_count = 0
        
        for i, email in enumerate(test_emails, 1):
            processed = preprocess_text(email)
            features = vectorizer.transform([processed])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0].max()
            
            result = "SPAM" if prediction == 1 else "HAM"
            confidence_pct = probability * 100
            
            # Count predictions
            if prediction == 1:
                spam_count += 1
            else:
                ham_count += 1
            
            print(f"\n{i:2d}. Email: {email[:70]}...")
            print(f"    Prediction: {result} (Confidence: {confidence_pct:.1f}%)")
        
        print(f"\n" + "=" * 60)
        print(f"Test Results Summary:")
        print(f"Total emails tested: {len(test_emails)}")
        print(f"Predicted as SPAM: {spam_count}")
        print(f"Predicted as HAM: {ham_count}")
        print("=" * 60)
    
    except FileNotFoundError:
        print("Model files not found. Please run training first.")

if __name__ == "__main__":
    print("Email Spam Detection Model Training")
    print("=" * 40)
    
    # Note: For a real implementation, you would load a proper dataset
    # like the SMS Spam Collection dataset or Enron email dataset
    print("\nThis script now uses your actual spam.csv dataset for training.")
    print("The model will be tested with a variety of spam and legitimate email examples.")
    
    train_spam_detector()

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please train the model first.")
    model = None
    vectorizer = None

def preprocess_text(text):
    """
    Enhanced preprocess the input text for spam detection
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
    
    # Remove most punctuation but keep apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def rule_based_spam_detection(text):
    """
    Enhanced rule-based spam detection to catch obvious spam patterns
    Returns (is_spam_boolean, confidence_score, detected_patterns)
    """
    text_lower = text.lower()
    detected_patterns = []
    spam_score = 0
    
    # Prize/Money patterns
    money_patterns = [
        r'\$\d+(?:,\d{3})*',  # Dollar amounts
        r'£\d+(?:,\d{3})*',   # Pound amounts
        r'\b(?:win|won|winner)\b.*\b(?:money|cash|prize|lottery|gift card)\b',
        r'\b(?:congratulations|congrats)\b.*\b(?:won|win|winner)\b',
        r'\bfree\s+(?:money|cash|iphone|gift)\b',
        r'\b(?:claim|collect)\s+(?:your|now)\b',
        r'\blucky\s+(?:winner|number)\b',
        r'\b(?:selected|chosen)\b.*\b(?:winner|prize|receive)\b'
    ]
    
    # Dating/Romance scam patterns
    dating_patterns = [
        r'\bhot\s+(?:singles|women|girls)\b',
        r'\bmeet\s+(?:local|sexy|beautiful)\b',
        r'\badult\s+(?:content|webcam|chat)\b',
        r'\blonely\b.*\b(?:chat|women|tonight)\b',
        r'\bdating\s+site\b',
        r'\bperfect\s+match\b',
        r'\blocal\s+(?:women|singles)\b.*\bmeet\b',
        r'\bsecret\s+admirer\b',
        r'\bhello\s+(?:beautiful|gorgeous)\b',
        r'\bwealth\w*\s+businessman\b',
        r'\bsoldier\s+stationed\b',
        r'\bprofile.*amazing\b',
        r'\bspoil\s+you\b'
    ]
    
    # Pharmacy/Medical spam patterns  
    pharmacy_patterns = [
        r'\bcheap\s+(?:prescription|medications?)\b',
        r'\bwithout\s+prescription\b',
        r'\bviagra|cialis|xanax\b',
        r'\b\d+%\s+discount\b',
        r'\blose\s+\d+\s+pounds\b',
        r'\bweight\s+loss\s+pill\b',
        r'\bmale\s+enhancement\b',
        r'\bcanadian\s+pharmacy\b',
        r'\bno\s+prescription\s+needed\b',
        r'\bdiscreet\s+shipping\b'
    ]
    
    # Tech support/Security scam patterns
    tech_patterns = [
        r'\bcomputer\s+(?:infected|slow)\b',
        r'\bviruses?\b.*\bdownload\b',
        r'\bmicrosoft\s+security\s+alert\b',
        r'\bwindows\s+license\s+expired\b',
        r'\biphone\s+(?:hacked|infected)\b',
        r'\bsecurity\s+(?:app|software)\b',
        r'\bpc\s+cleaner\b',
        r'\bspeed\s+up.*system\b',
        r'\bwarning.*infected\b'
    ]
    
    # Investment/Financial scam patterns
    investment_patterns = [
        r'\bstock\s+tip\b',
        r'\bcompany.*explode\b',
        r'\b\d+%\s+returns?\s+guaranteed\b',
        r'\bbuy\s+now.*too\s+late\b',
        r'\binvestment\s+opportunity\b',
        r'\bguaranteed.*returns?\b'
    ]
    
    # Subscription/Service scam patterns
    subscription_patterns = [
        r'\bnetflix.*cancelled\b',
        r'\bupdate.*payment\s+info\b',
        r'\bicloud\s+storage.*full\b',
        r'\bupgrade\s+now\b',
        r'\bsubscription.*expired?\b',
        r'\brenew\s+now\b'
    ]
    
    # Nigerian prince/Inheritance scam patterns
    inheritance_patterns = [
        r'\bnigerian\s+prince\b',
        r'\binheritance\b.*\bclaim\b',
        r'\bwidow\s+of.*executive\b',
        r'\battorney\s+handling\b',
        r'\bdeceased.*without\s+heirs\b',
        r'\bselected\s+to\s+inherit\b',
        r'\btransfer.*million\b',
        r'\bbusiness\s+proposal\b'
    ]
    
    # Travel/Vacation scam patterns
    travel_patterns = [
        r'\bcruise\s+deal\b',
        r'\b\d+%\s+off.*cruise\b',
        r'\bluxury.*cruise\b',
        r'\bbook\s+now.*too\s+late\b',
        r'\bfree.*vacation\b',
        r'\btrip.*lifetime\b'
    ]
    
    # Charity/Religious scam patterns
    charity_patterns = [
        r'\bpastor.*donation\b',
        r'\bprayers\s+and\s+support\b',
        r'\bhurricane\s+victims\b',
        r'\bdonation.*fund\b',
        r'\borphans.*africa\b',
        r'\bemergency\s+relief\b'
    ]
    
    # Debt/Financial relief patterns
    debt_patterns = [
        r'\bdebt\s+forgiveness\b',
        r'\breduce.*debt\b',
        r'\btax\s+(?:debt|relief)\b',
        r'\bstudent\s+loan\s+forgiveness\b',
        r'\bfinal\s+notice.*debt\b',
        r'\bwage\s+garnishment\b'
    ]
    
    # Urgency patterns
    urgency_patterns = [
        r'\burgent\b.*\b(?:act|call|click|respond)\b',
        r'\blimited\s+time\b',
        r'\bact\s+now\b',
        r'\bcall\s+now\b',
        r'\bexpires?\s+(?:today|soon|in)\b',
        r'\bfinal\s+(?:notice|warning|chance)\b',
        r'\bimmediately\b',
        r'\bbefore.*too\s+late\b'
    ]
    
    # Contact patterns (suspicious)
    contact_patterns = [
        r'\bcall\s+\d{1}-?\d{3}-?\d{3}-?\d{4}\b',
        r'\btext\s+\w+\s+to\s+\d+\b',
        r'\bclick\s+here\b',
        r'\bvisit\s+www\.',
        r'\bgo\s+to\s+\w+\.com\b',
        r'\bmessage\s+(?:me\s+)?back\b',
        r'\breply\s+(?:now|yes|no)\b'
    ]
    
    # Financial scam patterns
    financial_patterns = [
        r'\baccount\s+(?:suspended|frozen|locked|cancelled)\b',
        r'\bverify\s+(?:your|account|identity)\b',
        r'\bupdate\s+(?:payment|billing)\b',
        r'\bno\s+(?:credit\s+check|fees|cost)\b',
        r'\bguaranteed\s+(?:approval|loan)\b',
        r'\bunusual\s+activity\b',
        r'\bpaypal.*verify\b'
    ]
    
    
    # Check all patterns
    pattern_groups = [
        (money_patterns, 'Prize/Money scam', 3),
        (dating_patterns, 'Dating/Romance scam', 4),
        (pharmacy_patterns, 'Pharmacy/Medical spam', 4),
        (tech_patterns, 'Tech support scam', 4),
        (investment_patterns, 'Investment scam', 3),
        (subscription_patterns, 'Subscription scam', 3),
        (inheritance_patterns, 'Inheritance scam', 4),
        (travel_patterns, 'Travel/Vacation scam', 3),
        (charity_patterns, 'Charity/Religious scam', 3),
        (debt_patterns, 'Debt relief scam', 3),
        (urgency_patterns, 'Urgency tactics', 2),
        (contact_patterns, 'Suspicious contact', 2),
        (financial_patterns, 'Financial scam', 3)
    ]
    
    for patterns, pattern_name, weight in pattern_groups:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_patterns.append(pattern_name)
                spam_score += weight
                break  # Only count each pattern type once
    
    # Additional scoring based on text characteristics
    if len(re.findall(r'!', text)) >= 3:  # Multiple exclamation marks
        spam_score += 1
        detected_patterns.append('Excessive punctuation')
    
    if len(re.findall(r'[A-Z]{3,}', text)) >= 2:  # Multiple ALL CAPS words
        spam_score += 1
        detected_patterns.append('Excessive capitalization')
    
    # Check for suspicious combinations
    if any(word in text_lower for word in ['free', 'win', 'winner', 'prize']) and \
       any(word in text_lower for word in ['click', 'call', 'text', 'visit']):
        spam_score += 2
        detected_patterns.append('Prize + Action combination')
    
    # Determine if it's spam based on score
    is_spam = spam_score >= 2  # Lowered threshold to catch more spam
    confidence = min(spam_score / 8.0, 1.0)  # Convert to 0-1 scale
    
    return is_spam, confidence, detected_patterns

@app.route('/')
def index():
    """
    Render the main page with default values
    """
    return render_template('index.html', 
                         prediction=None, 
                         spam_score=None, 
                         triggered_rules=None, 
                         email_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if the email is spam or not
    """
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure model files are present.'
        }), 500
    
    try:
        # Get the email text from the request
        email_text = request.json.get('email_text', '')
        
        if not email_text.strip():
            return jsonify({
                'error': 'Please provide email text for prediction.'
            }), 400
        
        # Preprocess the text
        processed_text = preprocess_text(email_text)
        
        # Transform the text using the loaded vectorizer
        text_features = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        prediction_proba = model.predict_proba(text_features)[0]
        
        # Determine the result
        is_spam = bool(prediction)
        confidence = float(max(prediction_proba))
        
        # Rule-based detection for obvious spam
        rule_based_is_spam, rule_based_confidence, detected_patterns = rule_based_spam_detection(email_text)
        
        # Combine results
        is_spam = is_spam or rule_based_is_spam
        confidence = max(confidence, rule_based_confidence)
        
        return jsonify({
            'is_spam': is_spam,
            'confidence': confidence,
            'prediction': 'Spam' if is_spam else 'Not Spam',
            'detected_patterns': detected_patterns
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    import os
    
    # Enable debug mode for development
    app.debug = True
    
    # Check if model files exist
    if not os.path.exists('logistic_model.pkl') or not os.path.exists('count_vectorizer.pkl'):
        print("Warning: Model files are missing! Please run train_model.py first.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

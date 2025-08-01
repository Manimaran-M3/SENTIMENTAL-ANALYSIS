**Sentiment Analysis Project**

This project analyzes the subject lines of customer complaint tickets raised in a banking system to predict both sentiment (positive or negative) and emotions (like anger, sadness, joy, etc.). The goal is to understand customer feedback better and prioritize responses based on emotional tone and urgency.

**Project Overview**

Banks receive hundreds of complaint tickets every day. While most are text-heavy, the subject line alone can reveal important clues about customer mood and urgency. In this project, we perform sentiment analysis and emotion classification using transformer-based NLP models from Hugging Face.

**Main Goals:**

1. Clean and preprocess ticket subjects

2. Predict sentiment using a pretrained model

3. Detect emotions using an emotion-aware model

4. Compare predictions against true labeled sentiments

5. Visualize trends and evaluate performance

**Technologies Used :**

->Python 3

->Hugging Face transformers

->pandas, matplotlib, seaborn

->scikit-learn for evaluation metrics

**Pretrained Models:**

Sentiment: distilbert-base-uncased-finetuned-sst-2-english

Emotion: bhadresh-savani/distilbert-base-uncased-emotion

**Data Preprocessing**

1. Subject lines are cleaned using a custom function:

2. Lowercased text

3. Removed ticket prefixes like RE: or FW:

4. Removed punctuation, numbers, and common banking acronyms (VPN, EOD, SQL, etc.)

5. Trimmed extra spaces

**Model Usage**

We use two Hugging Face pipelines:

Sentiment Analysis
->Predicts whether a subject line is positive or negative

Emotion Detection
->Identifies emotional tone such as anger, joy, fear, sadness, etc.

python:

from transformers import pipeline

sentiment_analyser = pipeline("sentiment-analysis")

emotional_analyser = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

**Evaluation Metrics:**

We compared predicted sentiments against the true labels using:

Accuracy, Precision, Recall, F1 Score, Confusion Matrix

**output:**

Accuracy:  0.77

Precision: 0.67

Recall:    0.67

F1 Score:  0.67

**Visualizations**

->Distribution of sentiment labels

->Distribution of emotions detected

->Confusion matrix showing model performance on true vs predicted sentiment

**Sample Predictions**

Complaint Subject	Sentiment	Emotion
"Unable to withdraw cash at ATM"	Negative	Anger
"Service issue resolved quickly"	Positive	Joy
"Transaction failed but money debited"	Negative	Disappointment
"Thanks for timely response"	Positive	Gratitude

**Future Scope**

Apply to full complaint descriptions, not just subject lines
Build a dashboard to monitor customer sentiment trends
Integrate with a ticket management system for real-time analysis
Include "neutral" sentiment category for better granularity

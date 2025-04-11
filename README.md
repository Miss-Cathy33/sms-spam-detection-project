# Spam Message Classification using Machine Learning

## Overview

This project demonstrates how to classify spam and ham ie not spam (legitimate) messages using Natural Language Processing (NLP) and Machine Learning. By analyzing patterns in text messages, we build models that automatically detect fraudulent emails and SMS messages.

## Features

- **Data Cleaning & Preprocessing**: Tokenization, stopword removal, lemmatization, and feature engineering (e.g., using LabelEncoder).
- **Feature Extraction**: TF-IDF vectorization for word importance.
- **Machine Learning Models**: Logistic Regression, Naïve Bayes, and Random Forest Ensemble Voting Classifier.
- **Performance Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix.
- **Data Visualization**: Word clouds and spam frequency analysis.

## Installation

1. Clone the repository:

   ```bash
   git clone C:\Users\Hp\Desktop\Github Repos\spam classification repo\data_visualization__
   cd spam-classifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Datavisualization.ipynb
   ```

## Dataset

The dataset consists of SMS messages labeled as "spam" or "ham." It is preprocessed to enhance model accuracy.

## Usage

- Explore the dataset and visualize spam trends.
- Train different ML models to classify messages.
- Evaluate the models and improve classification performance.

## Results

The Naïve Bayes classifier performed best in terms of precision, efficiently detecting spam with minimal false positives.

## Dataset Insights and Suggestions for Improvement

### Dataset Insights

#### 1. Class Distribution

- The dataset contains two classes: **spam** and **ham**.
- It's essential to check for class imbalance, as it can affect model performance.
- Techniques like **SMOTE (Synthetic Minority Oversampling Technique)** have been applied in the notebook to address this issue.

#### 2. Message Length

- **Spam messages** tend to have longer lengths compared to **ham messages**.
- This is evident from the **boxplot** and **histogram** visualizations in the notebook.
- The feature `Message_Length` serves as a strong indicator for classification.

#### 3. Specific Words

- Words like **"free," "win," "offer," "urgent,"** and **"cash"** are more frequent in spam messages.
- The feature `Specific_Words_Count` effectively captures this pattern.

#### 4. Punctuation and Special Characters

- Spam messages often contain more **punctuation** and **special characters** to grab attention.
- The features `Punctuation_Count` and `Special_Char_Count` help capture this behavior.

#### 5. Sentiment Polarity

- Sentiment analysis using **TextBlob** reveals:
  - Spam messages often have a **neutral** or slightly **positive sentiment**.
  - Ham messages may vary more in sentiment.

#### 6. Capitalized Words

- Spam messages frequently use **capitalized words** to emphasize offers or urgency.
- The feature `Capitalized_Words` captures this trend effectively.

---

### Suggestions for Improvement

#### 1. Feature Engineering

- Introduce additional features such as:
  - **Ratio of digits to total characters**.
  - **Presence of URLs** in messages.
  - **Frequency of specific spam-related keywords**.

#### 2. Model Evaluation

- While the **Naïve Bayes classifier** performed well, consider exploring advanced models like:
  - **Deep Learning models** such as **LSTMs** or **transformers (e.g., BERT)** for improved accuracy.

#### 3. Visualization

- Use **word clouds** for spam and ham messages to provide a quick visual summary of the most frequent words in each class.

#### 4. Explainability

- Leverage tools like **SHAP** or **LIME** to:
  - Explain model predictions.
  - Understand feature importance and how features contribute to classification decisions.

---

By leveraging these insights and suggestions, the project can be further refined to achieve better classification results and provide a deeper understanding of the dataset.

## Connect with Me

 **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/stephen-njoroge-83b56b277/recent-activity/all/)  
 **GitHub**: [Your GitHub Profile](https://github.com/)  

## License

This project is licensed under the MIT License.

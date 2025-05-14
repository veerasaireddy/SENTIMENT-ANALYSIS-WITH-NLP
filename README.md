# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : BUSIREDDY VEERA SAI REDDY

*INTERN ID* : CT04DM982

*DOMAIN* : MACHINE LEARNING

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTHOSH KUMAR



📄 Task 2: Sentiment Analysis Using TF-IDF and Logistic Regression
As part of the CodTech Machine Learning Internship, Task 2 focuses on building a sentiment analysis model using Natural Language Processing (NLP) techniques. The primary objective of this task is to analyze a dataset of customer reviews and determine whether the sentiment expressed in each review is positive or negative. To accomplish this, we used TF-IDF vectorization to convert textual data into numerical features and employed a Logistic Regression model for binary classification.

📦 Dataset and Problem Overview:
We began by creating a small synthetic dataset of customer reviews, each labeled as either positive (1) or negative (0). Each review consisted of a short sentence simulating real-world customer feedback such as "I love this product!" or "Worst purchase ever." These samples serve as a simplified training ground for implementing a sentiment analysis pipeline. While a small dataset was used for demonstration, the approach is fully scalable to larger, real-world datasets such as those from Amazon, Yelp, or IMDb.

🧹 Data Cleaning and Preprocessing:
Before training the model, the text data needed to be cleaned and preprocessed. This included converting all text to lowercase to eliminate case sensitivity issues, removing punctuation and special characters using regular expressions, and reducing extra whitespaces. This preprocessing step is crucial because machine learning models cannot interpret raw text — they require consistent and clean numerical input.

🧠 Feature Extraction Using TF-IDF:
Once the text was cleaned, the next step was to convert it into a format that a machine learning algorithm could work with. For this, TF-IDF (Term Frequency-Inverse Document Frequency) was used. TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection. It penalizes common words (like "the", "and", "is") and gives more weight to unique or rare terms that might carry more semantic meaning in the context of sentiment. The TfidfVectorizer from Scikit-learn was used to transform the cleaned text data into a TF-IDF matrix.

🔀 Splitting the Dataset:
The dataset was then split into training and testing sets using Scikit-learn’s train_test_split() function. Initially, an issue arose where the split resulted in a testing set that did not include samples from both sentiment classes. This led to warnings about undefined precision and recall during evaluation. To solve this, the stratify=y parameter was added to ensure both classes were equally represented in both sets, improving the evaluation process.

🤖 Model Training with Logistic Regression:
A Logistic Regression model was then trained using the TF-IDF-transformed training data. Logistic Regression is a simple yet effective algorithm for binary classification tasks like sentiment analysis. It models the probability that a given input belongs to a particular class using a logistic function. After training, the model was used to predict sentiments on the test set.

📈 Model Evaluation:
The model's performance was evaluated using accuracy score, confusion matrix, and classification report (which includes precision, recall, and F1-score). These metrics provided insight into how well the model could generalize to unseen data. In cases where the dataset was small or imbalanced, the metrics indicated poor performance, highlighting the importance of using sufficiently large and balanced datasets in real applications.

🧾 Conclusion:
Task 2 provided hands-on experience with one of the most common NLP applications in real-world data science — sentiment analysis. Through this task, we explored end-to-end text processing, including cleaning, feature extraction, model training, and evaluation. The use of TF-IDF helped in capturing meaningful textual patterns, while Logistic Regression served as an efficient and interpretable classifier. Although a small dataset was used for demonstration, the same workflow can be extended to large-scale sentiment analysis projects in industry.

The final deliverable is a well-structured Jupyter Notebook named task2_sentiment_analysis.ipynb, which includes:

Data cleaning and preprocessing steps

TF-IDF feature transformation

Model training with logistic regression

Model evaluation using multiple metrics

Commentary and analysis of results

This task builds foundational skills in NLP, machine learning, and model evaluation — all of which are essential for aspiring data scientists.

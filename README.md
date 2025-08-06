# Fake-News-Detection-Using-Machine-Learning
This project investigates how machine learning techniques can be used to detect fake news articles. It uses real and fake news datasets and walks through a full CRISP-DM-inspired data science process — including loading, cleaning, exploration, modeling, and evaluation, all within modular Jupyter/Colab notebooks.

## Project Overview

This repository contains four notebooks, each representing a stage in the pipeline:

1. **Data_Loader.ipynb**  
   Loads and combines real and fake news datasets, assigns unique identifiers, and stores the unified dataset in a SQLite database.

2. **Data_Understanding.ipynb**  
   Conducts exploratory data analysis (EDA) using frequency analysis and word clouds to identify common tokens and patterns in fake vs. real articles.

3. **Data_Preprocessing&Validation.ipynb**  
   Cleans the data, merges article title and body into a `content` column, and performs a stratified split into training and testing sets.

4. **Modeling_Evaluation.ipynb**  
   Trains and evaluates multiple classifiers (Logistic Regression, Naive Bayes, Random Forest, and Linear SVM) using TF-IDF features and outputs comparative performance metrics.

---

## Dataset Description

The dataset consists of two CSV files:

- `Fake.csv`: News articles classified as fake.
- `True.csv`: News articles classified as real.

Each file includes:
- `title`: Headline of the news article
- `text`: Body of the article
- `subject`: Category of the article
- `date`: Publication date

These are merged and labeled (1 for fake, 0 for real) during ingestion.

---

## Technologies & Libraries Used

- **Languages**: Python 3
- **Libraries**:
  - `pandas`, `numpy`, `os`, `sqlalchemy`
  - `sklearn` for machine learning and evaluation
  - `nltk` for natural language processing
  - `wordcloud`, `matplotlib`, `seaborn` for visualization
  - `sqlite3` for lightweight data storage

---

## Machine Learning Models Evaluated

Each model is trained using TF-IDF vectorized text features:
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Random Forest Classifier**
- **Linear Support Vector Machine (SVM)**

Evaluation metrics include:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Visual comparisons across models

---

## Database Structure

The project uses a single SQLite database (`fakenews.db`) containing:

| Table Name      | Description                                |
|-----------------|--------------------------------------------|
| `news_article`  | Combined dataset from both CSV files       |
| `train_articles`| Preprocessed training set (80%)            |
| `test_articles` | Preprocessed testing set (20%)             |

---

## How to Run the Project

1. Clone this repository and upload the CSV files (`Fake.csv`, `True.csv`) to the appropriate path.
2. Open the notebooks in order:
   - `Data_Loader.ipynb`
   - `Data_Understanding.ipynb`
   - `Data_Preprocessing&Validation.ipynb`
   - `Modeling_Evaluation.ipynb`
3. Run all cells sequentially in each notebook.
4. Review metrics and plots to analyze model performance.

---

## Sample Visuals

- Word Clouds for Fake vs. Real News
- Token Frequency Plots
- Confusion Matrices per Model
- Accuracy/F1 Comparison Chart

---

## Authors
#### Alex McAnnally, Kyle Govender, Craig Runnels

###### This project was developed as part of the Advanced Business Analytics coursework at the University of Alabama at Birmingham.
---

## Future Plans

As this project continues to evolve beyond the classroom, the following enhancements and features are planned for future development:
Dockerization

    Package the entire project in a Docker container to ensure consistent environment setup, simplify deployment, and streamline collaboration.

    Create a docker-compose configuration to separate model serving logic from the front-end UI, enabling modular architecture and easier maintenance.

### Front-End Interface

    Develop a user-friendly graphical interface using Flask, Streamlit, or a similar framework to allow users to input articles or headlines and receive real-time predictions.

    Include visualization features to show confidence scores, model explanations (e.g., feature importance), and prediction history.

### Model Improvements

    Experiment with more advanced NLP techniques, such as transformer-based models (e.g., BERT, RoBERTa) to improve classification accuracy and robustness.

    Fine-tune hyperparameters and expand the training dataset to better generalize across diverse sources and writing styles.

### API Development

    Expose a RESTful or GraphQL API to allow external applications to programmatically interact with the fake news detection model.

    Implement rate-limiting, logging, and authentication for security and monitoring.

### Logging and Feedback

    Add backend logging of prediction requests (with user consent) for performance analysis and retraining.

    Introduce optional user feedback mechanisms to flag incorrect predictions and continuously improve model performance.

### Deployment

    Deploy the app on a cloud platform (e.g., Heroku, AWS, or Render) for public access and demonstration purposes.

    Enable HTTPS support and basic user session management if needed for persistent usage or feedback tracking.

---

## License
   MIT License
   
   Copyright (c) 2025 Alex McAnnally
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

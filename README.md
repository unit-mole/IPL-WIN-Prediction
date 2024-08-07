# IPL-WIN-Prediction

<!DOCTYPE html>
<html>
<head>
    <title>IPL Win Prediction</title>
</head>
<body>

<h1>IPL Win Prediction</h1>

<p>This repository contains a project aimed at predicting the winner of the IPL 2025 using various machine learning models. The project includes data preprocessing, model training, evaluation, and visualization of results.</p>

<h2>Files in the Repository</h2>
<ul>
    <li><code>matches.csv</code>: Contains historical IPL match data.</li>
    <li><code>deliveries.csv</code>: Contains ball-by-ball delivery data for IPL matches.</li>
    <li><code>IPL_Win_Prediction.ipynb</code>: Jupyter Notebook with the entire workflow, including data preprocessing, model building, evaluation, and visualizations.</li>
</ul>

<h2>Project Workflow</h2>
<ol>
    <li><strong>Data Preprocessing</strong>: The datasets are loaded and cleaned. Missing values are filled, and irrelevant columns are dropped.</li>
    <li><strong>Feature Engineering</strong>: Various features are engineered from the datasets to be used for model training.</li>
    <li><strong>Model Building</strong>: Multiple machine learning models are built and tuned using techniques such as Grid Search and Randomized Search. The models include:
        <ul>
            <li>Random Forest Classifier</li>
            <li>Gradient Boosting Classifier</li>
            <li>XGBoost Classifier</li>
            <li>Neural Network Classifier</li>
            <li>Support Vector Machine Classifier</li>
            <li>K-Nearest Neighbors Classifier</li>
            <li>Naive Bayes Classifier</li>
        </ul>
    </li>
    <li><strong>Model Evaluation</strong>: The models are evaluated based on accuracy, precision, recall, and F1 score. The best model is selected based on these metrics.</li>
    <li><strong>Visualization</strong>: Various visualizations are created to understand the data better and present the model results. These include:
        <ul>
            <li>Number of matches played by each team</li>
            <li>Win percentage of each team</li>
            <li>Toss decisions across seasons</li>
            <li>And more...</li>
        </ul>
    </li>
    <li><strong>Win Prediction for IPL 2025</strong>: Using the best-performing model, predictions are made for hypothetical matchups to determine the most likely winner of IPL 2025.</li>
</ol>

<h2>Results</h2>
<p>The best model based on the evaluation metrics is the <strong>XGBoost Classifier</strong> with the following parameters:</p>
<ul>
    <li>Learning Rate: 0.1</li>
    <li>Max Depth: 3</li>
    <li>Number of Estimators: 100</li>
</ul>
<p>Based on this model, the predicted winner of IPL 2025 is the <strong>Lucknow Super Giants</strong>.</p>

<h2>Usage</h2>
<p>To run the project on your local machine, follow these steps:</p>
<ol>
    <li>Clone the repository: <code>git clone &lt;repository_url&gt;</code></li>
    <li>Navigate to the repository directory: <code>cd IPL_Win_Prediction</code></li>
    <li>Install the required dependencies: <code>pip install -r requirements.txt</code></li>
    <li>Run the Jupyter Notebook: <code>jupyter notebook IPL_Win_Prediction.ipynb</code></li>
</ol>

<h2>Dependencies</h2>
<p>The project requires the following Python packages:</p>
<ul>
    <li>numpy</li>
    <li>pandas</li>
    <li>scikit-learn</li>
    <li>xgboost</li>
    <li>matplotlib</li>
    <li>seaborn</li>
</ul>

<h2>License</h2>
<p>This project is licensed under the MIT License.</p>

</body>
</html>

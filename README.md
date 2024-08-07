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

<h2>Consolidated Results</h2>
<table>
    <tr>
        <th>Model</th>
        <th>Best Parameters</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1 Score</th>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}</td>
        <td>0.541284</td>
        <td>0.508799</td>
        <td>0.541284</td>
        <td>0.504944</td>
    </tr>
    <tr>
        <td>Gradient Boosting</td>
        <td>{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}</td>
        <td>0.568807</td>
        <td>0.567114</td>
        <td>0.568807</td>
        <td>0.561579</td>
    </tr>
    <tr>
        <td>XGBoost</td>
        <td>{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}</td>
        <td>0.577982</td>
        <td>0.580677</td>
        <td>0.577982</td>
        <td>0.570402</td>
    </tr>
    <tr>
        <td>Neural Network</td>
        <td>{'hidden_layer_sizes': (100,), 'learning_rate_init': 0.01, 'max_iter': 200}</td>
        <td>0.339450</td>
        <td>0.297095</td>
        <td>0.339450</td>
        <td>0.296075</td>
    </tr>
    <tr>
        <td>Support Vector Machine</td>
        <td>{'C': 1.6601864044243653, 'gamma': 0.16599452033620266}</td>
        <td>0.394495</td>
        <td>0.415541</td>
        <td>0.394495</td>
        <td>0.390608</td>
    </tr>
    <tr>
        <td>K-Nearest Neighbors</td>
        <td>{'n_neighbors': 7, 'weights': 'distance'}</td>
        <td>0.426606</td>
        <td>0.444625</td>
        <td>0.426675</td>
        <td>0.426675</td>
    </tr>
    <tr>
        <td>Naive Bayes</td>
        <td>N/A</td>
        <td>0.238532</td>
        <td>0.248542</td>
        <td>0.238532</td>
        <td>0.233931</td>
    </tr>
</table>

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

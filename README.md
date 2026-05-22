# Iris Species Classification

Universidad de la Costa — Data Mining
Professor: Jose Escorcia-Gutierrez, Ph.D.

**Team members:** Camilo Andres Visbal Beltran, Julian Paba de la Hoz, Adrian Llamas Quintero, Roberto De los Reyes Castro

## Purpose

This project applies a complete data mining pipeline to the Iris dataset. A Random Forest classifier is trained to predict flower species (Iris-setosa, Iris-versicolor, Iris-virginica) from four measurements: sepal length, sepal width, petal length, and petal width. The workflow covers data loading, exploratory analysis, preprocessing, model training, evaluation, and interactive visualization through a Streamlit dashboard.

## Files

| File | Description |
|---|---|
| Proyect.py | Streamlit dashboard — main application |
| requirements.txt | Python dependencies |
| Iris.csv | Dataset used by the application |

## Running locally

Python 3.10 or higher is required.

Install dependencies:

```
pip install -r requirements.txt
```

Start the dashboard:

```
streamlit run Proyect.py
```

The app loads the dataset from the GitHub repository directly, so an internet connection is needed.

## Dashboard sections

The sidebar provides filters by species, sepal length range, and petal length range. The top of the page shows four model performance cards: accuracy, precision, recall, and F1 score. Below that, four tabs organize the visualizations:

- Distribution: histograms and boxplots for any selected feature, broken down by species.
- Scatter Matrix: pairwise scatter plots across all four features.
- 3D View and Prediction: interactive 3D scatter plot; entering custom measurements and clicking Predict places the new sample on the plot and returns the predicted species with class probabilities.
- Confusion Matrix: heatmap of test-set predictions alongside a feature importance chart.

A collapsible table at the bottom displays the filtered records.

## Deployed dashboard

https://iris-classification-n6phkmyrptdkycheydbt6b.streamlit.app/#model-performance

# CrimeRadar - Crime Hotspot Prediction Dashboard

## Overview
CrimeRadar is a comprehensive crime prediction and analysis dashboard that utilizes machine learning models to predict crime hotspots and trends. The dashboard is built using Python libraries like `Plotly`, `Dash`, `Pandas`, `Scikit-learn`, and `Statsmodels` to showcase crime data analysis, clustering, and predictive insights.

This project aims to provide actionable insights into crime data by using advanced techniques such as Random Forest for classification and ARIMA for time series forecasting. It also includes dynamic visualizations, predictions, and clustering of crime hotspots across various cities.

## Dataset
The dataset used in this project is the **Indian Crime Dataset**, which contains crime-related data collected across various cities in India. The dataset includes details like crime descriptions, location, time of occurrence, and case closure status. 

The dataset can be accessed from Kaggle here:
[Indian Crime Dataset](https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset)

## Features
The dashboard includes the following features:
- **Crime Clusters**: Visualizes the crime distribution across different cities and their respective clusters using KMeans clustering.
- **Crime Trends**: Displays a time series graph of monthly crime occurrences.
- **Crime Predictions**: Provides real-time predictions based on user-selected parameters, such as city, crime type, month, and hour of occurrence.
- **Dynamic Interactions**: Users can interact with dropdowns, sliders, and charts to explore various crime statistics and predictions.
- **Crime Distribution Pie Chart**: A dynamic pie chart that updates based on the city and crime type selected by the user. This chart displays the distribution of crimes by different cities, showing how the selected crime is distributed across the cities. It provides a visual understanding of crime prevalence in each city.

## Key Models and Techniques
1. **Random Forest Classifier**: Used to predict whether a crime case will be closed based on features like city, time, and crime description.
2. **ARIMA (AutoRegressive Integrated Moving Average)**: A time series forecasting model to predict the crime trends in the coming months.
3. **KMeans Clustering**: Used for identifying crime hotspots in different cities based on the frequency of crime incidents.

## Requirements
The following Python packages are required to run the project:

```bash
pip install pandas numpy plotly scikit-learn statsmodels dash dash-table

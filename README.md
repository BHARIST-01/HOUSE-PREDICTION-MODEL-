# California House Price Prediction

Live app: https://house-price-prediction-in-ml.streamlit.app/

A small Streamlit app that estimates the median house value for a California district using a trained linear regression model. The project includes a Jupyter notebook for exploring the data and training the model.

> **Note:** This is an educational estimate based on district-level data. It is not a guaranteed value for an individual home.

## Features

- Explore the California Housing dataset in the notebook.
- Train and evaluate a linear regression model.
- Enter eight district features in the app to get a predicted median value.
- Display the prediction in US dollars.

## Project files

| File | Description |
| --- | --- |
| `app.py` | Streamlit application for making predictions |
| `house_price_prediction.ipynb` | Data exploration, model training, and evaluation |
| `linear_regression_model.pkl` | Trained model loaded by the app |

## Requirements

Use Python 3.9 or later. Install the packages used by the app and notebook:

```bash
python -m pip install streamlit pandas scikit-learn joblib numpy matplotlib seaborn jupyter
```

The notebook downloads the California Housing dataset through scikit-learn the first time it runs, so an internet connection may be needed.

## Run the app

From the project directory, start Streamlit:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Use the notebook

Start Jupyter from the project directory:

```bash
jupyter notebook house_price_prediction.ipynb
```

Run the cells in order to explore the dataset, split it into training and test sets, fit the linear regression model, save `linear_regression_model.pkl`, and calculate evaluation metrics.

## Prediction inputs

The app expects the following California Housing dataset features, in the same order used to train the model:

`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

The model predicts `MedHouseVal`, which is measured in units of $100,000. The app converts that output to dollars for display. Results depend on the trained model and the inputs provided; they should not be used as a professional appraisal.

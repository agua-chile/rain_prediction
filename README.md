# Australian Rainfall Prediction

A Python project that predicts daily rainfall in the Melbourne area using historical weather data and machine learning models.

## Author

**Agua Chile**
- GitHub: [@agua_chile](https://github.com/aguachile)
- Project: rain_prediction


## Acknowledgements

Jeff Grossman
Abhishek Gagneja
IBM Corporation

## Features

- 🌧️ **Rainfall Prediction**: Predicts daily rainfall using weather data
- 🧠 **Machine Learning Models**: Random Forest and Logistic Regression classifiers
- 🛠️ **Modular Pipeline**: Separate modules for data processing, modeling, evaluation, and visualization
- 📊 **Rich Evaluation**: Accuracy, precision, recall, F1-score, and feature importance
- 📈 **Visualization Suite**: Plots for target distribution, seasonal/location patterns, feature importance, and correlation heatmaps
- ⚙️ **Configurable Architecture**: Easily extendable for new models or features

## Technology Stack

- **ML Frameworks**: scikit-learn
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment**: Python, Jupyter Notebook

## Setup

### Prerequisites

- Python 3.13
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone the repository**
	```bash
	git clone https://github.com/agua-chile/rain_prediction
	cd rain_prediction
	```

2. **Create and activate virtual environment and install dependencies**
	```bash
	cd env
	chmod +x setup.sh && ./setup.sh
	```

3. **Run the notebook**
	- Open `rainfall_prediction.ipynb` in Jupyter or VS Code and run the cells to execute the workflow.

## Usage

1. **Launch Jupyter Notebook or open in VS Code**
	- Navigate to the `rain_prediction/` directory.

2. **Run the notebook**
	- Follow the workflow in `rainfall_prediction.ipynb` to preprocess data, train models, and visualize results.

## Project Structure

```
rain_prediction/
├── README.md                         # Project documentation
├── rainfall_prediction.ipynb         # Main notebook for analysis and modeling
├── env/
│   └── requirements.txt              # Python dependencies
├── rainfall_prediction_py/           # Modular Python package
│   ├── __init__.py
│   ├── data_processing.py            # Data loading, cleaning, and preprocessing
│   ├── evaluation.py                 # Model evaluation metrics and analysis
│   ├── models.py                     # Model training and hyperparameter optimization
│   ├── utils.py                      # Utility functions and error handling
│   └── visualization.py              # Professional plotting and analysis tools
```

## Configuration

Hyperparameters and pipeline settings can be configured in the respective Python modules. Key parameters include:

- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **Logistic Regression**: `max_iter`, regularization settings
- **Preprocessing**: feature selection, encoding methods

## Features in Detail

### Data Processing
- Loads and cleans weather data
- Handles missing values and feature engineering (season, location, etc.)

### Modeling
- Trains Random Forest and Logistic Regression classifiers
- Hyperparameter optimization with GridSearchCV

### Evaluation
- Computes accuracy, precision, recall, F1-score, and AUC-ROC
- Generates classification reports and confusion matrices

### Visualization
- Plots target distribution, seasonal/location patterns, feature importance, and correlation heatmaps

## License

This project is licensed under the MIT License.

## Support

For questions or issues:
- Review error messages and tracebacks printed in the notebook
- Ensure all dependencies from `requirements.txt` are installed
- Verify that the dataset is accessible and correctly loaded

---

*Built using scikit-learn, pandas, and matplotlib*

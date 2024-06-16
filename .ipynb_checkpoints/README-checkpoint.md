# Challenge 13

## README: Alphabet Soup Startup Success Predictor

### Project Overview

Alphabet Soup is a fictitious venture capital firm that receives many funding applications from startups daily. This project aims to create a binary classification model using a deep neural network to predict whether an applicant will become successful if funded by Alphabet Soup.

### Project Structure

The project is divided into the following steps:

1. **Data Preprocessing:**
   - Load and clean the dataset.
   - Encode categorical variables.
   - Split the data into training and testing sets.
   - Scale the features.

2. **Model Development:**
   - Build and compile a neural network model.
   - Train and evaluate the model.
   - Save the trained model.

3. **Model Optimization:**
   - Attempt various optimization techniques to improve model accuracy.
   - Save the optimized models.

### Files Included

- `applicants_data.csv`: The dataset containing information about the startups.
- `AlphabetSoup_Model.ipynb`: Jupyter notebook containing the complete code for data preprocessing, model development, and optimization.
- `AlphabetSoup.h5`: The initial trained model.
- `AlphabetSoup_optimized_1.h5`, `AlphabetSoup_optimized_2.h5`, `AlphabetSoup_optimized_3.h5`: The optimized models.

### Setup Instructions

1. **Install Required Libraries:**
   Ensure you have the following Python libraries installed:
   - Pandas
   - Scikit-learn
   - TensorFlow
   - Keras


2. **Download the Dataset:**
   Make sure `applicants_data.csv` is in the same directory as your Jupyter notebook or Python script.

### How to Run the Code

1. **Open the Jupyter Notebook:**
   Open the `AlphabetSoup_Model.ipynb` notebook using Jupyter Notebook or JupyterLab.

2. **Run the Cells:**
   Execute the cells in the notebook sequentially. The notebook is organized in the following order:
   - Data Preprocessing
   - Model Development
   - Model Optimization

3. **Evaluate the Models:**
   The notebook will display the accuracy and loss of the initial model and the optimized models. Compare these results to determine the best model.

4. **Use the Saved Models:**
   The trained models are saved as HDF5 files (`.h5`). You can load and use these models for predictions as needed.

### Summary of Results

Initial Model:

Loss: 0.5597551465034485
Accuracy: 0.7254332304000854
Optimized Model 1:

Loss: 0.5539121031761169
Accuracy: 0.7295413613319397
Optimized Model 2:

Loss: 0.5523213148117065
Accuracy: 0.7306873798370361
Optimized Model 3:

Loss: 0.5583446025848389
Accuracy: 0.7275789976119995

### Future Work

Further optimizations and feature engineering can be done to improve the model accuracy. Consider experimenting with different neural network architectures, feature selection techniques, and hyperparameter tuning.

# Product Price Prediction

This repository contains a project that focuses on predicting the price of a product based on various features such as name, description, category, and brand name. The project utilizes a GRU-based RNN (Recurrent Neural Network) for price prediction and employs an SVM (Support Vector Machine) for brand name inference.

## Project Overview

The goal of this project is to develop a predictive model that can estimate the price of a product given its relevant attributes. The attributes used for prediction include the product's name, description, category, and brand name. The project involves two main components: a GRU-based RNN for price prediction and an SVM for brand name inference.

## Repository Structure

The repository is structured as follows:

```
- .gitignore
- README.md
- gru_model.py
- predictor_class.py
- requirements.txt
- svm_model.ipynb
```

- The `.gitignore` file specifies the files and directories that should be ignored by the Git version control system.

- The `README.md` file provides an overview of the project, its features, and instructions on how to use the repository.

- The `gru_model.py` file contains the implementation of the GRU-based RNN model for price prediction. This script defines the architecture of the model, trains it on the provided dataset, and saves the trained model for later use.

- The `predictor_class.py` file includes the class definition for the predictor. This class encapsulates the functionality of loading the trained models and making predictions based on the input data.

- The `requirements.txt` file lists the dependencies required to run the project. You can install these dependencies using the `pip install -r requirements.txt` command.

- The `svm_model.ipynb` notebook demonstrates the steps for training and evaluating the SVM model for brand name inference. It utilizes the dataset from the provided [Mercari](https://www.mercari.com/us/help_center/product-info/item-conditions/) link.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine using the command: 
   ```
   git clone https://github.com/your-username/product-price-prediction.git
   ```

2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

3. Open the `svm_model.ipynb` notebook to train and evaluate the SVM model for brand name inference using the provided dataset.

4. Use the `gru_model.py` script to train the GRU-based RNN model for price prediction. Adjust the script according to your requirements and run it to train the model.

5. Once the models are trained, you can utilize the `predictor_class.py` script to load the trained models and make predictions based on the input data.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please submit an issue or a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to acknowledge the authors of the original dataset from the [Mercari](https://www.mercari.com/us/help_center/product-info/item-conditions/) website for providing the data used in this project.

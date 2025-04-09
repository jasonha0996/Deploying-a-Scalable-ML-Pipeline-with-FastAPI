# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a **Logistic Regression** classifier trained to predict whether a person earns over \$50,000 per year based on U.S. Census data. It uses scikit-learn for modeling and includes preprocessing steps for categorical variables using OneHotEncoder and a LabelBinarizer for the binary target. The model and encoders are serialized using `pickle`.

## Intended Use

The model is intended for educational purposes in the context of deploying machine learning pipelines with FastAPI. It demonstrates how to preprocess data, train a model, perform evaluation on categorical slices, and serve predictions through a REST API.

## Training Data

The training data comes from the UCI Adult Census Income dataset. It includes demographic features such as:

- Workclass
- Education
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Native country

The dataset was split 80/20 into training and test datasets using `train_test_split`.

## Evaluation Data

The test set consists of 20% of the full dataset, held out from training. The same preprocessing steps (encoder and label binarizer) used in training were applied to the test data.

## Metrics

The model was evaluated using:
- **Precision**: 0.7357  
- **Recall**: 0.5633  
- **F1-score**: 0.6381  

Additionally, performance metrics were calculated for each categorical slice in `slice_output.txt` to ensure fairness across different subgroups.

## Ethical Considerations

This model is trained on a dataset that may contain historical biases based on gender, race, or education. These biases can be reflected in the model's predictions. The model is not intended for production use or decision-making in sensitive applications such as hiring, credit approval, or social services.

## Caveats and Recommendations

- The model may not generalize well to unseen distributions or demographic groups not well represented in the training data.
- Future work should include bias mitigation strategies, additional fairness evaluation, and hyperparameter tuning.
- Regular model retraining and auditing are recommended if this were to be used in real-world applications.
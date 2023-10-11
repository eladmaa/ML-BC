'''
course: advanced programing
submitted by: Elad Maayan, Dorian
description: 1. create model
             2. perform PCA (principal component analysis) to get 95% of the variance in order to reduce feature number
             3. train it
             4. use it
for target I used nottingham prognostic index
'''
import joblib

'''
install moduls to venv:
scikit-learn==1.2.2
keras
tensorflow
scikeras
joblib
pandas
'''
import sys
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
script_dir = os.path.dirname(os.path.abspath(__file__))

def create_model(shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=shape))  # First hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Remove activation for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(save_model_path, epochs, batch_size):
    genes = pd.read_csv(os.path.join(script_dir, "METABRIC_RNA_Mutation.csv"), low_memory=False)

    #genes = pd.read_csv("METABRIC_RNA_Mutation.csv")
    #X = genes[['brca1','brca2']].values
    X = genes.loc[:, 'brca1':'ugt2b7'].values
    y = genes['nottingham_prognostic_index'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform PCA to cover 95% of variance
    #pca = PCA(n_components=0.95, random_state=42)
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)

    # Create and compile the Keras model
    model = create_model(X_train.shape[1])

    # Train the model on your data
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")


    # Evaluate the model on the test set
    #loss, accuracy = model.evaluate(X_test_pca, y_test)
    # Evaluate the model using regression metrics
    predictions = model.predict(X_test)  #X_test_pca
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    # print(f"Mean Squared Error: {mse:.4f}")
    # print(f"Mean Absolute Error: {mae:.4f}")

    # Save the model
    model.save(save_model_path)
    joblib.dump(scaler, 'scaler.pkl')
    performance_data = {
    'Accuracy': accuracy,
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae
}

    # Define the path to the output JSON file
    output_json_file = './model/model_perform.json'

    # Serialize the dictionary to JSON and write it to the file
    with open(output_json_file, 'w') as json_file:
        json.dump(performance_data, json_file)
    return 0

def main(option, gene_expression_file, save_model_path, epochs=10, batch_size=32):
    scaler = StandardScaler()

    if option == '1':
        # Load the existing model
        model = tf.keras.models.load_model(save_model_path)
        scaler = joblib.load('./model/scaler.pkl')
        genes = pd.read_csv(gene_expression_file)
        #X = genes[['brca1': 'ugt2b7']].values
        X = genes.loc[:, 'brca1':'ugt2b7'].values
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        # Specify an absolute path for predictions.json
        predictions_file_path = os.path.abspath('./model/predictions.json')

        # Create and write the predictions to the absolute path
        with open(predictions_file_path, 'w') as predictions_file:
            json.dump(prediction.flatten().tolist(), predictions_file)

        print(f"Predicted Values: {prediction[:10].flatten()}")
        #sys.exit(predictions_file_path)  # Exit with the path to predictions.json
        sys.exit(0)
    elif option == '2':
        train_model(save_model_path, epochs, batch_size)
        sys.exit(0)
    else:
        print("Invalid option. Please use 1 to use an existing model or 2 to train a new model.")

if __name__ == '__main__':
    option = sys.argv[1]
    if option == '1' and len(sys.argv) == 4:  # use model
        gene_expression_file = sys.argv[2]
        saved_model_path = sys.argv[3]
       
        main(option, gene_expression_file, saved_model_path)
    elif option == '2' and len(sys.argv) == 6:  #   train model
        print("arguments recieved: ", sys.argv)
        dataset = os.path.join(script_dir, sys.argv[2])
        print("--------dataset:", dataset)
        epochs = int(sys.argv[4])  # Extract epochs from command line arguments
        batch_size = int(sys.argv[5])  # Extract batch size from command line arguments
        saved_model_path = sys.argv[3]
        main(option, dataset, saved_model_path)
    else:
        print("Usage: python main.py 2 (train model) [dataset_file] [save_model_path] [epochs] [batch_size]")
        print("Or: python main.py 1 (use model) [gene_expression_file] [load_model_path] [save_model_path] [epochs] [batch_size]")
        print('option: ', option, ' total ', len(sys.argv))
#1st run string: 1 gene_vector.csv saved_model.h5
#2nd run string: 2 METABRIC_RNA_Mutation.csv saved_model.h5
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os


def init_params():
  
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
   
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
   
    return Z > 0

def softmax(Z):
    
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):

    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y, classes=10):
    
    one_hot_Y = np.zeros((Y.size, classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
   
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
   
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
   
    W1, b1, W2, b2 = init_params()
    _, m = X.shape
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Accuracy: {accuracy:.2%}")
            
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction_with_label(index, X, Y, W1, b1, W2, b2):
    
    current_image = X[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y[index]
    
    print(f"Prediction: {prediction[0]}")
    print(f"Actual Label: {label}")
    
   
    current_image = current_image.reshape((28, 28)) * 255
    plt.figure(figsize=(3, 3))
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Actual: {label}")
    plt.show()

def save_model(W1, b1, W2, b2, filename='mnist_model.pkl'):
  
    with open(filename, 'wb') as f:
        pickle.dump((W1, b1, W2, b2), f)
    print(f"Model saved to {filename}")

def load_model(filename='mnist_model.pkl'):
    """Load a trained model"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            W1, b1, W2, b2 = pickle.load(f)
        print(f"Model loaded from {filename}")
        return W1, b1, W2, b2
    else:
        print(f"Model file {filename} not found. Please train the model first.")
        return None, None, None, None

def preprocess_train_data(data_path):
   
   
    data = pd.read_csv(data_path)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  
    
   
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.  
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.  
    
    return X_train, Y_train, X_dev, Y_dev

def get_user_choice():
    
    print("\nMNIST Digit Recognition - Menu")
    print("-------------------------------")
    print("1. Train new model")
    print("2. Load existing model")
    print("3. Test on training examples")
    print("4. Test on development set")
    print("5. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if 1 <= choice <= 5:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    
    print("MNIST Digit Recognition Neural Network")
    print("--------------------------------------")
    
    train_path = './digit-recognizer/train.csv' 
    model_path = 'mnist_model.pkl'
    
   
    X_train, Y_train, X_dev, Y_dev = None, None, None, None
    W1, b1, W2, b2 = None, None, None, None
    
   
    if not os.path.exists(train_path):
        print(f"Warning: Training data file {train_path} not found!")
    
    while True:
        choice = get_user_choice()
        
        if choice == 1:  
            if X_train is None:
                print("\nLoading and preprocessing training data...")
                if os.path.exists(train_path):
                    X_train, Y_train, X_dev, Y_dev = preprocess_train_data(train_path)
                    print(f"Data loaded: {X_train.shape[1]} training examples, {X_dev.shape[1]} dev examples")
                else:
                    print(f"Error: Training data file {train_path} not found!")
                    continue
                    
            print("\nTraining neural network...")
            iterations = int(input("Enter number of iterations (recommended: 500): "))
            alpha = float(input("Enter learning rate (recommended: 0.1): "))
            W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=alpha, iterations=iterations)
            
            save_choice = input("\nDo you want to save the model? (y/n): ").lower()
            if save_choice == 'y':
                save_model(W1, b1, W2, b2, model_path)
        
        elif choice == 2:  
            W1, b1, W2, b2 = load_model(model_path)
            if W1 is None:
                print("Could not load model. Please train a new model first.")
        
        elif choice == 3:  
            if W1 is None:
                print("No model loaded. Please train or load a model first.")
                continue
                
            if X_train is None:
                print("\nLoading and preprocessing training data...")
                if os.path.exists(train_path):
                    X_train, Y_train, X_dev, Y_dev = preprocess_train_data(train_path)
                else:
                    print(f"Error: Training data file {train_path} not found!")
                    continue
            
            while True:
                try:
                    idx = int(input(f"\nEnter index of training example to test (0-{X_train.shape[1]-1}, -1 to return to menu): "))
                    if idx == -1:
                        break
                    if 0 <= idx < X_train.shape[1]:
                        test_prediction_with_label(idx, X_train, Y_train, W1, b1, W2, b2)
                    else:
                        print(f"Invalid index. Please enter a number between 0 and {X_train.shape[1]-1}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        elif choice == 4: 
            if W1 is None:
                print("No model loaded. Please train or load a model first.")
                continue
                
            if X_dev is None:
                print("\nLoading and preprocessing training data...")
                if os.path.exists(train_path):
                    X_train, Y_train, X_dev, Y_dev = preprocess_train_data(train_path)
                else:
                    print(f"Error: Training data file {train_path} not found!")
                    continue
            
            print("\nTesting on development set...")
            dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
            dev_accuracy = get_accuracy(dev_predictions, Y_dev)
            print(f"Development set accuracy: {dev_accuracy:.2%}")
            
            while True:
                try:
                    idx = int(input(f"\nEnter index of dev example to test (0-{X_dev.shape[1]-1}, -1 to return to menu): "))
                    if idx == -1:
                        break
                    if 0 <= idx < X_dev.shape[1]:
                        test_prediction_with_label(idx, X_dev, Y_dev, W1, b1, W2, b2)
                    else:
                        print(f"Invalid index. Please enter a number between 0 and {X_dev.shape[1]-1}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        elif choice == 5:  
            print("\nExiting program. Goodbye!")
            break


if __name__ == "__main__":
    main()
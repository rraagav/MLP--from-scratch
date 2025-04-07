import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, MultiLabelBinarizer
import wandb
import pickle

class DataExploration:
    def __init__(self,df):
        self.df = df

    def describe_dataset(self, print_all):
        if print_all:
            print(self.df.describe())
        # print(self.df.head())
        # print(self.df.info())

    def correlation_list(self, printer):
        correlationdict = {}        
        for feature in self.df.columns:
            correlation = self.df[feature].corr(self.df['quality'])
            correlationdict[feature] = correlation

        average_correlation = {}

        for feature, correlation_value in correlationdict.items():
            average_correlation[feature] = abs(correlation_value)

        sorted_table = sorted(average_correlation.items(), key=lambda item: item[1], reverse=True)

        headers = ["Feature", "Absolute Correlation"]
        if printer:
            print(tabulate(sorted_table, headers, tablefmt="grid"))

class EDA_Plots(DataExploration):        
    def histogram_of_features(self, plotter):
        if plotter:
            self.df.hist(figsize=(15, 10), bins=1140, edgecolor = "blue")
            plt.suptitle('How the Numerical Features are Distributed')
            plt.show()

    def correlation_matrix(self, plotter):
        if plotter:
            correlation_matrix = self.df.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()

    def pairplot_top_features(self, plotter):
        if plotter:
            selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
            sns.pairplot(self.df, vars=selected_features, hue='quality', palette='tab20', plot_kws={'alpha': 0.5})
            plt.suptitle('Pairplot of Selected Features Colored by Genre', y=1.02)
            plt.show()    
    
    def pairplot_all_features(self, plotter):
        if plotter:
            sns.pairplot(self.df, hue='quality', palette='tab20', plot_kws={'alpha': 0.5})
            plt.suptitle('Pairplot of Selected Features Colored by Genre', y=1.02)
            plt.show()

class DataSplit(EDA_Plots):

    def normalize_data(self):
        features = self.df.drop(columns=['quality', 'Id'])
        labels = self.df['quality']
    
        ss = StandardScaler()
        features_standardized = ss.fit_transform(features)
        
        normz = MinMaxScaler()
        features_normalized = normz.fit_transform(features_standardized)

        df_normalized = pd.DataFrame(features_normalized, columns=features.columns)

        df_normalized['quality'] = labels.values
        df_normalized['Id'] = self.df['Id']

        path = 'C:/Users/raaga/OneDrive/Desktop/IIIT-H/3-1/SMAI/smai-m24-assignments-rraagav/data/interim/3/WineQT_Normalized.csv'
        df_normalized.to_csv(path, index=False)  

        return df_normalized

    def split_data_wine(self, data):
        features = data.drop(columns=['quality', 'Id'])
        # features = data[['alcohol', 'volatile acidity', 'sulphates', 'citric acid']]      
        labels = pd.Categorical(data['quality']).codes

        # print("Original labels:")
        # print(data['quality'].unique())
        
        # print("\nEncoded labels:")
        # print(np.unique(labels))
        
        # # Display the distribution of labels
        # print('\nDistribution of the original labels:')
        # print(data['quality'].value_counts())
        
        np.random.seed(42)
        indices = np.arange(len(features))
        np.random.shuffle(indices)

        train_size = 0.8
        train_index = int(len(features) * train_size)

        val_size = 0.1
        val_index = int(len(features) * val_size)
        
        train_indices = indices[:train_index]
        val_indices = indices[train_index:train_index+val_index]
        test_indices = indices[train_index+val_index:]
    
        X_train = features.iloc[train_indices].values
        y_train = labels[train_indices]

        X_eval = features.iloc[val_indices].values
        y_eval = labels[val_indices]

        X_test = features.iloc[test_indices].  values
        y_test = labels[test_indices]

        return X_train, y_train, X_eval, y_eval, X_test, y_test

class MLP_Classifier:
    def __init__(self, input_size, output_size):
        self.alpha = None
        self.activation_function = None
        self.activation_derivative = None
        self.optimizers = None
        self.hidden_layers = None
        self.neurons_per_layer = None
        self.batch_size = None
        self.epochs = None
        self.weights = []
        self.biases = []
        self.losses = []
        self.accuracies = []
        self.input_size = input_size
        self.output_size = output_size
        
    def set_params(self, alpha, activation_function, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs):
        self.alpha = alpha
        self.set_activation_function(activation_function)
        self.optimizers = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.epochs = epochs
        self.batch_size = batch_size
        
    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.neurons_per_layer + [self.output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 # Xavier initialization, or He 
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
    def forward(self, X):
        activations = [X]
        Zs = []
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            Z = np.dot(activations[-1], w) + b
            Zs.append(Z)
            if i == len(self.weights) - 1: 
                # Output/last layer in nn. Do softmax. 
                A = self.softmax(Z)
            else:
                # hidden layer present, use the activation function 
                A = self.activation_function(Z)
            activations.append(A)
        return activations, Zs
        
    def backward(self, activations, Zs, y):
        gradients_w = []
        gradients_b = []
        m = y.shape[0]
        
        # One-hot encode y
        y_one_hot = np.zeros_like(activations[-1])
        y_one_hot[np.arange(m), y] = 1
        
        # Output layer error
        delta = activations[-1] - y_one_hot 
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m  
            dB = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.activation_derivative(Zs[i-1])
        return gradients_w, gradients_b

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def tanh_derivative(self, Z):
        return 1 - np.tanh(Z) ** 2
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def linear(self, Z):
        return Z
    
    def linear_derivative(self, Z):
        return np.ones_like(Z)
    
    def softmax(self, Z):
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def set_activation_function(self, func_name):
        if func_name == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif func_name == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif func_name == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif func_name == "linear":
            self.activation_function = self.linear
            self.activation_derivative = self.linear_derivative
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-8)) / m
        return loss

    def compute_accuracy(self, y_pred, y_true):
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_true)
        return accuracy
        
    def compute_classification_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return precision, recall, f1

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients_w[i]
            self.biases[i] -= self.alpha * gradients_b[i]
    
    def bgd(self, X_train, y_train):
        activations, Zs = self.forward(X_train)
        gradients_w, gradients_b = self.backward(activations, Zs, y_train)
        self.update_parameters(gradients_w, gradients_b)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def mini_bgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        num_batches = int(np.ceil(m / self.batch_size))
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            activations, Zs = self.forward(X_batch)
            gradients_w, gradients_b = self.backward(activations, Zs, y_batch)
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def sgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        for i in range(m):
            X_sample = X_shuffled[i:i+1]
            y_sample = y_shuffled[i:i+1]
            activations, Zs = self.forward(X_sample)
            gradients_w, gradients_b = self.backward(activations, Zs, y_sample)
            
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def fit(self, X_train, y_train, early_stopping, patience=5):
        m = X_train.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        self.initialize_weights()
        
        for epoch in range(self.epochs):
            if self.optimizers == 'bgd':
                epoch_loss, epoch_accuracy = self.bgd(X_train, y_train)
            elif self.optimizers == 'mini_bgd':
                epoch_loss, epoch_accuracy = self.mini_bgd(X_train, y_train)
            elif self.optimizers == 'sgd':
                epoch_loss, epoch_accuracy = self.sgd(X_train, y_train)
            else:
                raise ValueError("Invalid optimizer name")
            
            self.losses.append(epoch_loss)
            self.accuracies.append(epoch_accuracy)

            # Early stopping part
            if early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    def predict(self, X):
        activations, Zs = self.forward(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return y_pred
    
    def gradient_check(self, X, y, epsilon=1e-7):
        activations, Zs = self.forward(X)
        gradients_w, gradients_b = self.backward(activations, Zs, y)
        
        params = []
        grads = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        for gw, gb in zip(gradients_w, gradients_b):

            
            grads.extend([gw, gb])
        
        num_grads = []
        for param in params:
            num_grad = np.zeros_like(param)
            iterator = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not iterator.finished:
                idx = iterator.multi_index
                original_value = param[idx]
                
                param[idx] = original_value + epsilon
                activations_plus, _ = self.forward(X)
                loss_plus = self.compute_loss(activations_plus[-1], y)
                
                param[idx] = original_value - epsilon
                activations_minus, _ = self.forward(X)
                loss_minus = self.compute_loss(activations_minus[-1], y)
                
                param[idx] = original_value
                
                num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                iterator.iternext()
            num_grads.append(num_grad)
        
        for i in range(len(grads)):
            grad = grads[i]
            num_grad = num_grads[i]
            numerator = np.linalg.norm(grad - num_grad)
            denominator = np.linalg.norm(grad) + np.linalg.norm(num_grad) + 1e-8
            relative_error = numerator / denominator
            print(f"Parameter {i+1} - Relative Error: {relative_error:.10e}")
            if relative_error > 1e-5:
                print("Gradients dont match!!!vfgsdgbhjxndwgtbeiurowefnipwmod.")
                return
        print("It worksssssssssssssssss")

class MLP_Classifier_HPT:
    def __init__(self, input_size, output_size):
        self.alpha = None
        self.activation_function = None
        self.activation_derivative = None
        self.optimizers = None
        self.hidden_layers = None
        self.neurons_per_layer = None
        self.batch_size = None
        self.epochs = None
        self.weights = []
        self.biases = []
        self.losses = []
        self.accuracies = []
        self.input_size = input_size
        self.output_size = output_size
        
    def set_params(self, alpha, activation_function, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs):
        self.alpha = alpha
        self.set_activation_function(activation_function)
        self.optimizers = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.epochs = epochs
        self.batch_size = batch_size
        
    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.neurons_per_layer + [self.output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 # Xavier initialization
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
    def forward(self, X):
        activations = [X]
        Zs = []
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            Z = np.dot(activations[-1], w) + b
            Zs.append(Z)
            if i == len(self.weights) - 1: 
                # Output/last layer in nn. Do softmax. 
                A = self.softmax(Z)
            else:
                # hidden layer present, use the activation function 
                A = self.activation_function(Z)
            activations.append(A)
        return activations, Zs
        
    def backward(self, activations, Zs, y):
        gradients_w = []
        gradients_b = []
        m = y.shape[0]
        
        y_one_hot = np.zeros_like(activations[-1])
        y_one_hot[np.arange(m), y] = 1
        
        delta = activations[-1] - y_one_hot  
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m 
            dB = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.activation_derivative(Zs[i-1])
        return gradients_w, gradients_b

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def tanh_derivative(self, Z):
        return 1 - np.tanh(Z) ** 2
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def linear(self, Z):
        return Z
    
    def linear_derivative(self, Z):
        return np.ones_like(Z)
    
    def softmax(self, Z):
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def set_activation_function(self, func_name):
        if func_name == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif func_name == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif func_name == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif func_name == "linear":
            self.activation_function = self.linear
            self.activation_derivative = self.linear_derivative
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-8)) / m
        return loss

    def compute_accuracy(self, y_pred, y_true):
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_true)
        return accuracy
    
    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients_w[i]
            self.biases[i] -= self.alpha * gradients_b[i]
    
    def bgd(self, X_train, y_train):
        activations, Zs = self.forward(X_train)
        gradients_w, gradients_b = self.backward(activations, Zs, y_train)
        self.update_parameters(gradients_w, gradients_b)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def mini_bgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        num_batches = int(np.ceil(m / self.batch_size))
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            activations, Zs = self.forward(X_batch)
            gradients_w, gradients_b = self.backward(activations, Zs, y_batch)
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def sgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        for i in range(m):
            X_sample = X_shuffled[i:i+1]
            y_sample = y_shuffled[i:i+1]
            activations, Zs = self.forward(X_sample)
            gradients_w, gradients_b = self.backward(activations, Zs, y_sample)
            
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def compute_classification_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return precision, recall, f1

    def fit(self, X_train, y_train, X_eval, y_eval, early_stopping=False, patience=5):
        m = X_train.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        self.initialize_weights()
        
        for epoch in range(self.epochs):
            if self.optimizers == 'bgd':
                epoch_loss, epoch_accuracy = self.bgd(X_train, y_train)
            elif self.optimizers == 'mini_bgd':
                epoch_loss, epoch_accuracy = self.mini_bgd(X_train, y_train)
            elif self.optimizers == 'sgd':
                epoch_loss, epoch_accuracy = self.sgd(X_train, y_train)
            else:
                raise ValueError("Invalid optimizer name")
            
            # Evaluate on validation set
            # if X_eval is not None and y_eval is not None:
            val_activations, _ = self.forward(X_eval)
            val_loss = self.compute_loss(val_activations[-1], y_eval)
            val_accuracy = self.compute_accuracy(val_activations[-1], y_eval)
            y_eval_pred = self.predict(X_eval)
            y_train_pred = self.predict(X_train)
            
            val_precision, val_recall, val_f1 = self.compute_classification_metrics(y_eval, y_eval_pred)
            train_precision, train_recall, train_f1 = self.compute_classification_metrics(y_train, y_train_pred)

            metrics = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_accuracy': epoch_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1_score': train_f1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1_score': val_f1
            }
            
            wandb.log(metrics)
            
            if early_stopping and val_loss is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self.save_model('best_model.npz')
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            if val_loss is not None:
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def predict(self, X):
        activations, Zs = self.forward(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return y_pred
    
    def gradient_check(self, X, y, epsilon=1e-7):
        activations, Zs = self.forward(X)
        gradients_w, gradients_b = self.backward(activations, Zs, y)
        
        params = []
        grads = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        for gw, gb in zip(gradients_w, gradients_b):
            grads.extend([gw, gb])
        
        num_grads = []
        for param in params:
            num_grad = np.zeros_like(param)
            iterator = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not iterator.finished:
                idx = iterator.multi_index
                original_value = param[idx]
                
                param[idx] = original_value + epsilon
                activations_plus, _ = self.forward(X)
                loss_plus = self.compute_loss(activations_plus[-1], y)
                
                param[idx] = original_value - epsilon
                activations_minus, _ = self.forward(X)
                loss_minus = self.compute_loss(activations_minus[-1], y)
                
                param[idx] = original_value
                
                num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                iterator.iternext()
            num_grads.append(num_grad)
        
        for i in range(len(grads)):
            grad = grads[i]
            num_grad = num_grads[i]
            numerator = np.linalg.norm(grad - num_grad)
            denominator = np.linalg.norm(grad) + np.linalg.norm(num_grad) + 1e-8
            relative_error = numerator / denominator
            print(f"Parameter {i+1} - Relative Error: {relative_error:.10e}")
            if relative_error > 1e-5:
                print("Gradients dont match!!!vfgsdgbhjxndwgtbeiurowefnipwmod.")
                return
        print("It worksssssssssssssssss")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            data = {
                'weights': self.weights,
                'biases': self.biases,
                'activation_function': self.activation_function.__name__,
                'optimizer': self.optimizers,
                'alpha': self.alpha,
                'hidden_layers': self.hidden_layers,
                'neurons_per_layer': self.neurons_per_layer
            }
            pickle.dump(data, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.biases = data['biases']
        self.set_activation_function(data['activation_function'])
        self.optimizers = data['optimizer']
        self.alpha = data['alpha']
        self.hidden_layers = data['hidden_layers']
        self.neurons_per_layer = data['neurons_per_layer']

class DataSplit_multilabel(EDA_Plots):

    def normalize_data(self):
        df_new = self.df.copy()
        df_new.drop(columns=['occupation', 'city'], inplace=True)

        le_gender = LabelEncoder()
        df_new['gender'] = le_gender.fit_transform(df_new['gender'])

        # For 'education', encode as ordinal
        education_order = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
        df_new['education'] = df_new['education'].map(education_order)

        le_married = LabelEncoder()
        df_new['married'] = le_married.fit_transform(df_new['married'])

        df_new.drop(columns=['most bought item'], inplace=True)

        ss = StandardScaler()
        df_new[['income', 'purchase_amount']] = ss.fit_transform(df_new[['income', 'purchase_amount']])

        df_new['labels'] = df_new['labels'].apply(lambda x: x.split())
        mlb = MultiLabelBinarizer()
        label_encoded = mlb.fit_transform(df_new['labels'])
        labels_df = pd.DataFrame(label_encoded, columns=mlb.classes_)

        df_new = pd.concat([df_new.drop(columns=['labels']), labels_df], axis=1)

        path = r"C:\Users\raaga\OneDrive\Desktop\IIIT-H\3-1\SMAI\smai-m24-assignments-rraagav\data\interim\3\advertisement_norm.csv"
        df_new.to_csv(path, index=False)

        return df_new, mlb

    def split_data(self, data):
        # Assuming data is df_new from normalize_data()
        # Split data into features and labels
        label_columns = [col for col in data.columns if col not in ['age', 'gender', 'income', 'education', 'married', 'children', 'purchase_amount']]
        
        features = data[['age', 'gender', 'income', 'education', 'married', 'children', 'purchase_amount']]
        labels = data[label_columns]

        # Then split into train, val, test
        np.random.seed(42)
        indices = np.arange(len(features))
        np.random.shuffle(indices)

        train_size = 0.8
        val_size = 0.1

        train_end = int(len(features) * train_size)
        val_end = train_end + int(len(features) * val_size)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        X_train = features.iloc[train_indices].values
        y_train = labels.iloc[train_indices].values

        X_eval = features.iloc[val_indices].values
        y_eval = labels.iloc[val_indices].values

        X_test = features.iloc[test_indices].values
        y_test = labels.iloc[test_indices].values

        return X_train, y_train, X_eval, y_eval, X_test, y_test, label_columns      

class MLP_MultiLabel_Classifier:
    def __init__(self, input_size, output_size):
        self.alpha = None
        self.activation_function = None
        self.activation_derivative = None
        self.optimizer = None
        self.hidden_layers = None
        self.neurons_per_layer = None
        self.batch_size = None
        self.epochs = None
        self.weights = []
        self.biases = []
        self.losses = []
        self.accuracies = []
        self.input_size = input_size
        self.output_size = output_size

    def set_params(self, alpha, activation_function, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs):
        self.alpha = alpha
        self.set_activation_function(activation_function)
        self.optimizer = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.epochs = epochs
        self.batch_size = batch_size

    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.neurons_per_layer + [self.output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            # w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])  # He initialization, instead of Xavier
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 # Xavier initialization
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        Zs = []
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            Z = np.dot(activations[-1], w) + b
            Zs.append(Z)
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.activation_function(Z)
            activations.append(A)
        return activations, Zs

    def backward(self, activations, Zs, y):
        gradients_w = []
        gradients_b = []
        m = y.shape[0]

        delta = activations[-1] - y  

        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m  
            dB = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.activation_derivative(Zs[i-1])
        return gradients_w, gradients_b

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def tanh(self, Z):
        return np.tanh(Z)

    def tanh_derivative(self, Z):
        return 1 - np.tanh(Z) ** 2

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def linear(self, Z):
        return Z

    def linear_derivative(self, Z):
        return np.ones_like(Z)

    def set_activation_function(self, func_name):
        if func_name == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif func_name == "tanh":
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif func_name == "relu":
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif func_name == "linear":
            self.activation_function = self.linear
            self.activation_derivative = self.linear_derivative

    def compute_loss(self, y_pred, y_true):
        epsilon = 1e-8  # To prevent log(0)
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    def compute_accuracy(self, y_pred, y_true):
        y_pred_labels = (y_pred >= 0.5).astype(int)
        correct_predictions = (y_pred_labels == y_true).astype(int)
        accuracy = np.sum(correct_predictions) / (correct_predictions.shape[0] * correct_predictions.shape[1])
        return accuracy

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients_w[i]
            self.biases[i] -= self.alpha * gradients_b[i]

    def mini_bgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        num_batches = int(np.ceil(m / self.batch_size))
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            activations, Zs = self.forward(X_batch)
            gradients_w, gradients_b = self.backward(activations, Zs, y_batch)
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
        
    def sgd(self, X_train, y_train):
        m = X_train.shape[0]
        perm = np.random.permutation(m)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        for i in range(m):
            X_sample = X_shuffled[i:i+1]
            y_sample = y_shuffled[i:i+1]
            activations, Zs = self.forward(X_sample)
            gradients_w, gradients_b = self.backward(activations, Zs, y_sample)
            
            self.update_parameters(gradients_w, gradients_b)
        activations, _ = self.forward(X_train)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy
    
    def bgd(self, X_train, y_train):
        activations, Zs = self.forward(X_train)
        gradients_w, gradients_b = self.backward(activations, Zs, y_train)
        self.update_parameters(gradients_w, gradients_b)
        loss = self.compute_loss(activations[-1], y_train)
        accuracy = self.compute_accuracy(activations[-1], y_train)
        return loss, accuracy

    def compute_classification_metrics(self, y_true, y_pred):
        y_pred_labels = (y_pred >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred_labels, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred_labels, average='samples', zero_division=0)
        f1 = f1_score(y_true, y_pred_labels, average='samples', zero_division=0)
        hamming = hamming_loss(y_true, y_pred_labels)
        return precision, recall, f1, hamming

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, early_stopping=False, patience=5):
        m = X_train.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        self.initialize_weights()

        for epoch in range(self.epochs):
            if self.optimizer == 'bgd':
                epoch_loss, epoch_accuracy = self.bgd(X_train, y_train)
            elif self.optimizer == 'mini_bgd':
                epoch_loss, epoch_accuracy = self.mini_bgd(X_train, y_train)
            elif self.optimizer == 'sgd':
                epoch_loss, epoch_accuracy = self.sgd(X_train, y_train)
            else:
                raise ValueError("Invalid optimizer name")

            if X_eval is not None and y_eval is not None:
                val_activations, _ = self.forward(X_eval)
                val_loss = self.compute_loss(val_activations[-1], y_eval)
                val_accuracy = self.compute_accuracy(val_activations[-1], y_eval)
                y_eval_pred = val_activations[-1]
                y_train_pred = self.forward(X_train)[0][-1]

                val_precision, val_recall, val_f1, val_hamming = self.compute_classification_metrics(y_eval, y_eval_pred)
                train_precision, train_recall, train_f1, train_hamming = self.compute_classification_metrics(y_train, y_train_pred)

                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_accuracy,
                    'train_precision': train_precision,
                    'train_recall': train_recall,
                    'train_f1_score': train_f1,
                    'train_hamming_loss': train_hamming,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1_score': val_f1,
                    'val_hamming_loss': val_hamming
                }

                wandb.log(metrics)

                if early_stopping and val_loss is not None:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        # Save the best model weights
                        self.save_model('best_model_2.6.pkl')
                    else:
                        patience_counter += 1
                    if patience_counter > patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1 Score: {train_f1:.4f}, Train Hamming Loss: {train_hamming:.4f}")
                print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1 Score: {val_f1:.4f}, Validation Hamming Loss: {val_hamming:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': epoch_accuracy
                }
                wandb.log(metrics)

    def predict(self, X):
        activations, _ = self.forward(X)
        y_pred = (activations[-1] >= 0.5).astype(int)
        return y_pred

    def save_model(self, filename):
        with open(filename, 'wb') as f:  
            data = {
                'weights': self.weights,
                'biases': self.biases,
                'activation_function': self.activation_function.__name__,
                'optimizer': self.optimizer,
                'alpha': self.alpha,
                'hidden_layers': self.hidden_layers,
                'neurons_per_layer': self.neurons_per_layer
            }
            pickle.dump(data, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.biases = data['biases']
        self.set_activation_function(data['activation_function'])
        self.optimizer = data['optimizer']
        self.alpha = data['alpha']
        self.hidden_layers = data['hidden_layers']
        self.neurons_per_layer = data['neurons_per_layer']

df = pd.read_csv(r'C:\Users\raaga\OneDrive\Desktop\IIIT-H\3-1\SMAI\smai-m24-assignments-rraagav\data\interim\3\WineQT.csv')
DataExplorer = DataExploration(df)
DataExplorer.describe_dataset(print_all=False)
# df.describe()
DataExplorer.correlation_list(printer=False)

Explorer_plots = EDA_Plots(df)
Explorer_plots.histogram_of_features(plotter=False)
Explorer_plots.correlation_matrix(plotter=False)
Explorer_plots.pairplot_top_features(plotter=False)
Explorer_plots.pairplot_all_features(plotter=False)

splitter = DataSplit(df)
df_normalized = splitter.normalize_data()

X_train, y_train, X_eval, y_eval, X_test, y_test = splitter.split_data(df)

# For Best Model 
output_txt_file = open("C:/Users/raaga/OneDrive/Desktop/IIIT-H/3-1/SMAI/smai-m24-assignments-rraagav/assignments/3/figures/best_accuracy.txt", "w")

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

mlp_activ = MLP_Classifier(input_size, output_size)

mlp_activ.set_params(alpha=0.01, activation_function='tanh', optimizer='sgd', hidden_layers= 3, neurons_per_layer=[128, 64, 32], batch_size=32, epochs=50)
mlp_activ.fit(X_train, y_train, early_stopping=True, patience=5)

test_activations, _ = mlp_activ.forward(X_test)
test_loss = mlp_activ.compute_loss(test_activations[-1], y_test)
test_accuracy = mlp_activ.compute_accuracy(test_activations[-1], y_test)
test_precision, test_recall, test_f1 = mlp_activ.compute_classification_metrics(y_test, mlp_activ.predict(X_test))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

train_activations, _ = mlp_activ.forward(X_train)
train_loss = mlp_activ.compute_loss(train_activations[-1], y_train)
train_accuracy = mlp_activ.compute_accuracy(train_activations[-1], y_train)
train_precision, train_recall, train_f1 = mlp_activ.compute_classification_metrics(y_train, mlp_activ.predict(X_train))
print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

eval_activations, _ = mlp_activ.forward(X_eval)
eval_loss = mlp_activ.compute_loss(eval_activations[-1], y_eval)
eval_accuracy = mlp_activ.compute_accuracy(eval_activations[-1], y_eval)
eval_precision, eval_recall, eval_f1 = mlp_activ.compute_classification_metrics(y_eval, mlp_activ.predict(X_eval))
print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
print(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}")

output_txt_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
output_txt_file.write(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}\n\n")

output_txt_file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
output_txt_file.write(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}\n\n")

output_txt_file.write(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}\n")
output_txt_file.write(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}")

# Effect of non linearity
activ_list = ['sigmoid', 'tanh', 'relu', 'linear']
output_txt_file = open("C:/Users/raaga/OneDrive/Desktop/IIIT-H/3-1/SMAI/smai-m24-assignments-rraagav/assignments/3/figures/activation_functions.txt", "w")

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

mlp_activ = MLP_Classifier(input_size, output_size)

for activ in activ_list:
    mlp_activ.set_params(alpha=0.01, activation_function=activ, optimizer='sgd', hidden_layers= 3, neurons_per_layer=[128, 64, 32], batch_size=32, epochs=50)
    mlp_activ.fit(X_train, y_train, early_stopping=True, patience=5)

    test_activations, _ = mlp_activ.forward(X_test)
    test_loss = mlp_activ.compute_loss(test_activations[-1], y_test)
    test_accuracy = mlp_activ.compute_accuracy(test_activations[-1], y_test)
    test_precision, test_recall, test_f1 = mlp_activ.compute_classification_metrics(y_test, mlp_activ.predict(X_test))
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    train_activations, _ = mlp_activ.forward(X_train)
    train_loss = mlp_activ.compute_loss(train_activations[-1], y_train)
    train_accuracy = mlp_activ.compute_accuracy(train_activations[-1], y_train)
    train_precision, train_recall, train_f1 = mlp_activ.compute_classification_metrics(y_train, mlp_activ.predict(X_train))
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

    eval_activations, _ = mlp_activ.forward(X_eval)
    eval_loss = mlp_activ.compute_loss(eval_activations[-1], y_eval)
    eval_accuracy = mlp_activ.compute_accuracy(eval_activations[-1], y_eval)
    eval_precision, eval_recall, eval_f1 = mlp_activ.compute_classification_metrics(y_eval, mlp_activ.predict(X_eval))
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
    print(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}")

    output_txt_file.write(f"Activation Function: {activ}\n")
    output_txt_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
    output_txt_file.write(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}\n\n")

    output_txt_file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
    output_txt_file.write(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}\n\n")

    output_txt_file.write(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}\n")
    output_txt_file.write(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}\n\n")

# Effect of Learning Rate:
lr_list = [0.001, 0.01, 0.1, 0.5]
output_txt_file = open("C:/Users/raaga/OneDrive/Desktop/IIIT-H/3-1/SMAI/smai-m24-assignments-rraagav/assignments/3/figures/vary_alpha.txt", "w")

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

mlp_a = MLP_Classifier(input_size, output_size)

for a in lr_list:
    mlp_a.set_params(alpha=a, activation_function='tanh', optimizer='sgd', hidden_layers= 3, neurons_per_layer=[128, 64, 32], batch_size=32, epochs=50)
    mlp_a.fit(X_train, y_train, early_stopping=True, patience=5)

    test_activations, _ = mlp_a.forward(X_test)
    test_loss = mlp_a.compute_loss(test_activations[-1], y_test)
    test_accuracy = mlp_a.compute_accuracy(test_activations[-1], y_test)
    test_precision, test_recall, test_f1 = mlp_a.compute_classification_metrics(y_test, mlp_a.predict(X_test))
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    train_activations, _ = mlp_a.forward(X_train)
    train_loss = mlp_a.compute_loss(train_activations[-1], y_train)
    train_accuracy = mlp_a.compute_accuracy(train_activations[-1], y_train)
    train_precision, train_recall, train_f1 = mlp_a.compute_classification_metrics(y_train, mlp_a.predict(X_train))
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

    eval_activations, _ = mlp_a.forward(X_eval)
    eval_loss = mlp_a.compute_loss(eval_activations[-1], y_eval)
    eval_accuracy = mlp_a.compute_accuracy(eval_activations[-1], y_eval)
    eval_precision, eval_recall, eval_f1 = mlp_a.compute_classification_metrics(y_eval, mlp_a.predict(X_eval))
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
    print(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}")

    output_txt_file.write(f"Learning Rate: {a}\n")
    output_txt_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
    output_txt_file.write(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}\n\n")

    output_txt_file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
    output_txt_file.write(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}\n\n")

    output_txt_file.write(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}\n")
    output_txt_file.write(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}\n\n")

# Effect of Batch Size

batch_size = [16, 32, 64, 128]
output_txt_file = open("C:/Users/raaga/OneDrive/Desktop/IIIT-H/3-1/SMAI/smai-m24-assignments-rraagav/assignments/3/figures/vary_batch_size.txt", "w")

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

mlp_bs = MLP_Classifier(input_size, output_size)

for bs in batch_size:
    mlp_bs.set_params(alpha=0.01, activation_function='tanh', optimizer='sgd', hidden_layers= 3, neurons_per_layer=[128, 64, 32], batch_size=bs, epochs=50)
    mlp_bs.fit(X_train, y_train, early_stopping=True, patience=5)

    test_activations, _ = mlp_bs.forward(X_test)
    test_loss = mlp_bs.compute_loss(test_activations[-1], y_test)
    test_accuracy = mlp_bs.compute_accuracy(test_activations[-1], y_test)
    test_precision, test_recall, test_f1 = mlp_bs.compute_classification_metrics(y_test, mlp_bs.predict(X_test))
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    train_activations, _ = mlp_bs.forward(X_train)
    train_loss = mlp_bs.compute_loss(train_activations[-1], y_train)
    train_accuracy = mlp_bs.compute_accuracy(train_activations[-1], y_train)
    train_precision, train_recall, train_f1 = mlp_bs.compute_classification_metrics(y_train, mlp_bs.predict(X_train))
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

    eval_activations, _ = mlp_bs.forward(X_eval)
    eval_loss = mlp_bs.compute_loss(eval_activations[-1], y_eval)
    eval_accuracy = mlp_bs.compute_accuracy(eval_activations[-1], y_eval)
    eval_precision, eval_recall, eval_f1 = mlp_bs.compute_classification_metrics(y_eval, mlp_bs.predict(X_eval))
    print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
    print(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}")

    output_txt_file.write(f"Batch Size: {bs}\n")
    output_txt_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
    output_txt_file.write(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}\n\n")

    output_txt_file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n")
    output_txt_file.write(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}\n\n")

    output_txt_file.write(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}\n")
    output_txt_file.write(f"Eval Precision: {eval_precision:.4f}, Eval Recall: {eval_recall:.4f}, Eval F1: {eval_f1:.4f}\n\n")

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'final_val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'alpha': {
            'values': [0.01, 0.1]
        },
        'activation_function': {
            'values': ['relu', 'tanh', 'sigmoid', 'linear']
        },
        'optimizer': {
            'values': ['sgd', 'bgd']
        },
        'architecture': {
            'values': [
                {'hidden_layers': 1, 'neurons_per_layer': [32]},
                {'hidden_layers': 2, 'neurons_per_layer': [64, 32]}
            ]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'epochs': {
            'value': 50
        }
    }
}

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

def train():
    wandb.init()
    config = wandb.config

    architecture = config.architecture
    hidden_layers = architecture['hidden_layers']
    neurons_per_layer = architecture['neurons_per_layer']

    mlp = MLP_Classifier_HPT(input_size=input_size, output_size=output_size)
    mlp.set_params(
        alpha=config.alpha,
        activation_function=config.activation_function,
        optimizer=config.optimizer,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    best_val_accuracy = 0  # Track the best validation accuracy

    for epoch in range(config.epochs):
        mlp.fit(X_train, y_train, X_eval, y_eval, early_stopping=True, patience=5)

        # Evaluate on validation set
        val_activations, _ = mlp.forward(X_eval)
        val_loss = mlp.compute_loss(val_activations[-1], y_eval)
        val_accuracy = mlp.compute_accuracy(val_activations[-1], y_eval)

        # Log the metrics to W&B
        wandb.log({
            'epoch': epoch, 
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            mlp.save_model('best_model.pkl')  # Save the best model using pickle

    # Log final metrics
    print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
    wandb.log({'final_val_loss': val_loss, 'final_val_accuracy': val_accuracy})

sweep_id = wandb.sweep(sweep_config, project="wine_quality_classification")
wandb.agent(sweep_id, function=train)

# Instantiate and train the model
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

# load model
mlp = MLP_Classifier(input_size, output_size)
mlp.load_model('best_model.pkl')
# mlp.set_params(alpha=0.01, activation_function='tanh', optimizer='sgd', hidden_layers= 3, neurons_per_layer=[128, 64, 32], batch_size=32, epochs=50)

mlp.fit(X_train, y_train, X_eval, y_eval, early_stopping=True, patience=5)


# Multi label

multilabel_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'final_val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'alpha': {
            'values': [0.01, 0.1]
        },
        'activation_function': {
            'values': ['relu', 'tanh', 'sigmoid']
        },
        'optimizer': {
            'values': ['sgd']
        },
        'architecture': {
            'values': [
                {'hidden_layers': 1, 'neurons_per_layer': [32]},
                {'hidden_layers': 2, 'neurons_per_layer': [64, 32]},
                {'hidden_layers': 3, 'neurons_per_layer': [128, 64, 32]}
            ]
        },
        'batch_size': {
            'values': [16]
        },
        'epochs': {
            'value': 50
        }
    }
}

input_size = X_train.shape[1]
output_size = y_train.shape[1]

def train():
    # Initialize W&B
    with wandb.init() as run:
        config = wandb.config

        # Get architecture configuration from W&B sweep
        architecture = config.architecture
        hidden_layers = architecture['hidden_layers']
        neurons_per_layer = architecture['neurons_per_layer']

        # Initialize the multi-label classifier
        mlp = MLP_MultiLabel_Classifier(input_size=input_size, output_size=output_size)
        mlp.set_params(
            alpha=config.alpha,
            activation_function=config.activation_function,
            optimizer=config.optimizer,
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            batch_size=config.batch_size,
            epochs=config.epochs
        )

        best_val_accuracy = 0  # Track the best validation accuracy

        for epoch in range(config.epochs):
            mlp.fit(X_train, y_train, X_eval, y_eval, early_stopping=True, patience=5)

            # Evaluate on the validation set
            val_activations, _ = mlp.forward(X_eval)
            val_loss = mlp.compute_loss(val_activations[-1], y_eval)
            val_accuracy = mlp.compute_accuracy(val_activations[-1], y_eval)

            # Compute additional classification metrics on validation set
            val_precision, val_recall, val_f1, val_hamming = mlp.compute_classification_metrics(y_eval, val_activations[-1])

            # Log metrics to W&B
            wandb.log({
                'epoch': epoch + 1, 
                'val_loss': val_loss, 
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1_score': val_f1,
                'val_hamming_loss': val_hamming
            })

            # Save the best model if validation accuracy improves
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                mlp.save_model('best_model_2.6.pkl')  # Save the best model using pickle

        # Log final validation metrics
        print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
        wandb.log({'final_val_loss': val_loss, 'final_val_accuracy': best_val_accuracy})

sweep_id = wandb.sweep(multilabel_sweep_config, project="advertisement_multi_label_classification")
wandb.agent(sweep_id, function=train, count=18)
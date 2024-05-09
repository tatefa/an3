""" 
Group A: Assignment No.3
Assignment Title: Write a Python Program using Perceptron Neural
Network to recognise even and odd numbers. Given numbers are in ASCII
form 0 to 9
"""
import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.dot(x)
        a = self.activation_fn(z)
        return a
    
    def train(self, X, labels):
        for _ in range(self.epochs):
            for i in range(len(labels)):
                x = np.insert(X[i], 0, 1)
                y_pred = self.predict(X[i])
                error = labels[i] - y_pred
                self.W = self.W + self.lr * error * x

def train_perceptron():
    # Training data
    X_train = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 9
    ]

    # Labels: 0 for even, 1 for odd
    y_train = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Create and train the perceptron
    perceptron = Perceptron(input_size=10)
    perceptron.train(X_train, y_train)
    print("Perceptron trained successfully!")
    return perceptron

def test_perceptron(perceptron):
    test_cases = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 9
    ]

    correct_predictions = 0
    for i, test_case in enumerate(test_cases):
        prediction = perceptron.predict(test_case)
        if prediction == 0 and i % 2 == 0:
            correct_predictions += 1
        elif prediction == 1 and i % 2 != 0:
            correct_predictions += 1
    for i, test_case in enumerate(test_cases):
        prediction = perceptron.predict(test_case)
        if prediction == 0:
            print(f"{test_cases[i]} which is number {i} is even")
        else:
            print(f"Number {i} is odd")
    
    accuracy = (correct_predictions / len(test_cases)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

def main():
    perceptron = None
    while True:
        print("\nMENU:")
        print("1. Train Perceptron")
        print("2. Test Perceptron")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            if perceptron is None:
                perceptron = train_perceptron()
            else:
                print("Perceptron already trained!")
        elif choice == '2':
            if perceptron is None:
                print("Please train the perceptron first!")
            else:
                print("Testing the perceptron:")
                test_perceptron(perceptron)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
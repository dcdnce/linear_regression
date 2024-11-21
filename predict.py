import sys
import json

class Predict:
    def __init__(self) -> None:
        self.theta0 = 0
        self.theta1 = 0
        self.prediction = 0
    
    def parse_coeffs_file(self, coeffs_file):
        try:
            with open(coeffs_file) as file:
                data = json.load(file)
            self.theta0 = data['theta0']
            self.theta1 = data['theta1']
        except Exception as e:
            print(f"Coeffs file: An error occurred: {e}")
            self.theta0 = 0
            self.theta1 = 0
        print(f"Coefficients for prediction :\n\ttheta0 = {self.theta0}\n\ttheta1 = {self.theta1}")

    def predict(self):
        self.prediction = self.theta0 + self.theta1 * 200000

def main():
    predict = Predict()
    predict.parse_coeffs_file("model_coeffs.json")
    predict.predict()
    print(f"Price predicted: {predict.prediction}")


if __name__ == "__main__":
    main()
import sys
import json

class Predict:
    def __init__(self) -> None:
        # self.theta0 = 0
        # self.theta1 = 0
        # self.price_min = 0
        # self.price_max = 0
        # self.mileage_min = 0
        # self.mileage_max = 0
        self.prediction = 0
    
    def parse_coeffs_file(self, coeffs_file):
        try:
            with open(coeffs_file) as file:
                data = json.load(file)
            self.theta0 = data['theta0']
            self.theta1 = data['theta1']
            self.mileage_min = data['mileage_min']
            self.mileage_max = data['mileage_max']
            self.price_min = data['price_min']
            self.price_max = data['price_max']
        except Exception as e:
            print(f"Coeffs file: An error occurred: {e}")
            self.theta0 = 0
            self.theta1 = 0
            self.price_min = 0
            self.price_max = 0
            self.mileage_min = 0
            self.mileage_max = 0

        print(f"Coefficients for prediction :\n\ttheta0 = {self.theta0}\n\ttheta1 = {self.theta1}")

    def predict(self, x):
        self.x_normalized = self.normalize(x, self.mileage_min, self.mileage_max)
        print(f"{x} normalized : {self.x_normalized}")
        self.prediction_normalized = self.theta0 + self.theta1 * self.x_normalized
        self.prediction = self.denormalize(self.prediction_normalized, self.price_min, self.price_max)
    
    def normalize(self, x, min, max):
        return ((x - min) / (max - min))
    
    def denormalize(self, x, min, max):
        return (min + x * (max - min))

def main():
    predict = Predict()
    predict.parse_coeffs_file("model_coeffs.json")
    predict.predict(380000)
    print(f"Prediction : \n\tNormal : {predict.prediction}\n\tNormalized : {predict.prediction_normalized}")


if __name__ == "__main__":
    main()
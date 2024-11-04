import csv
import matplotlib.pyplot as plt
import numpy as np

class Model:
	def __init__(self, data_file):
		self.mileage, self.price = self.parse_csv(data_file)
		self.mileage_normalized = self.normalize_array(self.mileage)
		self.price_normalized = self.normalize_array(self.price)
		self.theta0 = 1
		self.theta1 = 0
		self.m = len(self.mileage)

	def parse_csv(self, data_file):
		mileage = []
		price = []
		try:
			with open(data_file) as datafile:
				csvreader = csv.reader(datafile, delimiter=',')
				for row in csvreader:
					try:
						mileage_value = float(row[0])
						price_value = float(row[1])
						mileage.append(mileage_value)
						price.append(price_value)
					except ValueError:
						print(f"Non numeric value ignored : {row}")
		except Exception as e:
			print(f"An error occurred: {e}")
		
		return [mileage, price]

	def normalize_array(self, a):
		return [((x - min(a)) / (max(a) - min(a)) )for x in a]

	def compute_MSE(self):
		mean_squared_error = 0
		for i in range(self.m):
			mean_squared_error += ((self.theta0 + self.theta1 * self.mileage_normalized[i]) - self.price_normalized[i]) ** 2
		return (mean_squared_error / float(self.m)) # same as * (1 / m)

	def gradient_descent(self, iteration_number):
		t0 = 0
		t1 = 0
		X = self.mileage_normalized
		Y = self.price_normalized

		for j in range(iteration_number):
			derivative_t0 = 0
			derivative_t1 = 0
			for i in range(self.m):
				derivative_t0 += ((t0 + t1 * X[i]) - Y[i])
				derivative_t1 += ((t0 + t1 * X[i]) - Y[i]) * X[i]
			t0 = t0 - (0.1) * (derivative_t0 * (2 / self.m))
			t1 = t1 - (0.1) * (derivative_t1 * (2 / self.m))

		self.theta0 = t0
		self.theta1 = t1

	def plot_results(self):
		plt.close('all')
		plt.figure(figsize=(10, 6))

		# Data entry
		plt.scatter(self.mileage_normalized, self.price_normalized, color='blue', label='Entry data')
		# Prediction line
		x = np.linspace(min(self.mileage_normalized), max(self.mileage_normalized))
		y = self.theta0 + self.theta1 * x
		plt.plot(x, y, 'r-', label='Prediction Line')
		# Shifting apparent scale
		plt.xticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(i * (max(self.mileage) - min(self.mileage)) + min(self.mileage))}" for i in np.linspace(0, 1, 5)])
		plt.yticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(i * (max(self.price) - min(self.price)) + min(self.price))}" for i in np.linspace(0, 1, 5)])

		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.legend()
		plt.grid(True)
		plt.show(block=True)


def main():
	model = Model("data.csv")	
	print(f"Mean squared error before regression: {model.compute_MSE()}")
	model.gradient_descent(1000)
	print(f"Gradient descent: theta0 = {model.theta0}, theta1 = {model.theta1}")
	print(f"Mean squared error after regression: {model.compute_MSE()}")
	model.plot_results()
	# model.save_coefficients()

if __name__ == "__main__":
	main()
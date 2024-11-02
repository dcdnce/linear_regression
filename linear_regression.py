import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
	mileage, price = parse_csv()
	
	mileage_normalized = normalize_array(mileage)
	price_normalized = normalize_array(price)

	print(f"Mean squared error before regression: {compute_MSE(mileage_normalized, price_normalized, 0, 0)}")
	theta0, theta1 = gradient_descent(mileage_normalized, price_normalized, 1000)
	print(f"Gradient descent: theta0 = {theta0}, theta1 = {theta1}")
	print(f"Mean squared error after regression: {compute_MSE(mileage_normalized, price_normalized, theta0, theta1)}")

	plot_results(mileage_normalized, price_normalized, mileage, price, theta0, theta1)

def parse_csv():
	mileage = []
	price = []
	try:
		with open("data.csv") as datafile:
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

def normalize_array(a):
	return [((x - min(a)) / (max(a) - min(a)) )for x in a]

def gradient_descent(X, Y, iteration_number):
	theta0 = 0
	theta1 = 0

	for j in range(iteration_number):
		m = len(X)
		derivative_theta0 = 0
		derivative_theta1 = 0
		for i in range(m):
			derivative_theta0 += ((theta0 + theta1 * X[i]) - Y[i])
			derivative_theta1 += ((theta0 + theta1 * X[i]) - Y[i]) * X[i]
		theta0 = theta0 - (0.1) * (derivative_theta0 * (2 / m))
		theta1 = theta1 - (0.1) * (derivative_theta1 * (2 / m))

	return (theta0, theta1)

def compute_MSE(X, Y, theta0, theta1):
	mean_squared_error = 0
	for i in range(len(X)):
		mean_squared_error += ((theta0 + theta1 * X[i]) - Y[i]) ** 2
	return (mean_squared_error / float(len(X))) # same as * (1 / m)

def plot_results(mileage_normalized, price_normalized, mileage, price, theta0, theta1):
	plt.close('all')
	plt.figure(figsize=(10, 6))

	# Data entry
	plt.scatter(mileage_normalized, price_normalized, color='blue', label='Entry data')
	# Prediction line
	x = np.linspace(min(mileage_normalized), max(mileage_normalized))
	y = theta0 + theta1 * x
	plt.plot(x, y, 'r-', label='Prediction Line')
	# Shifting apparent scale
	plt.xticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(min(mileage) + (max(mileage) - min(mileage)) * i)}" for i in np.linspace(0, 1, 5)])
	plt.yticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(min(price) + (max(price) - min(price)) * i)}" for i in np.linspace(0, 1, 5)])

	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.legend()
	plt.grid(True)
	plt.show(block=True)

if __name__ == "__main__":
	main()
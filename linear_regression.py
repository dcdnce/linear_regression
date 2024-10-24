import csv

def main():
	mileage, price = parse_csv()
	for i in range(len(mileage)):
		print(f"{mileage[i]} : {price[i]}")

	print(f"Mean squared error before regression: {compute_MSE(mileage, price, 0, 0)}")
	theta0, theta1 = gradient_descent(mileage, price, 1000)
	print(f"Mean squared error after regression: {compute_MSE(mileage, price, theta0, theta1)}")

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

def gradient_descent(X, Y, iteration_number):
	theta0 = 0
	theta1 = 0

	for j in range(iteration_number):
		print(f"Iteration {j}: theta0 = {theta0}, theta1 = {theta1}")
		m = len(X)
		derivative_theta0 = 0
		derivative_theta1 = 0
		for i in range(m):
			# Partial derivatives :
			#	- (θ0 + θ1 * X[i]) = estimatePrice, the linear function
			#	- Derived from loss function
			derivative_theta0 += ((theta0 + theta1 * X[i]) - Y[i])
			derivative_theta1 += ((theta0 + theta1 * X[i]) - Y[i]) * X[i]
		theta0 = theta0 - (0.1) * (derivative_theta0 * (2 / m))
		theta1 = theta0 - (0.1) * (derivative_theta1 * (2 / m))

	return (theta0, theta1)

def compute_MSE(X, Y, theta0, theta1):
	mean_squared_error = 0
	for i in range(len(X)):
		mean_squared_error += ((theta0 + theta1 * X[i]) - Y[i]) ** 2
	return (mean_squared_error / float(len(X))) # same as * (1 / m)

if __name__ == "__main__":
	main()
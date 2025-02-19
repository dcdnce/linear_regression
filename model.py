import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import json


class Model:
	def __init__(self):
		self.mileage = 0
		self.price = 0
		self.mileage_normalized = 0
		self.price_normalized = 0
		self.theta0 = 0 
		self.theta1 = 0
		self.m = 0
		self.figure = None
		self.stop_training = False
		self.init_plots()

	def on_close(self, event):
		self.stop_training = True

	def parse_data(self, data_file):
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
						print(f"Data file: Error value ignored : {row}")
		except Exception as e:
			print(f"An error occurred: {e}\nExiting")
			sys.exit(1)
		
		print(f"Data file: mileage (min,max) : {min(mileage)}, {max(mileage)}")
		print(f"Data file: price (min,max) : {min(price)}, {max(price)}")

		self.mileage = mileage
		self.price = price 
		self.mileage_normalized = self.normalize_array(mileage)
		self.price_normalized = self.normalize_array(price)
		self.m = len(mileage)

	def normalize_array(self, a):
		return [((x - min(a)) / (max(a) - min(a)) )for x in a]
	
	def init_plots(self):
		plt.ion()
		self.figure = plt.figure(figsize=(12, 4))
		self.b_values = []
		self.m_values = []
		self.error_values = []

	def compute_MSE(self, t0, t1):
		mean_squared_error = 0
		for i in range(self.m):
			mean_squared_error += ((t0 + t1 * self.mileage_normalized[i]) - self.price_normalized[i]) ** 2
		return (mean_squared_error / float(self.m)) # same as * (1 / m)

	def gradient_descent(self, iteration_number, learning_rate):
		t0 = 0
		t1 = 0
		X = self.mileage_normalized
		Y = self.price_normalized
		previous_mse = float('inf')
		patience_counter = 0 # tracking consecutive insignificant improvements

		for j in range(iteration_number+1):
			derivative_t0 = 0
			derivative_t1 = 0
			for i in range(self.m):
				derivative_t0 += ((t0 + t1 * X[i]) - Y[i])
				derivative_t1 += ((t0 + t1 * X[i]) - Y[i]) * X[i]

			t0 = t0 - (learning_rate) * (derivative_t0 * (2 / self.m))
			t1 = t1 - (learning_rate) * (derivative_t1 * (2 / self.m))

			# Compute MSE and check for early stopping
			current_mse = self.compute_MSE(t0, t1)
			if abs(previous_mse - current_mse) < 1e-10:
				patience_counter += 1
				if patience_counter >= 5:
					print(f"Early stopping at iteration {j} with MSE: {current_mse}, tolerance : 1e-10, patience_counter : 5")
					break
			else:
				patience_counter = 0
			previous_mse = current_mse

			# Check user stop
			if self.stop_training:
				print(f"User stopping at iteration {j}")
				break

			# Live plotting
			if ((j % 10 == 0 or j < 10)):
				self.plot_results(t0, t1, j)

		self.theta0 = t0
		self.theta1 = t1
	
	def plot_results(self, t0, t1, iteration):
		plt.clf()
		fig = plt.gcf()
		fig.stop_training = False 
		fig.canvas.mpl_connect('close_event', self.on_close)

		# Subplot #1 :
		plt.subplot(1, 2, 1)
		plt.title(f"m = {t1:.4f} \n b = {t0:.4f}")
		# Data entry
		plt.scatter(self.mileage_normalized, self.price_normalized, color='blue', label='Entry data')
		# Prediction line
		x = np.linspace(min(self.mileage_normalized), max(self.mileage_normalized))
		y = t0 + t1 * x
		plt.plot(x, y, 'r-', label='Prediction Line')
		# Shifting apparent scale
		plt.xticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(i * (max(self.mileage) - min(self.mileage)) + min(self.mileage))}" for i in np.linspace(0, 1, 5)])
		plt.yticks(ticks=np.linspace(0, 1, 5), labels=[f"{int(i * (max(self.price) - min(self.price)) + min(self.price))}" for i in np.linspace(0, 1, 5)])
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.grid(True)

		# Subplot #2 :
		ax = plt.subplot(1, 2, 2, projection='3d')
		self.b_values.append(t0)
		self.m_values.append(t1)
		self.error_values.append(self.compute_MSE(t0, t1))
		# ax.set_title(f"Error: {self.error_values[-1]:.5f}")
		ax.set_title(f"Error: {self.error_values[-1]}")
		# Errors display through the iterations
		ax.scatter(self.b_values, self.m_values, self.error_values, color='red', s=10, edgecolors='black', alpha=1.0)
		# Error surface
		X = np.linspace(min(self.b_values), max(self.b_values))
		Y = np.linspace(min(self.m_values), max(self.m_values))
		X, Y = np.meshgrid(X, Y)
		Z = self.compute_MSE(X, Y)
		Z = np.clip(Z, None, max(self.error_values))
		ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
		ax.set_xlabel('b')
		ax.set_ylabel('m')
		ax.set_zlabel('Error')
		ax.grid(True)

		plt.suptitle("Iteration " + str(iteration))
		plt.pause(0.1)
	
	def save_coefficients(self):
		with open('model_coeffs.json', 'w') as file:
			json.dump(
				{
					"theta0": self.theta0, 
					"theta1": self.theta1,
					"price_min": min(self.price),
					"price_max": max(self.price),
					"mileage_min": min(self.mileage),
					"mileage_max": max(self.mileage),
				}, 
				file)

def main():
	data_file_name, iterations, learning_rate = user_input()

	model = Model()
	model.parse_data(data_file_name)
	print(f"\nMean squared error before regression: {model.compute_MSE(model.theta0, model.theta1)}")
	model.gradient_descent(iterations, learning_rate)
	print(f"Gradient descent: theta0 (b) = {model.theta0}, theta1 (m) = {model.theta1}")
	print(f"Mean squared error after regression: {model.compute_MSE(model.theta0, model.theta1)}")
	model.save_coefficients()
	plt.ioff()
	plt.show()

def user_input():
	if len(sys.argv) == 1:
		print("Usage: python3 model.py -f <file> -i <iterations> -l <learning_rate>\n")
	data_file_name = "data.csv"
	iterations = 500
	learning_rate = 0.1
	args = sys.argv[1:]
	try:
		for i in range(0, len(args), 2):
			if args[i] == "-f":
				data_file_name = args[i + 1]
			elif args[i] == "-i":
				iterations = int(args[i + 1])
			elif args[i] == "-l":
				learning_rate = float(args[i + 1])
			else:
				print(f"\nUnknown argument: {args[i]}")
				print("Usage: python3 model.py -f <file> -i <iterations> -l <learning_rate>\n")
	except Exception as e:
		print(f"An error occurred: {e}\nExiting")
		sys.exit(1)
	print(f"Data file: {data_file_name}\niterations : {iterations}\nlearning rate: {learning_rate}")
	return data_file_name, iterations, learning_rate


if __name__ == "__main__":
	main()
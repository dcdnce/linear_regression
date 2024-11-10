import csv
import matplotlib.pyplot as plt
import numpy as np
import json

class Model:
	def __init__(self, data_file):
		self.mileage, self.price = self.parse_csv(data_file)
		self.mileage_normalized = self.normalize_array(self.mileage)
		self.price_normalized = self.normalize_array(self.price)
		self.theta0 = 0 
		self.theta1 = 0
		self.m = len(self.mileage)
		self.figure = None
		self.init_plots()

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
	
	def init_plots(self):
		plt.ion()
		self.figure = plt.figure(figsize=(16, 6))
		self.t0_plots = []
		self.t1_plots = []
		self.error_plots = []

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

		for j in range(iteration_number+1):
			derivative_t0 = 0
			derivative_t1 = 0
			for i in range(self.m):
				derivative_t0 += ((t0 + t1 * X[i]) - Y[i])
				derivative_t1 += ((t0 + t1 * X[i]) - Y[i]) * X[i]
			t0 = t0 - (learning_rate) * (derivative_t0 * (2 / self.m))
			t1 = t1 - (learning_rate) * (derivative_t1 * (2 / self.m))
			if (j % 10 == 0 or j < 10 ):
				self.plot_results(t0, t1, j)

		self.theta0 = t0
		self.theta1 = t1

	def plot_results(self, t0, t1, iteration):
		plt.clf()

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
		X = np.arange(0, 1.2, 0.1) #theta0
		Y = np.arange(-1, 0.2, 0.1) #theta1
		X, Y = np.meshgrid(X, Y)
		Z = self.compute_MSE(X, Y)
		ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
		self.t0_plots.append(t0)
		self.t1_plots.append(t1)
		self.error_plots.append(self.compute_MSE(t0, t1))
		ax.scatter(self.t0_plots, self.t1_plots, self.error_plots, color='red', s=10, edgecolors='black', alpha=1.0)
		ax.set_xlabel('b')
		ax.set_ylabel('m')
		ax.set_zlabel('Error')
		ax.set_title('Error Surface')
		ax.grid(True)

		# ax.set_xlim([0, 1])  # Limite pour theta0 (b)
		# ax.set_ylim([-1, 0.2])  # Limite pour theta1 (m)
		# ax.set_zlim([0, 0.5])  # Limite pour l'erreur MSE (z)


		plt.suptitle("Iteration " + str(iteration))
		plt.pause(0.1)
	
	def save_coefficients(self):
		with open('model_coeffs.json', 'w') as file:
			json.dump({"theta0": self.theta0, "theta1": self.theta1}, file)

def main():
	model = Model("data3.csv")	
	print(f"Mean squared error before regression: {model.compute_MSE(model.theta0, model.theta1)}")
	model.gradient_descent(300, 0.1)
	print(f"Gradient descent: theta0 (b) = {model.theta0}, theta1 (m) = {model.theta1}")
	print(f"Mean squared error after regression: {model.compute_MSE(model.theta0, model.theta1)}")
	# model.plot_results()
	model.save_coefficients()
	plt.ioff()
	plt.show()

if __name__ == "__main__":
	main()
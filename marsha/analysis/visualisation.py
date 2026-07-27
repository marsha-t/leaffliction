import matplotlib.pyplot as plt

def plot_bar_chart(distribution):
	"""
	Plots image counts by class in bar chart

	Args:
		distribution (dict): Dictionary of {directory-name: image count}
	"""
	plt.figure(figsize=(8,5))

	plt.bar(distribution.keys(), distribution.values())

	plt.title('Dataset Distribution by Counts')
	plt.xlabel('Class')
	plt.xticks(rotation=45)
	plt.ylabel('Number of Images')

	plt.tight_layout()

	plt.show()

def plot_pie_chart(distribution):
	"""
	Plot class distribution in pie chart
	
	Args:
		distribution (dict): Dictionary of {directory-name: image count}
	"""
	plt.figure(figsize=(6,6))

	plt.pie(distribution.values(), labels=distribution.keys(), autopct="%.1f", startangle=90)

	plt.title('Dataset Distribution (%)')

	plt.tight_layout()

	plt.show()
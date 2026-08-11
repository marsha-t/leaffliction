import matplotlib.pyplot as plt


def plot_charts(distribution):
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
	plot_pie_chart(ax1, distribution)
	plot_bar_chart(ax2, distribution)

	fig.suptitle('Dataset Distribution')

	plt.tight_layout()
	plt.show()


def plot_pie_chart(ax, distribution):
	"""
	Plot class distribution in pie chart
	
	Args:
	    ax (matplotlib.axes.Axes): Axes to draw on.
		distribution (dict): Dictionary of {directory-name: image count}
	"""
	ax.pie(distribution.values(), labels=distribution.keys(), autopct="%.1f%%", startangle=90, colors=["orange","pink","yellow","red"])

	ax.set_title('Proportions (%)')
	ax.axis("equal")


def plot_bar_chart(ax, distribution):
	"""
	Plots image counts by class in bar chart

	Args:
	    ax (matplotlib.axes.Axes): Axes to draw on.
		distribution (dict): Dictionary of {directory-name: image count}
	"""
	ax.bar(distribution.keys(), distribution.values(), color=["orange","pink","yellow","red"])

	ax.set_title('Counts')
	ax.set_xlabel('Class')
	ax.tick_params(axis='x', labelrotation=45)
	ax.set_ylabel('Number of Images')


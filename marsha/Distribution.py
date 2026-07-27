import argparse

from analysis.validation import validate_directory
from analysis.analysis import analyse_directory
from analysis.visualisation import plot_charts

def parse_args():
	"""
	Parse command-line arguments

	Returns:
		argparse.Namespace: parsed command-line arguments
	"""
	parser = argparse.ArgumentParser(description='Analyse data')
	parser.add_argument('directory', help='Dataset directory')
	args = parser.parse_args()
	return args
	
def main():
	"""
	Execute dataset analysis pipeline:
	- Parse command line arguments
	- Validate dataset directory
	- Analyse image distribution
	- Display bar and pie chart
	"""
	args = parse_args()

	warnings = validate_directory(args.directory)
	for warning in warnings:
		print(f'Warning: {warning}')

	distribution = analyse_directory(args.directory)

	plot_charts(distribution)

if __name__ == "__main__":
	main()
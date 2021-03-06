import pandas as pd


class Validator(object):
	"""Validates a given output with a manually generated test case.
	"""

	def __init__(self, pathValid="src/hand_calculated.csv", pathCalculated="src/sample_output.csv"):
		self.validDF = pd.read_csv(pathValid, sep=',')
		self.calcDF = pd.read_csv(pathCalculated)
		print(self.calcDF)

	def analyze(self):
		"""Analyzes the accuracy of the data."""
		for i in range(0, len(self.calcDF.columns), 1):
			pass




if __name__ == "__main__":
	validator = Validator()
	validator.analyze()

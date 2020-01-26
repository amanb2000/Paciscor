class Validator(object):
	"""Validates a given output with a manually generated test case.
	"""

	def __init__(self, _validData):
		self,validData = _validData

	def validate_row(self, exclude=[]):
		"""Checks to see if the row is completed while not accounting for the exclude items."""

import processing


class KeyChars(object):
	"""Contains attributes of key characteristics extracted from
	analyzing string objects."""

	def __init__(self):
		self.flyer_name = None  # week_page
		self.product_name = None  # from product inventory
		self.unit_promo_price = None  # price per unit
		self.uom = None  # unit of measurement
		self.least_unit_for_promo = 1  # minimum quantity before promotion applies
		self.save_per_unit = None  # savings per unit
		self.discount = None  # discount from original price
		self.organic = 0  # boolean indicating organic (1) or not organic (0)

	# def setFlyerName(_flyer_name):
	# 	"""Set the name of flyer."""
	# 	if not isinstance(_flyer_name, str):
	# 		raise TypeError("flyer name should be of type string")
	# 	self.flyer_name = _flyer_name

	# def setProductName(_product_name):
	# 	"""Set the name of product."""
	# 	if not isinstance(_product_name, str):
	# 		raise TypeError("product name should be of type string")
	# 	self.product_name = _product_name

	# def setUnitPromoPrice(_upp):
	# 	"""Set the unit promo price."""
	# 	if not isinstance(_upp, float):
	# 		raise TypeError("unit promo price should be of type float")
	# 	self.unit_promo_price = _upp

	# def setUOM(_uom):
	# 	"""Set the uom."""
	# 	if not isinstance(_upp, string):
	# 		raise TypeError("uom should be of type string")
	# 	self.uom = _uom

	# def setLeastUnitForPromo(_lufp):
	# 	"""Set the least unit for promo."""
	# 	if not isinstance(_lufp, int):
	# 		raise TypeError("least unit for promo should be of type int")
	# 	elif _lufp < 1:
	# 		raise ValueError("least unit for promo should be greater than or equal to 1")
	# 	self.least_unit_for_promo = _lufp

	# def setSavePerUnit(_spu):
	# 	"""Set the save per unit"""
	# 	self.save_per_unit = _spu

	# def setDiscount(_discount):
	# 	"""Set the discount"""
	# 	self.discount = _discount

	# def setOrganic(_organic):
	# 	"""Set organic boolean"""
	# 	self.organic = _organic


class NPLAnalyzer(object):
	"""Take in blocks (tuples) generated by OCR processing and analyzes
	them for key characteristics."""

	def __init__(self, _productInventory, _metricInventory):
		self.productInventory = _productInventory
		self.metricInventory = _metricInventory
		self.store = []  # stores a list of KeyChars for the most recent analyze call

	def analyze(self, block):
		"""Analyzes a block and generates key characteristics from
		each stringObject (dict) within."""
		if not isinstance(block, tuple):
			raise TypeError("segment should be of type tuple")
		# loop through the block and generate KeyChars for each segment
		for segment in block:
			# stores information
			keyChars = KeyChars()
			# extract phrase from segment in lower case
			phrase = segment["text"].lower()
			# check the phrase for information
			checkProductName(phrase)

	def checkFlyerNames(self, phrase):
		"""Check the phrase for flyer name"""
		pass

	def checkProductName(self, phrase):
		"""Check the phrase for product name"""

		pass

if __name__ == "__main__":
	analyzer = NPLAnalyzer(processing.generateProductInventory(), processing.generateUnitsInventory())
	pass
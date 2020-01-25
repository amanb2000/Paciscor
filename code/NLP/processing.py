import pandas as pd

def generateProductInventory(path="src/product_dictionary.csv"):
	"""Outputs a Dataframe structure, a column vector with names of possible products."""
	df = pd.read_csv(path)
	for i in range(df["product_name"].count()):
		df["product_name"][i] = df["product_name"][i].lower()
	return df


def generateUnitsInventory(path="src/units_dictionary.csv"):
	"""Outputs a Dataframe structure, a column vector with names of possible units of measurement."""
	return pd.read_csv(path)

if __name__ == "__main__":
	print(generateProductInventory())
	print(generateUnitsInventory())

import pandas as pd 
from PIL import Image

df = pd.read_csv('files.csv')

for index, row in df.iterrows():
    name = row['NAMES']
    im1 = Image.open(name)
    dot_index = name.index('.')

    name2 = name[:dot_index]
    name2 += '.png'

    print(name2)

    im1.save(name2)

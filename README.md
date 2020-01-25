# Paciscor
*Daisy Intelligence Hackathon 2020*

# Framing

## Given

* Set of coupon images (multiple ad blocks in each that must be segmented and processed).
* CSV of potential product names for each ad block.
* CSV of potential units of measure.

## Needed
1. `flyer_name`: Name of the file we got the rest of the data from.
2. `product_name`: Name of the product listed in the given ad block.
3. `unit_promo_price`: Promotion price for each unit.
4. `uom`: Unit of measurement of the product.
5. `least_unit_for_promo`: Least amount of the product the customer has to buy in order to use the promo.
6. `save_per_unit`: Amount of money saved per unit rounded off at 2 decimal places.
7. `discount`: Discount on the original price rounded off to 2 decimal places. (percentage in decimal)
8. `organic`: 0 or 1 binary values indicate if the product is organic or not described in the flyer.

# The Grand Unified Pipeline Objectives

## Pre-Processing
* Consider removing banners (if no information exists within them).
* Consider cropping the white spaces from the images (decrease compute time).

## Optical Character Recognition (OCR)
__The data structure produced should contain multiple sub-strings and attributes of those substrings.__

It should be specified to typed font faces.

* Ordering should matter (high -> low, left -> right).
* Specify the color {red/green/blue/yellow}.
* Specify the text size (px).
* Specify the font.
* Specify font styling {bold, italics}.

**Output in `tuple`** (immutable, ordered).

## Natural Language Processing
Regex... fuzzy matches...
* Matching
* Regex for (`%`, `/`, `$`)

## Image Recognition Libraries
Likely unnecessary, later stage project.


# Validation
## Creation of the Set
One of us must go through `>= 5` full flyers and create the desired output by hand.

## Autograding
To make our optimization process more smooth, we shall create an `autograder` that we can easily feed our algorithm into. It will then give us a cost function result.

# Our Process
## 1: Minimum Viable Product (MVP)
### A: Segmentation (Adam)
Goal: Create a script that takes in an image and outputs a list of coordinate pairs that corresponds to the top left corner and the bottom right corner of each ad block.

### B: OCR (Aman)
Assumptions: Each *ad block* is provided as a rectangle given the top right and the bottom left corner.
Goal: Create a script that takes in the list of coordinate pairs from `(A)` and outputs the raw text from that section
from top to bottom.

Extra points for differentiating different font sections and all that jazz.

### C: Natural Language Processing (Divy)
Input: A tuple of text objects which contain the OCR specifications.
Goal: Create a `regular expression` (regex) based system for taking the raw text of a given section and determining the following characteristics:

1. `flyer_name`: Name of the file we got the rest of the data from.
2. `product_name`: Name of the product listed in the given ad block.
3. `unit_promo_price`: Promotion price for each unit.
4. `uom`: Unit of measurement of the product.
5. `least_unit_for_promo`: Least amount of the product the customer has to buy in order to use the promo.
6. `save_per_unit`: Amount of money saved per unit rounded off at 2 decimal places.
7. `discount`: Discount on the original price rounded off to 2 decimal places.
8. `organic`: 0 or 1 binary values indicate if the product is organic or not described in the flyer.

### D: Validation
Goal: (1) Create a validation set comprised of `>= 5` flyers and their corresponding CSV outputs. (2) Create a script that takes a full algorithm/pipeline as input and then grades the algorithm on the aforementioned validation set.

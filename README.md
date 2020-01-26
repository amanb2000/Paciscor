# Paciscor

Daisy Intelligence Hackathon 2020

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

# Optimization Strategy

## 1: Back-Trace the Algorithm's Mistakes on the Validation Set

* If the error was on the NLP, use some smart regex to account for that.

## 2: Optimize Relevant Hyperparameters using a Juiced Up Computer (Google or Otherwise)

* Image pre-processing optimized based on final accuracy on training set.
* OCR hyperparameter optimization.
* etc.

## 3: Contact Experts

* Jad Ghalini.
* Robert.
* Hersh.
* Reddit.

## 4: Use the Help to Expand our Validation Set

This directly helps our final results in that we can have 100% accurate rows in our final product.
We can also use it to make sure we are not overfitting and to help with the backtracking.

# Questions

* See if we can actually access the file names - if we can, then we can do away with the red bars that contain the information about the date.
* Are we formally disallowed from going through the images by hand for our final CSV? Would/how would they know?
* Can we use external help for annotation?
* Can we talk to experts outside of the team?

# Final Sprint

## Things that Must Be Done

### OCR Pipeline

* [ ] Differentiate sections based on `size` and if it's grey.
  * [ ] Fix the `get_percent_coloured` function.
  * [ ] Fix the `get_avg_coloured_pixel` function.
* [ ] Integrate segmentation input.

### Segmentation

* [ ] Draw rectangles on plots
* [ ] Adjust weightings to perfect targetting

### NLP

* [ ] .

### Validation

* [ ] Create autograder.
* [ ] Generate hyperparameters (time permitting).
* [ ] Optimize hyperparameters (time permitting). 

### Put Together

* [ ] Integrate Segmentation
    * Interface inputs
    * Interface outputs
* [ ] Integrate OCR
    * Interface inputs
    * Interface outputs
* [ ] Integrate NLP
    * Interface inputs
    * Interface outputs
* [ ] Format output CSV (round to 2 decimals)
* [ ] Build runner for start to finish integration

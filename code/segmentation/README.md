# Segmentation

## Fuzzy-C Means Segmentation (FCM)

The Fuzzy-C Means algorithm will be applied in accordance with the findings of this paper: 
https://pdfs.semanticscholar.org/ead6/923824191bc23a378b7a67c5ee8200c1fabd.pdf


The algorithm was found to perform better than K-means and improved K-means in our general cases. This will be investigated in implementation as the first option.

## Algorithmic Steps

Goal: Take in an image of a flyer and return the top left and bottom right corners of each *ad block* segment.

1. Intake image
2. Remove white background
3. Apply C-Fuzzy
4. Determine top left and bottom right corner of segments
5. Return the above values

## Progression

1. Successfully draw a box around one ad block
2. Return ad block's position in pixels
3. Refine to work for the majority of blocks in a flyer page
4. Add final functionality with all ad blocks (overlapping segments)
5. Validate with many flyers
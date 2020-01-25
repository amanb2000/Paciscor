# Optical Character Recognition

## Tasks

* [ ] Get initial `Tesseract` OCR working in python and producing text ([google repo](https://github.com/tesseract-ocr/tesseract), [examples](https://github.com/nikhilkumarsingh/tesseract-python))
* [ ] Find bounding box information (if possible) from `Tesseract` output.
* [ ] Differentiate sections based on font size.
* [ ] Add font size to export.
* [ ] Add font color (based on bounding boxes) to export. 

## Research on OCR Algorithms

### Requirements

1. Must work well for `typed` font faces.
2. Should regress `size` of text.
3. Should classify `color`.
4. Should classify `font`.
5. Should classify `font-style`.

### Alternatives

* **Google Tesseract OCR: S Class**
  * [Repository](https://github.com/tesseract-ocr/tesseract)
  * Utilizes Neural Net LSTM model.
  * Output formats: 
    * Plain text.
	* HTML.
	* TSV. 
  * [Methods of quality improvement](https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality)
  * Does not seem to have the ability to tell the color or size or styling.
  * Use in Python: [This repository](https://github.com/nikhilkumarsingh/tesseract-python) has some simple examples.
  * **Overall** Fulfills requirement #1, will need to use other models for the rest of the requirements.

* **Manual Font Sizing: B Class**
  * If bounding box information for lines is available, we can leverage that to create a `font-size` regressor.

* **Manual Color Detection: A Class**
  * No matter what, we will be getting the coordinates for different text pieces.
  * If we look for the most common non-white pixel colors, we can deduce the font color relatively easily. 


## Input

1. `path`: Path to .jpg file we are analyzing.
2. `coords`: Tuple of tuples. Each sub-tuple contains coordinates of the top left and of the bottom right corner of each **ad block**.

### Example

```python
path = '../../src/flyers/week_1_page_1.jpg'
coords = (
    (
        (188, 23),
        (19, 233)
    ),
    ...
)
```

## Output

```python
`output` = `tuple`[
    `dict`{
        `text`: "$5 Off!",
        `color`: "red", # red, yellow, blue, green, etc.
        `size`: 15, # integer
        `font`: "..." # remains to be seen as I do more research. For now, don't worry too much about this.
        `styling`: "..." # remains to be seen as I do more research.
    },
    `dict`{
        ...
    }
]
```

# Optical Character Recognition

## Tasks

* [x] Create personal virtual environment.
* [x] Get initial `Tesseract` OCR working in python and producing text ([google repo](https://github.com/tesseract-ocr/tesseract), [examples](https://github.com/nikhilkumarsingh/tesseract-python), [tutorial](https://nanonets.com/blog/ocr-with-tesseract/#introduction))
* [x] Create a barebones OCR processing function to get started.
* [x] Find bounding box information (if possible) within the OCR processing function.
* [x] Implement pre-processing pipeline to optimize results.
* [x] Differentiate sections based on `color`.
* [ ] Differentiate sections based on `size` and if it's grey.
  * [x] Create histogram of sizes to determine which is what.
* [ ] Create JPEG => PNG conversion script.
* [ ] Create an OCR-based segmentation algorithm:
  * [x] Create paramaterized algorithm that assigns a score to each (x, y) coordinate pair based on how many OCR-recornized words there are in an `n` pixel radius. It should take `m` long steps.
  * [ ] Threshold filter.
  * [ ] Determine {upper left} and {bottom right} coordinates.

## Notes on Implementation


### Sizing Diagnostics for Line Heights

* < 16 : Green
* [16:30] : Red
* [30:38] : Blue
* [38:45] : Purple
* [45:90] : Orange
* > 90 : Pastel

### Testing and Command Line Notes

* The model performs ~10x better when the format is converted to PNG. This seems to be an issue with 
the encoding of DPI. The following errors occur when JPG format is used:

```JSON
Warning: Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 350
Error in boxClipToRectangle: box outside rectangle
Error in pixScanForForeground: invalid box
Error in boxClipToRectangle: box outside rectangle
Error in pixScanForForeground: invalid box
```

* The model has trouble recognizing things like 3.99 from representationw that have the cents as superscripts. At times they are transcribed as spaces or no spaces.
* Cropping the model search space seems to help slightly.
* Specifying to English and file output in command line: `tesseract image_path text_result.txt -l eng`
* `psm` argument: segmentation mode.
* `oem` argument: OCR engine mode (0-4), legacy engine => LSTM + Neural Net => Default based on availability.

#### Understanding PSM and OEM

**`--oem N`** (OCR Engine Models) where `N` can be:

* 0 = Original Tesseract only.
* 1 = Neural nets LSTM only.
* 2 = Tesseract + LSTM.
* 3 = Default, based on what is available.

**`--psm N`** (page segmentation models) where `N` can be:

* 0 = Orientation and script detection (OSD) only.
* 1 = Automatic page segmentation with OSD.
* 2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
* 3 = Fully automatic page segmentation, but no OSD. (Default)
* 4 = Assume a single column of text of variable sizes.
* 5 = Assume a single uniform block of vertically aligned text.
* 6 = Assume a single uniform block of text.
* 7 = Treat the image as a single text line.
* 8 = Treat the image as a single word.
* 9 = Treat the image as a single word in a circle.
* 10 = Treat the image as a single character.
* 11 = Sparse text. Find as much text as possible in no particular order.
* 12 = Sparse text with OSD.
* 13 = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

Nota bene: `script detection` is basically just `language detection`.


### Pytesseract and OpenCV

* `/py-testing/tesseract-test.py` contains a bare-bones string output.
  * Does not, however, have any bounding box outputs going on.
* Getting bounding boxes: see `bounding-box-1.py` and `bounding-box-2.py`.
* Still having trouble with getting the `$3^{99}` red text going. 
  * TODO: Try pre-processing techniques on these...
  * Could use Adam's algorithm(s) to localize those as they are relatively easy to recognize AFTER the bounding box is created.
  * With custom config and correct bounding box for search, we can get **something**:

  ```python
  custom_config = r'--oem 3 --psm 6'
  ```

### Tesseract (and other) Enhancements

* Tesseract 4.00: Neural Network-Based
  * We could fine-tune the model but wrangling external training data would be hard.
  * The model already has weights for a pretty hefty set of Latin-based language characters.
  * It would appear that we simply must be extremely careful with our `psm` and `oem` arguments.
  * TODO: Find out exactly what every single `psm` and `oem` mode does.
  * TODO: We should definitely check out [these models](https://github.com/tesseract-ocr/tessdata_best) that are apparently the best LSTM models (pending `psm` and `oem` mode research).
    * [These models](https://github.com/tesseract-ocr/tessdata_fast) are the fastest.
    * See [this website](https://www.endpoint.com/blog/2018/07/09/training-tesseract-models-from-scratch) if further reference is needed.

* [Nanonets](https://nanonets.com/pricing/)
  * Time permitting, we may be able to leverage this service to segment our images.
  * Their free developer package enables us to use their service on 100 images, and the 
  entire set consists of ~50(?) images.


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
		`type`: 
			1 - Red text, usually price.
			2 - Grey text, usually more description of savings.
			3 - Dark black text, usually name of item.
			4 - Lighter blcak text, usually more information on the item.
			5 - Body text, more information on deal or random shit. 
    },
    `dict`{
        ...
    }
]
```

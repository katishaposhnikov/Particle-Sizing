# Paricle Sizing

## Installation

1. Click on the green CODE button and choose `Download Zip`
2. Extract the zip file and find `kati.py`
3. Open a terminal (or CMD prompt) in the unzipped directory
4. `pip install -r requirements.txt`
5. `python kati.py`

## Assumptions

1. Scale is the longest connected component in the rectangular user selection
2. Scale is always equal to 1 mm in real life
3. Images are similar to `particle1.png`, a particle under a microscope

## Debugging

1. The _scale_ will be highlighted in red in the original image once you make a selection and press 'Done'. Double check
   that what the program thinks is the scale is indeed correct. Otherwise, don't trust the area calculation.

2. The _particle_ will be shown in white on the right side after the background has been filtered. If whatever is in
   white doesn't look like the outline of the particle, don't trust the area calculation.
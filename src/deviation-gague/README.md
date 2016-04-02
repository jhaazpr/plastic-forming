## Evaluating feature extraction of tracking points
1. run `$ python deviation-gague.py`
2. To evaluate the tracking points: click *exactly four* times in the following order:
   top left, top right, bottom right, bottom left. If you make a mistake, press 'r' to
   reset the points, then start over
3. Hit 'e' to evaluate the points. The squared errors of distances are printed
   to the console.

## Basic flow: extract contours
1. run `$ python deviation-gague.py`
2. Press 'H' to transform the image (crop and remove perspective), press 'T' to
   threshold the image to black and white, and press 'C' to save the contours
   to a file.

```
FILE HIERARCHY:
dataset
  L filipino
  |   L original
  |   |    L wo
  |   |    L wl
  |   |    L wh
  |   |    L so
  |   |    L sl
  |   |    L sh
  L nonfilipino
  |   L original
  |   |    L wo
  |   |    L wl
  |   |    L wh
  |   |    L so
  |   |    L sl
  |   |    L sh
  L raw
  |   L abrera
  |   L arre
  |   L borja
  |   L kampilan
  |   L ebdtheque
  L raw-initial
  |   L abrera
  |   |    L wo
  |   |    L so
  |   L arre
  |   |    L wo
  |   |    L so
  |   L borja
  |   |    L wo
  |   |    L so
  |   L kampilan
  |   |    L wo
  |   |    L so
  |   L ebdtheque
  |   |    L wo
  |   |    L so
```

Description:
This is a folder for managing the dataset for the special problem.
* `dataset/raw` - Initial randomly-selected scanned documents are put inside this directory
* `dataset/raw-initial` - Initial resized whole original page (wo) and segmented single panels (so) from those pages are stored here
* `dataset/filipino` and `dataset/nonfilipino` - final pre-processed working datasets to be used in cnn:
    * original - all images to be used which will be divided into the training, validation, and test datasets
    * train - dataset for training
    * validation - dataset for validation
    * test - dataset for testing
    Note: whole original (wo), whole lineart (wl), whole histogram (wh), single-panel original (so), single-panel lineart (wl), single-panel histogram (wh) 

Distribution:
Filipino Dataset
* Abrera
    * Kikomachine vol 8 - 35 pages
    * 12 - 35 pages
    Total: 70 pages
    will use: 56 pages training - 7 pages validation - 7 pages test
* Arre
    * The Mythology Class (2019) - 70 pages
    Total: 70 pages
    will use: 56 pages training - 7 pages validation - 7 pages test
* Borja
    * Hunghang Flashbacks Book 1 - 35 pages
    * Hunghang Flashbacks Book 2 - 35 pages
    Total: 70 pages
    will use: 56 pages training - 7 pages validation - 7 pages test
* Kampilan
    *  Dead Balagtas, Tomo 1: Mga Sayaw ng Dagat at Lupa - 70 pages
    Total: 70 pages
    will use: 56 pages training - 7 pages validation - 7 pages test

Non-Filipino dataset
* EBDtheque
    70 pages (American, French, and Japanese)
    will use: 56 pages training - 7 pages validation - 7 pages test

Filename format:
<\author>_<\number>

Additional Notes:
* All documents scanned on a Canon PIXMA E510 

References:
ImageLab 2012
OpenCV: OpenCV modules. (2022). Opencv.org. https://docs.opencv.org/4.x/
Keras - Convolution Neural Network. (2022). Tutorialspoint.com. https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm
API Reference. (2022). Scikit-Learn. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
Python OpenCV - Connected Component Labeling and Analysis - GeeksforGeeks. (2022, April 19). GeeksforGeeks. https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/

‌
# AutomatedMBS

The automated modified Bern score (MBS) is a trained deep learning model for the automatic and standardized grading of histological images of cartilage.

## Download the trained model

The trained model (state_dict_densenet_classifier1.pt) can be downloaded from: Mendeley Data at [doi: 10.17632/wrdjkxhhs7.1](https://doi.org/10.17632/wrdjkxhhs7.1)

## Grade images using the trained model

Once you have the trained model downloaded (state_dict_densenet_classifier1.pt), you can use the Jupyter Notebook file to grade the example images or your own images. This notebook is set up to grade the example image tiles grouped by donor. The tiles were gradeded manually by a user and are found in grade-specific subdirectories within a donor directory, similar as shown here.

```
+---pictures
    +---donor1
        +---MBS0
        |     AC1_1.tif
        |     AC1_2.tif
        +---MBS1-2
        |     AC1_3.tif
        +---MBS3-4
        |     AC1_4.tif
        +---MBS5-6
              AC1_5.tif
```
## How was the model trained

Cartilaginous tissues were engineered from human articular or nasal chondrocytes or from bone-marrow derived mesenchymal stromal cells. Safranin O and fast green stained histological images of the tissues were graded for chondrogenic quality according to a modified version of the Bern score (Modified Bern Score), which ranks images on a scale from 0 to 6 according to the intensity of staining and cell morphology. The images were graded by two experts and grouped into four categories with the following grades: 0, 1-2, 3-4, and 5-6. Deep learning with transfer learning was used to train a model to classify images into these histological score groups. 

Cartilaginous tissues of diverse quality were produced, covering all categories of the Modified Bern Score. Transfer learning using a pretrained DenseNet model was selected.

## Citation

If you use this tool, please cite the following article:

L. Power, L. Acevedo, R. Yamashita, D. Rubin, I. Martin, A. Barbero. Deep learning enables the automation of grading histological tissue engineered cartilage images for quality control standardization,
Osteoarthritis and Cartilage, Volume 29, Issue 3, 2021, Pages 433-443, ISSN 1063-4584, https://doi.org/10.1016/j.joca.2020.12.018.

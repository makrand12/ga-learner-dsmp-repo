### Project Overview

 This dataset was generated using HRSC nadir panchromatic image h0905_0000 taken by the Mars Express spacecraft. The images is located in the Xanthe Terra, centered on Nanedi Vallis and covers mostly Noachian terrain on Mars. The image had a resolution of 12.5 meters/pixel.

Problem statement
Determine if the instance is a crater or not a crater. 1=Crater, 0=Not Crater

About the dataset
Using the technique described by L. Bandeira (Bandeira, Ding, Stepinski. 2010.Automatic Detection of Sub-km Craters Using Shape and Texture Information) we identify crater candidates in the image using the pipeline depicted in the figure below. Each crater candidate image block is normalized to a standard scale of 48 pixels. Each of the nine kinds of image masks probes the normalized image block in four different scales of 12 pixels, 24 pixels, 36 pixels, and 48 pixels, with a step of a third of the mask size (meaning 2/3 overlap). We totally extract 1,090 Haar-like attributes using nine types of masks as the attribute vectors to represent each crater candidate. The dataset was converted to the Weka ARFF format by Joseph Paul Cohen in 2012.


### Learnings from the project

 After completing this project, you will have the better understanding of how feature selection effects the performance of a machine learning model. In this project, you will apply the following concepts.

How are the features important to our model.
How to select the most significant features out of many.
How to perform univariate feature selection.
How to perform multivariate feature selection.



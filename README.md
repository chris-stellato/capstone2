# Chris Stellato - Capstone 2 project




#### Project Proposal: Face Mask Detection in Images
Face Mask Detection image set
https://www.kaggle.com/andrewmvd/face-mask-detection

##### Dataset
This is an image set sorted into 3 classes: mask, no mask, and incorrectly worn mask. This dataset could be used to train and test various machine learning models and is extremely relevant for present and future applications. 

* image format: .png
* total images: 950

#### Machine Learning Model:
Use mask, no mask, and incorrect mask labels to develop and train a machine learning model using images from the dataset. Test the model against holdout data and after verifying accuracy and RMSE, test against user submitted photos. 



#
## Daily project updates

#### Monday: 
- read in image data
- subset data for EDA
- EDA: manual image processing and transformation (grayscale, sobel)
- Vectorize
- create randomforest model
- fit, predict, score rf model w/ simple parameters on small dataset
- fit, predict, score rf model w/ larger dataset (images 224x224)
- create, fit, predict, score GradientBoostingClassifier model w/ full dataset (images 224x224)
- Create keras image generator


#### Tuesday: 
- Create keras Sequential() model, fit, train, test, tune learning rate
- create, train, predict and evaluate modified vgg16-based sequential model


#### Wednesday: 
- tune # trainable paramteters in vgg16 based model 
- write helper functions to set up images folder structure and divide images for train-valid-test splits
- create, train, predict new model and save
  - 18 epochs, 3M trainable parameters, loss: 0.0323 - accuracy: 0.9830  (could use ~15 epochs)
- writing code to run predictions on new samples not in the original tvt split
- begin presentation outline
- clean up code and work on turning notebook code into functions


#### Thursday: 
- Clean up all notebook codes, re-run all models on current image set and update performance metrics
- create graphs of model performance, confusion matricies, and performance over epochs graphs
- Finish presentation and fill in updated performance metrics
- re-tune long-run model and prepare for final long-run this evening





#
## Sample Project progression: 
- Business understanding: outline how what insight we hope to gain from the data
- Data mining: Download the original images and dataset. Look for opportunities to combine multiple image sets
- Data cleaning: Reading in and preparing the images for machine learning.  
- Data exploration: Understanding image format and thinking about pipeline to process the images from various sources 
- Feature engineering: image data generator and image feature engineering
- Predictive Modeling: 
  - split holdout data
  - scale and normalize, vectorize
  - NMF or SVD to reduce number of features
  - grid search across various learning models
  - cross validate
  - assess RMSE, accuracy, R2 score and choose the best model and hyperparameters
  - assess model against holdout data and report on findings. 
- Data visualization: 
  -  graph of metrics of various models
  -  graph of distribution of sample classes
- Business understanding: test against people who submit sample photos. 

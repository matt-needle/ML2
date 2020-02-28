# --------------------------------------
# Exercise: Classification
# --------------------------------------

# 2019 Laboratorium of Geo-Information Science and Remote Sensing (GRS)
# Wageningen University
# 
# Devis Tuia
# Diego Marcos
# Sylvain Lobry
# Benjamin Kellenberger




# ----------------------------------------------
# 1. Setup
# ----------------------------------------------

# clear previous variables
rm(list=ls())

# import packages
library(raster)               # package for manipulating raster data
library(randomForest)         # our actual classifier is implemented in this package
library(caret)                # collection of various machine learning tools. We use it here for the confusion matrix at the end
library(rgdal)


# set working directory (MODIFY THE PATH FOR YOUR CASE ACCORDINGLY)
setwd('/home/sytze/Documents/WUR/Machine Learning/Week2/Project/ML2')




# ----------------------------------------------
# 2. Data Preparation
# ----------------------------------------------

# 2.1a Load the data

# images: RGB
im1 <- stack('data/image1_rgb.tif')
im2 <- stack('data/image2_rgb.tif')

# images: NIR
im1_nir <- stack('data/image1_nir.tif')
im2_nir <- stack('data/image2_nir.tif')


# stack together to R-G-B-NIR matrices
im1 <- stack(c(im1,im1_nir))
im2 <- stack(c(im2,im2_nir))


# give uniform names to the bands (we need that for the Random Forest later)
names(im1) <- c("R","G","B","NIR")
names(im2) <- names(im1)


# copy the data (for visualisation later)
im1_orig <- im1
im2_orig <- im2


# ground truth (for Image 1)
gt1 <- raster('data/image1_groundTruth.tif')



# classes and legend
classColors = c(
  rgb(1,1,1),               # Background
  rgb(0,0,0),               # Roads
  rgb(.4,.4,.4),            # Buildings
  rgb(0,.5,0),              # Trees
  rgb(0,1,0),               # Grass
  rgb(0.6,0.3,0),           # Bare Soil
  rgb(0,0,0.6),             # Water
  rgb(1,1,0),               # Railways
  rgb(0.6,0.6,1)            # Swimming Pools
)




# 2.1b Visualise the images

par(mfrow=c(1,2))       # plot two images side-by-side

# image 1
plotRGB(im1_orig,r=4,g=3,b=2,main="Image")                    # we plot bands (4,3,2) (NIR-R-G) as (R,G,B) Bands B and G were switched due to typo in code

# ground truth
plot(gt1,col=classColors[1:(max(unique(gt1))+1)],main="Ground Truth",axes=FALSE)   # the "unique" command retrieves all classes present in the image (may be less than in "legend"); the +1 is for R's 1-indexing

# ----------------------------------------------
# 3. Feature Generation
# ----------------------------------------------


# 3.1 NDVI
calcNDVI <- function(img) {
  ndvi <- (img[[4]]-img[[3]])/(img[[4]]+img[[3]])
  names(ndvi) <- "ndvi"       # assign standardised name to RasterLayer object
  return(ndvi)
}

# calc. NDVI values and append to images
ndvi1 <- calcNDVI(im1)
im1 <- stack(c(im1,ndvi1))
ndvi2 <- calcNDVI(im2)
im2 <- stack(c(im2,ndvi2))


# visualise NDVI for Image 1
graphics.off()               # clear current plots
plot(ndvi1,axes=FALSE,col=colorRampPalette(c("#20442C", "#D6FFD3"))(255))   # we give it a black-to-green colourmap



# 3.2a Local Average
calcLocalAverage <- function(img,KS) {
  
  # prepare empty output
  out <- brick()
  
  ####### YOUR CODE HERE #######
  
  # create window matrix
  window <- matrix(1, nrow = KS, ncol = KS)
  
  # loop through all layers (= bands) of img
  for (layer in 1:nlayers(img)) {
    layer.mean <- focal(x = img[[layer]], w = window, fun = mean)
    out <- addLayer(out, layer.mean)
  }

  # Hints:
  # - Calculating a local average is a moving window operation,
  #   also known as a "focal" operation. Try to find out how to
  #   do this in R by searching the Internet.
  # - Remember that we have five bands (R-G-B-NIR-NDVI); it thus
  #   makes sense to calculate local averages for all of them.
  # - In this exercise we will use a KS x KS moving window.
  # - What happens at the corners and borders of the image?
  #   Check the documentation to find out how to address these
  #   special cases.
  
  ##############################
  
  # assign names
  names(out) <- sprintf("locAvg_%s_%s",KS,names(img))
  
  return(out)
}

# define kernel size
KS = 5

# calculate local averages per image and append
locAvg_im1 <- calcLocalAverage(im1,KS)
im1 <- stack(c(im1,locAvg_im1))
locAvg_im2 <- calcLocalAverage(im2,KS)
im2 <- stack(c(im2,locAvg_im2))








# 3.2b Local Standard Deviation

# To calculate the local standard deviation we could in principle provide a
# custom function to R, but this is not optimised and therefore very slow.
# However, we can use a trick to avoid that. The local variance can also be
# calculated as follows:
#
#     var(x) = N/(N-1) * (mean(x^2) - mean(x)^2)
# 
# Where N is the number of elements involved (filter size in our case).
# This means that we can calculate the standard deviation by just calculating
# two means, which is much faster. Try to do that now by re-using the
# "calcLocalAverage" function you just completed above.
# Remember to only use the five bands at maximal resolution (R-G-B-NIR-NDVI),
# not the local averages. You can select bands by using "img[[1:5]]".

calcLocalStandardDeviation <- function(img,KS) {
  
  ####### YOUR CODE HERE #######
  
  # prepare empty output
  localSd <- brick()
  
  # calculate window size n
  n <- KS ^ 2

  # loop through all original bands of image
  for (layer in 1:5) {
    layer.var <- (n / (n - 1)) * ((calcLocalAverage(img[[layer]] ^ 2, KS) - calcLocalAverage(img[[layer]], KS) ^ 2))
    localSd <- addLayer(localSd, layer.var)
  }
  ##############################
  
  # remove potential NaN values
  localSd[is.na(localSd)] = 0
  
  # assign names
  names(localSd) <- sprintf("locStd_%s_%s",KS,names(img)[1:5])
  
  return(localSd)
}


# now apply the function, like above

####### YOUR CODE HERE #######
locSd_im1 <- calcLocalStandardDeviation(im1,KS)
im1 <- stack(c(im1,locSd_im1))
locSd_im2 <- calcLocalStandardDeviation(im2,KS)
im2 <- stack(c(im2,locSd_im2))
##############################




# visualise
graphics.off()
plot(locSd_im1)





# 3.3 Load additional features
loadFiles <- function(folder) {
  files = list.files(folder)
  
  feat = brick()
  for(f in files) {
    ff <- raster(paste(folder,f,sep=""))
    feat <- stack(c(feat,ff))
    
  }
  names(feat) <- sprintf("feat%s",seq(1:nlayers(feat)))
  return(feat)
}
im1_extraFeat <- loadFiles('data/extraFeatures/im1/')   # extra features (stored as TIFF images) for im1
im1 <- stack(c(im1,im1_extraFeat))
im2_extraFeat <- loadFiles('data/extraFeatures/im2/')   # the same for im2
im2 <- stack(c(im2,im2_extraFeat))





# resolve potential NaN values (again, just to be 100% sure)
im1[is.na(im1)] = 0
im2[is.na(im2)] = 0





# 3.4 Normalise the data
normalise <- function(img, means = NULL, stdDevs = NULL) {
  if(is.null(means)) {
    means <- cellStats(img,mean)
    stdDevs <- cellStats(img,sd)
  }

  # normalise
  normalised <- (img - means)/stdDevs

  return(list(normalised,means,stdDevs))
}


# apply the normalisation function

####### YOUR CODE HERE #######

# Hint: remember which statistics to use for which image.
# Refer to the PDF for details.
normalisation <- normalise(im1)
im1 <- normalisation[[1]]
im2 <- normalise(im2, normalisation[[2]], normalisation[[3]])[[1]]
##############################










# ----------------------------------------------
# 4. Training / Validation Set Splits
# ----------------------------------------------


# 4.1 Define splits

# define split amounts
fracTraining <- 0.7                                      # 70% of image1 for training, 30% for validation; image2 for testing

# split spatially
ext_im1 <- extent(im1)                                   # extent of the entire image1 (format: xmin, xmax, ymin, ymax)
splitLoc = round(ext_im1[4] * fracTraining)              # we split along y-axis

# create new extents for the training and validation set
ext_train <- extent(0,ext_im1[2],0,splitLoc)               # left 70% for training
ext_val <- extent(0,ext_im1[2],splitLoc+1,ext_im1[4])      # right 30% for validation



# apply extents to image1 and ground truth
im1_train <- crop(im1,ext_train)
im1_val <- crop(im1,ext_val)

gt1_train <- crop(gt1,ext_train)
gt1_val <- crop(gt1,ext_val)



# visualise set side-by-side
graphics.off()
par(mfrow=c(1,2))

# training set
plot(gt1_train,col=classColors[1:(max(unique(gt1_train))+1)],main="Training Set",axes=FALSE)

# validation set
plot(gt1_val,col=classColors[1:(max(unique(gt1_val))+1)],main="Validation Set",axes=FALSE)     




# set seed for comparison
set.seed(69)


# ----------------------------------------------
# 5. Model Creation
# ----------------------------------------------


# 5.1 Cross-validate parameters

# define range of parameters to use for grid search
numTrees_range = c(1,10,25,50,100)                   # typical num. trees can go up to 1000, depending on the dataset size and model complexity
minLeaf_range = c(1,2,5,10,25,50)

# prepare matrix for storing model OA reached on the validation set per parameter combination
oa_crossval = matrix(0,length(numTrees_range),length(minLeaf_range))

# we also measure the time it takes to train models
time_crossval = matrix(0,length(numTrees_range),length(minLeaf_range))


# prepare data: extract values
x_train <- getValues(im1_train)
x_val <- getValues(im1_val)

classes = unique(gt1)
classes = classes[classes!=0]                               # we remove the "background" class from all analyses

y_train <- factor(getValues(gt1_train), levels=classes)     # the "factor" command converts the ground truth into the given levels (categories),
y_val <- factor(getValues(gt1_val), levels=classes)         # otherwise the RandomForest would perform regression
                                                            # Important: we provide "classes" for levels which don't contain the zero class.
                                                            # As a result of this, R converts all zero entries to <NA> (not-a-number)



# also, we will take a tiny subset of all the pixels for training so that we don't have to wait for too long.
# We restrict it to 10 percent of each training class (only for the training set)
cls_train <- unique(y_train)                                                     # all classes in y_train
cls_train <- cls_train[!is.na(cls_train)]                                        # ignore the "background" class that still is in y_train
valid_train <- integer()
trn_frac = 0.1                                                                   # 10% of the pixels for training (per class)
for(c in cls_train) {
  cls_sub <- which(y_train==c)                                                   # find all training set indices for class c
  cls_sub <- sample(cls_sub,ceiling(trn_frac*length(cls_sub)),replace=FALSE)     # random sampling
  valid_train <- c(valid_train,cls_sub)                                          # append to new set of indices
}




# verify that the "zero" class is not present in the training set
sprintf("Training set classes: %s", paste(unique(y_train[valid_train]),collapse=", "))

# do the evaluation: iterate over all the parameter combinations
for(nT in 1:length(numTrees_range)) {
  for(mL in 1:length(minLeaf_range)) {
    
    # measure time
    starttime <- Sys.time()
    
    # train Random Forest
    
    ####### YOUR CODE HERE #######
    
    # Hint: remember to subset the data you use for training the Random Forest
    rf_model <- randomForest(y_train ~ ., data = x_train, subset = valid_train, ntree = numTrees_range[nT], mtry = minLeaf_range[mL])
    ##############################
    
    time_crossval[nT,mL] <- (Sys.time() - starttime)
    
    # predict on validation set
    pred <- predict(rf_model,x_val)
    
    # accuracy assessment: compare with ground truth
    oa_crossval[nT,mL] <- sum(pred==y_val,na.rm=TRUE) / length(!is.na(y_val))         # again: ignore background class
    
    
    print(sprintf("Num. Trees: %i, Min Leaf: %i  -  Time: %s s,  OA: %s perc.",numTrees_range[nT],minLeaf_range[mL],format(time_crossval[nT,mL],digits=2),format(100*oa_crossval[nT,mL],digits=4)))
  }
}


# visualise performances
graphics.off()
par(mfrow=c(1,2))

image(oa_crossval, axes=FALSE, xlab="Num Trees", ylab="Min Leaf Size")
axis(1, at=seq(0,1,length=length(numTrees_range)), labels=numTrees_range)
axis(2, at=seq(0,1,length=length(minLeaf_range)), labels=minLeaf_range)
e <- expand.grid(seq(0,1,length=length(numTrees_range)), seq(0,1,length=length(minLeaf_range)))
text(e, labels=format(100*oa_crossval,digits=4))
title("Overall Accuracies on Validation Set")

image(time_crossval, axes=FALSE, xlab="Num Trees", ylab="Min Leaf Size")
axis(1, at=seq(0,1,length=length(numTrees_range)), labels=numTrees_range)
axis(2, at=seq(0,1,length=length(minLeaf_range)), labels=minLeaf_range)
e <- expand.grid(seq(0,1,length=length(numTrees_range)), seq(0,1,length=length(minLeaf_range)))
text(e, labels=format(time_crossval,digits=2))
title("Time Required to train RF")




# pick your favourite parameters according to plot

####### YOUR CODE HERE #######
minLeafSize <- 2
numTrees <- 25
##############################





# ----------------------------------------------
# 6. Predictor importance
# ----------------------------------------------


# 6.1 Assess predictor importance

####### YOUR CODE HERE #######

# Hint: check the Random Forest documentation to see how to train it
# with predictor importance assessment turned on.

rf_model <- randomForest(...)
##############################

graphics.off()
varImpPlot(rf_model)                   # plot the variable importances


# select which variables you would like to keep

####### YOUR CODE HERE #######
vars_keep <- ...
##############################




# ----------------------------------------------
# 7. Train final model & predict on test set
# ----------------------------------------------

# 7.1 Train the final Random Forest

# now we are going to use the full training set (bar the "zero" class)
valid_train_full <- which(!is.na(y_train))           # take all pixels except those with not-a-number value ("background")


# first we train the Random Forest (this will take a while...)

####### YOUR CODE HERE #######
rf_model <- randomForest(...)
##############################





# 7.2 Predict Image 2 (this again may take a while...)
pred_im2 <- predict(rf_model,getValues(im2))


# pred_im2 is currently a 1D vector, so we have to reshape it to a 2D image
ext_im2 <- extent(im2)                                                                # size of image2 to which we want to reshape the prediction
pred_im2 <- matrix(as.numeric(classes)[pred_im2],nrow=ext_im2[2],ncol=ext_im2[4])     # convert to numeric values again for easier reshaping
pred_im2 <- t(pred_im2)                                                               # R is column-first, so to reconvert to a raster object we have to transpose (swap the x and y axes)
pred_im2 <- raster(pred_im2)




# finally load the ground truth for Image 2
gt2 <- raster('data/image2_groundTruth.tif')



# visualise the image, prediction and ground truth for image 2 side-by-side
par(mfrow=c(1,3))

# image 2
plotRGB(brick(im2_orig),r=4,b=3,g=2,main="Image 2")

# prediction
plot(pred_im2,col=classColors[2:(max(unique(pred_im2))+1)],main="Prediction",axes=FALSE)

# ground truth
plot(gt2,col=classColors[1:(max(unique(gt2))+1)],main="Ground Truth",axes=FALSE)





# ----------------------------------------------
# 8. Accuracy Assessment
# ----------------------------------------------

# 8.1 Run the accuracy assessment

# flatten predictions and ground truth into linear vectors that are directly comparable
gt2_flat <- as.vector(gt2)
pred2_flat <- as.vector(pred_im2)


# again exclude the "background" class
valid <- which(gt2_flat!=0)



# convert our prediction and ground truth to factors (nominal scale)
pred_im2_test <- factor(pred_im2[valid])    
gt_im2_test <- factor(gt2_flat[valid])




# calculate the confusion matrix
statistics <- confusionMatrix(pred_im2_test, gt_im2_test, dnn=c("Prediction", "Reference"))


# you can access individual properties of the output as follows (example):
statistics$table

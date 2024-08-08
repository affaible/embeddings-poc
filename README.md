# T-SNE for Posterior Analysis of Deep Learning Models

Vincent Warmerdam proposes a T-SNE approach to the problem of [bulk labelling](https://github.com/koaning/bulk?tab=readme-ov-file).

Here we extend this tooling to other parts of the ML workflow.

## Background


**"What does the model I developed based on my assumptions say about the data that I care about?"**

In a lecture on [Probabilistic Models](https://www.youtube.com/watch?v=DaqNNLidswA) David Blei describes an iterative workflow that he calls **Box's Loop** (@11:50).

![image](box_cycle.png "Box Cycle")

1. Make assumptions about the world
2. Discover patterns based on data and the assumptions
3. Use the discovered patterns to form predictions
4. Look critically at what happened
5. Use the learning and criticism phase to revise assumptions

These cyclical representations of workflows are common in Data Science.

## Problem Statement

Deep Learning models are often seen as a black box. This can create a disconnect between the model and the training data.
Furthermore labelling the large datasets for a dNN, combined with long training times, result in long cycle times in industrial workflows.
The meta-learning rate is slow, the researchers take a long time to learn about how the way the model interacts with the data.

- Rather than a linear 'train and forget' ML pipeline, an iterative process is required. Tools should support continuous improvement of the models.
- The inherent opacity of Deep Learning models should be minimised by providing visibility of the way the model relates to the data it was trained on.

## Baseline

We will use a simple computer vision classifier from `adimena` to create a spike: [cows-vs-no-cows](https://github.com/adimena/Cows-vs-no-cows). 
The project has used training data from a mixture of sources. This binary classification required no labelling step, the data was pre-sorted.

`Bulk` will not be used labelling for this task. The baseline step is purely about integrating the `adimena` data with the `Bulk` tool in the simplest conceivable way  to test technology and understanding.  

### 1. Data Preparation 

Take a random reduced (100 images) sample set across the entire dataset.

```
mkdir smpl_set && cp `ls | shuf -n 100` smpl_set
```

### 2. Create the Input File

Create the `ready.csv` input file with the [embetter](https://github.com/koaning/embetter) embeddings as shown in the [example](https://github.com/koaning/bulk?tab=readme-ov-file#bulk-image)

### 3. Run

The Bulk tool takes the created `ready.csv` as input

```
python -m bulk image ready.csv
```

----

![image](embetter-tsne.png "Embetter T-SNE")

The effect of using `embetter` to create the embeddings is apparent as the T-SNE plot has no evident clustering.

----

Taking a lasso of the top right:

![image](embetter-top-right.png "Embetter Top Right T-SNE")

----

And the bottom left:

![image](embetter-bottom-left.png "Embetter Bottom Left T-SNE")

----

We can see some visual differences between the two groups, but it is not compelling. This is to be expected with off-the-shelf embeddings. 
The technology and results work as expected at this point. We need to take this further to achieve tooling that assists with Box's Loop.

## Trained Model Embeddings

The baseline used off-the-shelf embeddings; `embetter`.
We are interested in the interaction between the trained model and the original data. 
To be useful for posterior analysis we should use the embeddings from the trained model. 
The `Bulk` ReadMe suggests we are free to use whatever embeddings we like.
The next steps show how we can change the data preparation to produce an input file that will provide the desired insight into the model-data interaction. 

### 1. Create Embeddings

Create embeddings from the `adimena` model.

```python
"""                                                                                                                                                           
Create embeddings from an existing CNN model and it's training data.                                                                                          
"""

import glob

import numpy as np
import pandas as pd
import keras
from keras.preprocessing import image
from keras.models import Sequential
import umap

# Load in adimena's training model:                                                                                                                           
training_model = keras.models.load_model('adimena-model.keras')

print(training_model.summary())

# There are three Dense layers - create a new model that drops these:                                                                                         

embeddings_model = Sequential()
for layer in training_model.layers[:-3]:
    embeddings_model.add(layer)

print(embeddings_model.summary())

# Iterate through the sample set to collect the images                                                                                                        

fnames = []
image_data = []
for fname in glob.glob("smpl_set/*.jpg"):
    fnames.append(fname)
    img = image.load_img(fname, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_data.append(x)

# Create the embeddings from the trained model

images = np.vstack(image_data)
embeddings_arr = embeddings_model.predict(images)
 
# Reduce the dimensionality                                                                                                                                  

reducer = umap.UMAP()
reduced_vectors = reducer.fit_transform(embeddings_arr)
 
# Create input csv file for Bulk                                                                                                                              
df = pd.DataFrame(reduced_vectors.tolist(), columns=['x', 'y'])
df['path'] = fnames
df = df[['path', 'x', 'y']]
df.to_csv('adimena_embeddings.csv', index=False)
```

The model used for training had the following architecture:

```
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896

 max_pooling2d (MaxPooling2D  (None, 74, 74, 32)       0 
 )

 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0 
 2D)

 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856

 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0
 2D)

 flatten (Flatten)           (None, 36992)             0

 dense (Dense)               (None, 512)               18940416

 dense_1 (Dense)             (None, 128)               65664

 dense_2 (Dense)             (None, 1)                 129

=================================================================
```

The model used to create the embeddings from the sample set:

```
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896

 max_pooling2d (MaxPooling2D  (None, 74, 74, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0 
 2D)

 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856

 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0 
 2D)

 flatten (Flatten)           (None, 36992)             0 

=================================================================
```

### 2. Run

Load the created embeddings into `Bulk`:

```
python -m bulk image adimena_embeddings.csv
```

----

The data points fill more of the surface than the `embetter` example and there is some clustering evident.

![image](adimena-tsne.png "Adimena T-SNE")

----

Taking a look at one of these clusters

![image](kaggle-set-tsne.png "Kaggle Set T-SNE")

This is encouraging, not only are the images visually similar they come from [the same dataset](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k)! The cluster comes from a single target category (cows). The kaggle weight detection dataset provides a readily available source of pre-labelled data. The photos are visually very similar to one another in backgrounds and compositions. However the composition is quite different from the data collected in the field. The distribution of the clusters suggest that the current balance of the data sources needs further work to improve the classification around the target variables.

----


![image](adimena-green-tsne.png "Green T-SNE")

Another cluster shows how there can be a group of data points that are visually similar. This time the cluster cuts across the desired categories.

----

## Conclusion

The opacity of models trained on a dNN necessitates specialist tooling to take advantage of methodologies promoted in other areas of ML. Labelling technologies can be applied to model analysis, providing that the workflow integration is sufficiently flexible. Moreover, a process that refines the training data and embeddings on successive iterations could change the nature of labelling. Rather than a single outsourced operation with long lead times, data labelling and training should be a continuous collaborative process.

With some small hacks the `Bulk` tool has been shown to remove the guess work on analysis of a simple CNN binary classifier. Understanding the influence of aspects of the data has the potential to improve training data quality and reduce trial and error. Visualisation tooling on the trained embeddings seems to fit well with iterative workflows (such as Box's Loop). 

Posterior analysis using T-SNE enables an efficient examination of the models interpretation of the training data.

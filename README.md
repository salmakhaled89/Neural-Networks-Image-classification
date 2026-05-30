# Neural-Networks-Image-classification
A Neural Networks framework implementing and comparing CNN and FFNN architectures for 6-class waste image classification (cardboard, glass, metal, paper, plastic, trash).
# Waste Classification Platform via Deep Learning (CNN vs. FFNN)

An automated, industrial-grade Deep Learning repository designed to categorize solid waste into six distinct eco-friendly categories. The project implements, trains, and evaluates **Convolutional Neural Networks (CNN)** against baseline dense **Feed-Forward Neural Networks (FFNN)** to optimize image-based classification accuracy.

##  Target Classes
The models are trained to classify image inputs into one of 6 structural classes:
1. `Cardboard`
2. `Glass`
3. `Metal`
4. `Paper`
5. `Plastic`
6. `Trash`

---

##  Model Architectures & Methodologies

### 1. Convolutional Neural Network (CNN)
- Designed with multiple stacked `Conv2D` layers utilizing **ReLU** activation to extract localized spatial hierarchies and textures from raw image matrices.
- Embedded `MaxPooling2D` layers for dimensional down-sampling and translation invariance.
- Finalized with a dense `Softmax` output layer predicting probabilities across the 6 waste categories.

### 2. Feed-Forward Neural Network (FFNN)
- Deployed as a baseline architectural model by flattening input image arrays into 1D vectors before passing them through fully connected dense hidden layers.
- Used to benchmark the computational superiority and structural necessity of convolutional feature extractors over standard dense nets for spatial data.

###  Memory & Large Dataset Optimization
> **Note:** Due to the large volume and size of the image directory datasets, the raw graphics are omitted from GitHub repository tracking via `.gitignore`. 
- **Data Pipelines:** Built using asynchronous batch loaders (e.g., Keras `ImageDataGenerator` or `tf.data`) to stream, normalize, and feed pixel arrays directly from localized runtimes into the tensor pipelines, eliminating CPU memory bottlenecking.

---

##  Tech Stack & Frameworks
- **Core Frameworks:** pytorch,numpy
- **Scientific Computing:** NumPy, Pandas
- **Visualization & Metrics:** Matplotlib, Seaborn, Scikit-Learn Confusion Matrix
-

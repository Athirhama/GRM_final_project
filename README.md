DGCNN for ShapeNet Part Segmentation
This project implements Dynamic Graph Convolutional Neural Networks (DGCNN) for point cloud part segmentation on the ShapeNet dataset. The repository includes scripts for data preprocessing, model training, evaluation, and 3D visualization of results.

Project Structure
data.py: Contains the ShapeNetPart dataset class and data augmentation logic (scaling and translation).

dataset_gen.py: Handles the download and conversion of raw ShapeNet data into optimized NumPy binaries.

main.py: The primary script for training and evaluating the model.

model.py: Defines the DGCNN architecture, including EdgeConv blocks and the segmentation head.

utils.py: Provides helper functions for graph feature extraction and IoU (Intersection over Union) metrics.

vision.py: A visualization tool to generate 3D scatter plots comparing predictions against ground truth.

checkpoints/: Folder containing the saved model weights (best_model.pth).

logs/: Folder containing training history and performance logs for review.

Installation
Ensure you have Python 3.8 or higher installed. Install the required dependencies using pip:

Bash
pip install torch numpy tqdm matplotlib kagglehub
Dataset Preparation
The model requires the ShapeNetPart dataset in a specific binary format. Run the following command to download and process the data:

Bash
python dataset_gen.py
This script converts the raw .pts and .seg files into .npy format and organizes them in a ./data/bin directory.

Usage
Using the Pre-trained Checkpoint
To evaluate the model using our provided checkpoint without retraining:


Run the evaluation script:

Bash
python main.py --eval --model_path checkpoints/best_model.pth
Training from Scratch
To start a new training session:

Bash
python main.py -epochs 100


Visualization
To generate 3D visual comparisons of the results:

Bash
python vision.py
This script processes samples from the test set and saves PNG images to the viz_output/ directory, showing the point cloud from three different angles.

Training Logs
For the teacher to see our training logs, those are stored in the logs/ directory. These files contain the loss values, Instance mIoU, and Class mIoU recorded during the original training process.
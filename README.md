# DGCNN for ShapeNet Part Segmentation

This project implements Dynamic Graph Convolutional Neural Networks (DGCNN) for point cloud part segmentation on the ShapeNet dataset. The repository includes scripts for data preprocessing, model training, evaluation, and 3D visualization of results.

---

## Project Structure

* **data.py**: Contains the ShapeNetPart dataset class and data augmentation logic like random scaling and translation.
* **dataset_gen.py**: Handles the download and conversion of raw ShapeNet data into optimized NumPy binaries.
* **main.py**: The primary script for training and evaluating the model using SGD and Cosine Annealing.
* **model.py**: Defines the DGCNN architecture, including EdgeConv blocks and the segmentation head.
* **utils.py**: Provides helper functions for graph feature extraction and IoU metrics.
* **vision.py**: A visualization tool to generate 3D scatter plots comparing predictions against ground truth.
* **checkpoints/**: Folder containing the saved model weights (`best_model.pth`).
* **logs/**: Folder containing training history and Slurm performance logs for review.

---

## Installation

Ensure you have Python 3.8 or higher installed. Install the required dependencies using pip:
```bash
pip install torch numpy tqdm matplotlib kagglehub
````

---

## Dataset Preparation

The model requires the ShapeNetPart dataset in a specific binary format. Run the following command to download and process the data:
```bash
python dataset_gen.py
```

This script converts the raw `.pts` and `.seg` files into `.npy` format and organizes them in a `./data/bin` directory.

---

## Usage


### Training from Scratch

To start a new training session:
```bash
python main.py
```

---

## Visualization

To generate 3D visual comparisons of the results (same session as the one for th training)
```bash
python vision.py
```

This script processes samples from the test set and saves PNG images to the `viz_output/` directory, showing the point cloud from three different angles.

Note: Due to file size constraints, the full set of generated images is not included in this repository. However, a selection of the most representative 3D segmentations (demonstrating the model's performance across different categories) is provided and analyzed in the final project report.

---

## Training Logs

For the teacher to see our training logs, those are stored in the `logs/` directory. These files contain the loss values, Instance mIoU, and Class mIoU recorded during the original training process.

---

## Note for the teachers

The default parameters (learning rate, scheduler details, k-neighbors, etc.) and a comprehensive analysis of the training process are documented in the **final project report**. 

The report also includes a detailed qualitative evaluation with 3D visualizations from `vision.py` that demonstrate the model's performance on the test set.

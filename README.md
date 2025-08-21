# Video Summarization Model

## Model Overview

A **Video Summarization Model** designed to generate concise summaries from lengthy videos by capturing their most important moments. It leverages a Multi-Layer Perceptron (MLP) for importance scoring and a knapsack algorithm for optimal segment selection. The model is tailored for applications such as content browsing, media archiving, and highlight generation, offering a data-driven solution to prioritize video content efficiently.

### Key Objectives
- **Importance Scoring**: Use an MLP to assign importance scores to video segments based on pre-extracted visual features and human-annotated ground truth.
- **Optimal Selection**: Apply a knapsack algorithm to select segments that maximize importance while adhering to a 15% length constraint.
- **Human-Centric Evaluation**: Compare generated summaries to human summaries using precision, recall, and F-score metrics.
- **Generalization**: Ensure compatibility with benchmark datasets like TVSum, with potential adaptability to other datasets.

### Motivation
With the rapid growth of video content, automated tools are essential to extract key information efficiently. Manual summarization is impractical for large-scale use, and generic methods often miss context-specific importance. This model addresses these challenges by learning from human annotations, providing a robust solution for video summarization.

## Model Workflow

The model processes videos through a structured pipeline:

1. **Data Preparation**:
   - Load a pre-processed HDF5 dataset (e.g., `eccv16_dataset_tvsum_google_pool5.h5`) containing 1024-dimensional visual features, human-annotated importance scores, change points, frame counts, and user summaries.
   - Sort videos for consistent processing and set a random seed for reproducibility.
   - Create a `log/` directory for outputs.

2. **Cross-Validation**:
   - Use 5-fold cross-validation to split the dataset into training (80%) and testing (20%) sets, ensuring robust evaluation across diverse videos.

3. **Model Training**:
   - Train an MLP to predict importance scores (0 to 1) for video segments.
   - Employ Mean Squared Error (MSE) loss, Adam optimizer, gradient clipping, and a step learning rate scheduler (reduce by 0.1 every 30 epochs).
   - Train for 50 epochs, logging losses and saving weights per fold.

4. **Summary Generation**:
   - **Process Overview**: The model generates summaries by transforming MLP predictions into a binary vector that prioritizes high-importance segments within a 15% length constraint. The steps are:
     - **Score Prediction**: The MLP predicts importance scores (0 to 1) for downsampled frames (picks, typically one every 15 frames, approximating 2-second segments at typical frame rates) provided in the dataset, indicating their relevance for the summary.
     - **Frame Score Assignment**: Scores are extended to all frames by assigning each predicted score to a temporal block (e.g., 15 frames). Remaining frames are assigned a score of 0.
     - **Segment Score Calculation**: Frames are grouped into segments using predefined change points (scene transitions). The average score within each segment represents its importance.
     - **Knapsack Optimization**: A knapsack algorithm selects segments to maximize total importance (segment scores as values, frame counts as weights) while keeping the summary within 15% of the video’s duration.
     - **Binary Summary Creation**: A binary vector is created, marking frames in selected segments as 1 (included) and others as 0 (excluded).
   - **Outcome**: This process yields concise summaries that balance relevance and brevity, effectively capturing key video moments.

5. **Performance Evaluation**:
   - Compare generated summaries to human summaries using precision, recall, and F-score, with aggregation as average for TVSum or maximum for SumMe.
   - Log per-video and fold-wise F-scores.

6. **Result Storage**:
   - Save model weights and F-scores to the `log/` directory.
   - Display training losses and evaluation metrics in the console.
 

## Key Features
- **Advanced Summary Generation**: Combines downsampled scoring, segment averaging, and knapsack optimization for efficient summaries.
- **Robust Evaluation**: Supports precision, recall, and F-score metrics, adaptable for TVSum (average) or SumMe (maximum).
- **Optimized Processing**: Uses pre-extracted features and downsampled scoring for computational efficiency.
- **Modular Design**: Separates model, summarization, and optimization logic for easy experimentation.
 

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/video-summarization-model.git
   cd video-summarization-model
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy scikit-learn h5py
   ```

3. Download the TVSum dataset:
   - Obtain `eccv16_dataset_tvsum_google_pool5.h5` from the [TVSum repository](https://github.com/yalesong/tvsum).
   - Place it in the `datasets/` directory:
     ```bash
     mkdir datasets
     mv path/to/eccv16_dataset_tvsum_google_pool5.h5 datasets/
     ```

## Usage

1. **Run the Model**:
   - Execute the main script to train, generate summaries, and evaluate:
     ```bash
     python main.py
     ```
   - The script will:
     - Load the TVSum dataset.
     - Perform 5-fold cross-validation.
     - Train the MLP for 50 epochs per fold.
     - Generate summaries and evaluate performance.
     - Save weights and results to `log/`.

2. **Output**:
   - Model weights: `log/model_foldX.pth.tar`.
   - Results: `log/fold_results.txt` with fold-wise and average F-scores.
   - Console: Shows training losses, per-video F-scores, and fold averages.

3. **Customization**:
   - Edit `main.py` hyperparameters:
     - `lr`: 1e-4
     - `max_epoch`: 50
     - `stepsize`: 30
     - `gamma`: 0.1
     - `split_count`: 5
   - Adjust summary length in `vsum.py` (default: 0.15).

## Dataset
The model uses the **TVSum dataset**:
- **50 videos**: Diverse genres (news, documentaries, vlogs).
- **Features**: 1024-dimensional GoogleNet pool5 features.
- **Annotations**: Importance scores, change points (defining ~2-second segments), frame counts, user summaries.
- **Format**: HDF5 (`eccv16_dataset_tvsum_google_pool5.h5`).
- Adaptable to SumMe with minor changes (e.g., `max` F-score).

## Evaluation Metrics
- **Precision**: Proportion of selected frames matching human summaries.
- **Recall**: Proportion of human summary frames captured.
- **F-score**: Harmonic mean of precision and recall, aggregated as average for TVSum or maximum for SumMe.

## Model Structure
```
video-summarization-model/
├── datasets/                    # Dataset directory
├── log/                        # Model weights and results
├── main.py                     # Main script
├── model.py                    # MLP model
├── vsum.py                     # Summary generation/evaluation
├── knapsack.py                 # Knapsack algorithm
└── README.md                   # Model documentation
```
 
 

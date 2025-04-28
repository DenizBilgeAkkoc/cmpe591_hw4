# CMPE591 Homework 4

This repository contains the implementation of CNMP


## Dataset Collection, Training and Testing

### Dataset Collection
  - You can find the data collection script in main.py `homework4.py`.
  - In the repository there already exits some dataset with different trajectory sizes (200, 2000, 5000)
    
### Training
  - You can find the training script in main.py `main.py`.

### Testing
  - The testing script is located in `test_dataset.py`.
  - Please change the following variable to test it using another dataset
    `# --- Parameters ---
    DATASET_PATH = "dataset_200.npz"
    NUM_MSE_TESTS = 100
    `

  - Here is the real output for the test script:
    ```python
    Loaded 200 trajectories from dataset_200.npz
    
    --- Running 100 MSE Evaluation Tests ---
    MSE Evaluation: 100%|█████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 5467.17it/s]
    
    --- Evaluation Results ---
    Number of successful tests: 100
    End-Effector MSE: Mean=0.000029, Std=0.000057
    Object MSE:       Mean=0.000022, Std=0.000044
    Results plot saved to test_dataset_results_plot.png
    ```
  - ![Test Bar Plot](https://github.com/DenizBilgeAkkoc/cmpe591_hw4/blob/main/test_dataset_results_plot.png)

---

## Training Loss

![Training Loss](https://github.com/DenizBilgeAkkoc/cmpe591_hw4/blob/main/training_loss.png)


## How to Run

 Run the training scripts:
   ```bash
   python main.py
   ```

Run the testing scripts:
   ```bash
   python test_dataset.py
   ```


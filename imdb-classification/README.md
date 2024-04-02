# IMDB Movie Review Classification

## Problem Statement
The aim of this assignment is to utilize the IMDB Movie Review dataset for classifying movie reviews as positive or negative, employing Vanilla-RNN and LSTM deep learning models.

## Findings
The findings reveal that, for state dimensions of 20, 50, and 100, the LSTM model consistently outperforms the Vanilla-RNN model, exhibiting an average performance improvement of 11%. However, for state dimensions of 200 and 500, the performance gap narrows, with a lower increase of 6.5%. Notably, augmenting the state dimensions generally correlates with enhanced test accuracy performance.

## Results

### Vanilla-RNN Parameters Tuning
| State Dimension | Epochs | Batch size | Learning rate | Test Accuracy (%) |
|-----------------|--------|------------|---------------|-------------------|
| 20              | 18     | 32         | 0.0001        | 68.76             |
| 50              | 12     | 32         | 0.0001        | 70.64             |
| 100             | 9      | 128        | 0.0001        | 72.66             |
| 200             | 17     | 128        | 0.0001        | 75.57             |
| 500             | 17     | 64         | 0.00001       | 76.19             |

### LSTM Parameters Tuning
| State Dimension | Epochs | Batch size | Learning rate | Test Accuracy (%) |
|-----------------|--------|------------|---------------|-------------------|
| 20              | 20     | 32         | 0.0001        | 77.00             |
| 50              | 20     | 64         | 0.0001        | 78.66             |
| 100             | 20     | 32         | 0.0001        | 80.95             |
| 200             | 19     | 64         | 0.0001        | 80.75             |
| 500             | 12     | 31         | 0.0001        | 81.20             |


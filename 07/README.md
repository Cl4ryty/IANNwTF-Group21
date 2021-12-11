## Usage
- `run.py` contains the code to generate the dataset, initialize the model and start training and testing - **run this file**.
- `model.py` contains the LSTM Cell, Layer, and Model
- `train_and_test_func.py` contains functions for training and testing a model. **(note: functions modified to only use the last hidden_state as prediction)**

## Questions we should answer
### • Can / should you use truncated BPTT here?
Truncated BPTT only performs backpropagation up to a fixed number of time steps instead of unrolling all time steps as normal BPTT does.
So it can be used in this case but for the few time steps we consider here, it probably does not make much sense to use it. After all, regular BPTT works fast enogh.

  
### • Should you rather take this as a regression, or a classification problem?
As the target is a binary decision and we do not try to predict the value of a continuos variable this should be seen as a classification problem.

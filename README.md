# Exploring the learnability of QGANs
This project is a full dive into the realm of quantum generative adversarial networks under the context of `$DIS` stock price prediction. From initiation to completion, all notebooks are documented for reproducibility. Through several numerical experiments, I hope to take an empirical stab at answering the question: which hyperparameters are optimal for quantum generative adversarial learning. 

Along the way, we develop intuitions behind alleviating common training problems we encountered like vanishing gradients. With the application of learning consecutive equity prices, we built a proof-of-concept quantum generative adversarial network and a simple quantum neural network to accomplish the task of predicting the next four days of prices given fifteen prior days of prices. We evaluate quantum model capacity, convergence characteristics, training efficency, and more on the basis of 7 hyperparameters we tuned.


## File structure
To document the complete journey venturing into the space of QGANs, I've uploaded all the notebooks along the way which guided my decisions along the way.

`./data`: Data fetching and processing into dataset

`./QNN Approach/`: Here we evaluate the upper bound of the number of data samples a simple quantum neural network can learn/memorize and a naive attempt to see how a simple parameterized quantum circuit could fit to predict DIS stock prices.

`./Conditional QGAN.ipynb/`: Quantum generative adversarial network used to learn n batches of $DIS data samples. Used for hyperparameter tuning and conclusions.

`./Conditional QWGAN.ipynb`: A conditional quantum wasserstein generative adversarial network architecture applied to this task.

`./Hyperparameter_tuning_logs.pdf`: A record of various hyperparameter configurations and the resulting effect on loss and accuracy.

`./QGANs explained.md`: An accessible introduction to quantum generative adversarial networks.

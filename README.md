# Exploring the learnability of QGANs
This project takes an empirical stab at answering the question: which hyperparameters are optimal for quantum generative adversarial learning. With the application of learning consecutive equity prices, we built a proof-of-concept quantum generative adversarial network and a simple quantum neural network to accomplish the task of predicting the next four days of prices given fifteen prior days of prices. We evaluate quantum model capacity, convergence characteristics, training efficency, and more on the basis of 7 hyperparameters we tuned.

QGANs blog post: [Quantum Generative Adversarial Networks](https://pavanjayasinha.medium.com/quantum-generative-adversarial-networks-76243d1c6888) 

## File structure
To document the complete journey venturing into the space of QGANs, I've uploaded all the notebooks along the way which guided my decisions along the way.

`data_collection.ipynb ; data_processing.ipynb`: Data fetching and processing into dataset

`A simple qnn to evaluate capacity of ansatzes.ipynb`: Here we evaluate the upper bound of the number of data samples a simple quantum neural network can learn/memorize. 

`A simple quantum neural network to learn DIS stock prices.ipynb`: A naive attempt to see how a simple parameterized quantum circuit could fit to predict DIS stock prices.

`Quantum generative adversarial network.ipynb`: Quantum generative adversarial network used to learn n batches of $DIS data samples. Used for hyperparameter tuning and conclusions.

`Conditional QWGAN.ipynb`: A conditional quantum wasserstein generative adversarial network architecture applied to this task.

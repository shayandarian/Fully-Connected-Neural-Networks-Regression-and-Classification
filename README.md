This project was from 2024.<br/><br/>
Datasets:<br/>
The datasets used for this project were:<br/>
  1. The [Diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) dataset from scikit-learn, which includes baseline variables for diabetes patients, along with a quantitative measure of disease progression one year after baseline.<br/>
  2. The [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) dataset from the UCI Machine Learning Repository, which includes radar returns from the ionosphere collected by an array of high-frequency antennas. The radar returns have been labeled "Good" and "Bad", based on whether they show evidence of structure in the ionosphere.<br/><br/>
Tasks:<br/>
  3. Regression:<br/>
    - Two models were constructed for the Diabetes dataset relating body mass index and average blood pressure to the target measure of disease progression one year after baseline.<br/>
    - The first model used linear regression, and the second model used a simple multi-layer perceptron.<br/>
  3a. Linear Regression Model<br/>
    - The data was split into training and test sets, and then a linear regression model was constructed. For this model, the sci-kit learn library was used.<br/>
  3b. Multi-layer Perceptron<br/>
    - For the multi-layer perceptron, each of the following steps was implemented using "vanilla" Python, meaning that no third-party machine learning libraries or statistical modules were used, with the exception of using sci-kit learn to load and split the dataset, and using Matplotlib to plot learning curves.<br/>
    - A neural network was constructed, consisting of two hidden layers, with two neurons each, using sigmoid activation. The network consisted of a single output layer.<br/>
    - An appropriate loss function was then selected, followed with a manual implementation of backpropagation to find an expression for the gradient of the loss function, with respect to each of the weights.<br/>
    - Gradient descent was implemented to optimize the weights, and a learning curve was plotted to show the training and test loss as a function of the number of epochs.<br/>
    - A comparison was then performed between the performance of this neural network with the performance of the linear regression model on the training and test sets.<br/>
  4. Classification:<br/>
    - A dense feed-forward neural network model was constructed for the Ionosphere dataset classifying radar returns as "Good" or "Bad", based on the measured attributes.<br/>
    - For this model, PyTorch was used as the neural network framework, though Keras was also experimented with.<br/>
    - The Ionosphere dataset was retrieved using the "ucimlrepo" library.<br/>
    - Some other libraries that were used to assist with this model were Pandas, Numpy, Matplotlib, and Scikit-Learn.<br/>
  4a. Constructing the Model<br/>
    - The data was split into training and test sets, and the fully-connected neural network was then trained to classify the returns. Appropriate specifications (i.e. sizes and numbers) for layers, activation functions, optimization, and other hyperparameters were determined and chosen.<br/>
    - Learning curves for the training and test sets were plotted, and the performance of the model was evaluated.<br/>
  4b. Tuning the Model<br/>
    - A variety of regularization techniques were applied to improve the performance of the model on the test set.<br/>
    - The precision and accuracy of the model was then compared to the "Baseline Model Performance for Neural Network Classification" shown on the UCI Machine Learning Repository page.

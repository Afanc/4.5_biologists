[Source Code for task 2a](src/main/py/svm/svm_1.0.py)

[Source Code for task 2b](src/main/py/mlp/mlpMNIST.py)

![Result Plot for task 2b MLP on the MNIST](src/main/py/mlp//figures/MLP_4.5_Biologists__bs_64__lr_0.1__hw_1024__no%20of%20epochs_4.png)
final accuracy (testing) on task 2b : 97.46%

[Source Code for task 2c](src/main/py/cnn/model_task2c.py)
final accuracy (testing) on task 2c : 99.64% 

[Source Code for task 2d CNN on the permutated MNIST](src/main/py/cnn/model_task2d.py)
final accuracy (testing) on task 2d (MLP) : 95.01%

[Source Code for task 2d MLP on the permutated MNIST](src/main/py/mlp/mlpPermutMNIST.py)
final accuracy (testing) on task 2d (CNN) : 99.43% 

![Result Plot for task 2d MLP on the permutated MNIST](src/main/py/mlp//figures/MLP_PermutMNIST_4.5_Biologists__bs_64__lr_0.009__hw_1024__no%20of%20epochs_15.png)

conclusions on 2d :
the network seems to perform as well when the data is permuted than when not. This must because the network sees patterns that are not "visible" to us (they are, we are just bad at spotting them) - this because the data was always permutated in the same way (thus the dataset has as much information as before, it's just ordered differently).
In the end, a network deals with this permutations easily if it's fully connected (it has an input node for each pixel. It performs slightly worse, maybe because some features, like straight lines, become more complex to encode) or if it's a cnn (it sees patterns we don't, and also maybe because it's way more powerful for this task in general). 

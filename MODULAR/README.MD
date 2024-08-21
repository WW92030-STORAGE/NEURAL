# MODULAR

- Implementations of neural networks using a more modular system. Each type of layer is a separate class.
- Layers fundamentally have an input size and an output size. Some layers (e.g. CNN layers) have more variables that depend on each other.
- The base layer class has methods to do elementary operations such as convolution. Said class also serves as an identity layer (that echoes the output from the input). Inherited classes have more diverse behaviors, backpropagations, and algorithms.

# NOTES ON THE NOTES

- These methods are developed in a weird order. For example, the CNN was made first then the basic NN. So if any of the notes sounds weird or out of place then please take this into account.

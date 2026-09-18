from server.vision.model import ConvAutoencoder

# The spectrogram is a fixed-size single-channel image, so the same
# convolutional autoencoder architecture used for surface textures applies
# here too - only the input distribution differs.
SpectrogramAutoencoder = ConvAutoencoder

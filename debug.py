import polycraft_nov_det.data.mnist_loader as mnist_loader


_, _, dataloaders = mnist_loader.torch_mnist(batch_size=1, shuffle=False)

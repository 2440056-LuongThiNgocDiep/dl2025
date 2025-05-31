from CNN import CNN

if __name__ == "__main__":
    train_folder = "mnist_png/training"
    test_folder = "mnist_png/testing"

    cnn = CNN("config.txt")

    train_images, train_labels = cnn.load_from_folder_structure(train_folder)
    test_images, test_labels = cnn.load_from_folder_structure(test_folder)
    cnn.train(train_images, train_labels, epochs=50, lr=0.01)
    cnn.plot_loss()

    accuracy = cnn.evaluate(test_images, test_labels)
    print(f"Accuracy: {accuracy:.4f}%")

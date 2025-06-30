import torch
import numpy as np
import pickle
import os
import torchvision
from scipy.stats import dirichlet

cpath = os.path.dirname(__file__)
NUM_USER = 100       
NUM_CLASSES = 10   
ALPHA = 0.2        


np.random.seed(6)


DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = not False

class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)



def generate_dirichlet_data(num_clients, num_classes, alpha=0.5):
    """
    Generate non-i.i.d. data for federated learning based on Dirichlet distribution.
    
    Args:
        num_clients (int): Number of clients (users).
        num_classes (int): Number of classes (for MNIST, it's 10).
        alpha (float): Hyperparameter controlling the concentration of the Dirichlet distribution.
                        Smaller alpha -> more skewed data distribution across clients.
    
    Returns:
        client_train_data (list): A list containing training data for each client. Each client will have a list of images and labels.
        client_test_data (list): A list containing test data for each client. Each client will have a list of images and labels.
    """
    # Step 1: Generate Dirichlet distributions for each client
    dirichlet_distributions = dirichlet(alpha * np.ones(num_classes), num_clients)
    
    # Step 2: Load MNIST dataset and get the total data size
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    total_data = trainset.data.shape[0]
    
    # Step 3: Calculate how many samples each client will have
    client_samples = np.random.randint(low=1000, high=2000, size=num_clients)
    client_samples = client_samples / client_samples.sum() * total_data  # Normalize to match total data size

    # Prepare MNIST dataset
    train_mnist = ImageDataset(trainset.data, trainset.targets)
    test_mnist = ImageDataset(testset.data, testset.targets)

    images, labels = train_mnist.data.numpy(), train_mnist.target.numpy()
    test_images, test_labels = test_mnist.data.numpy(), test_mnist.target.numpy()

    client_train_data = []
    client_test_data = []
    
    for i in range(num_clients):
        # Get Dirichlet distribution for current client
        client_distribution = dirichlet_distributions[i]
        
        # Calculate how many samples of each class this client should have (for train)
        class_sample_count_train = np.random.multinomial(client_samples[i], client_distribution)
        class_sample_count_test = np.random.multinomial(client_samples[i] // 10, client_distribution)  # Test set size smaller
        
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        # For train data
        for class_idx in range(num_classes):
            class_idx_data = images[labels == class_idx]
            class_sample_data_train = class_idx_data[np.random.choice(class_idx_data.shape[0], class_sample_count_train[class_idx], replace=False)]
            train_images.append(class_sample_data_train)
            train_labels.extend([class_idx] * class_sample_count_train[class_idx])

        # For test data
        for class_idx in range(num_classes):
            class_idx_data = test_images[test_labels == class_idx]
            class_sample_data_test = class_idx_data[np.random.choice(class_idx_data.shape[0], class_sample_count_test[class_idx], replace=False)]
            test_images.append(class_sample_data_test)
            test_labels.extend([class_idx] * class_sample_count_test[class_idx])
        
        client_train_data.append((np.concatenate(train_images), np.array(train_labels)))
        client_test_data.append((np.concatenate(test_images), np.array(test_labels)))
    
    return client_train_data, client_test_data


def main():
    # Generate federated data based on Dirichlet distribution
    client_train_data, client_test_data = generate_dirichlet_data(NUM_USER, NUM_CLASSES, alpha=ALPHA)
    
    # Create dictionary structure to store user data
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Split data for each client and assign it to the dictionary
    for i in range(NUM_USER):
        uname = i
        
        # Assign training data
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': client_train_data[i][0], 'y': client_train_data[i][1]}
        train_data['num_samples'].append(len(client_train_data[i][0]))

        # Assign test data
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': client_test_data[i][0], 'y': client_test_data[i][1]}
        test_data['num_samples'].append(len(client_test_data[i][0]))
    
    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save the data to .pkl files
    train_path = '{}/data/train/all_data_random_niid.pkl'.format(cpath)
    test_path = '{}/data/test/all_data_random_niid.pkl'.format(cpath)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save the user data to pickle files
    with open(train_path, 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_data, outfile)

    print('>>> Data saved to files.')

if __name__ == '__main__':
    main()

# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np  # Import numpy for numerical operations
from skimage.metrics import structural_similarity as ssim  # Import SSIM from skimage
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import wandb  # Import wandb for experiment tracking

# Initialize wandb
wandb.init(project="mnist-classification")  # Initialize a wandb project (update the project name as needed)

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier()  # .to('cpu') by default
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    # Load the trained model state
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    # List of image filenames
    image_filenames = ['img_1.jpg', 'img_2.jpg', 'img_3.jpg']

    for filename in image_filenames:
        # Load and preprocess the image
        img_path = filename  # Image file path
        img = Image.open(img_path).convert('L')  # Convert the image to grayscale
        img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')  # Convert image to tensor and add batch dimension

        # Get the model's prediction
        with torch.no_grad():
            output = clf(img_tensor)
            prediction = torch.argmax(output, dim=1).item()  # Get the predicted class

        # Create a copy of the original image to overlay the prediction
        pred_img = img.copy()

        # Convert images to numpy arrays for MSE and SSIM calculations
        img_np = np.array(img)
        pred_img_np = np.array(pred_img)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((img_np - pred_img_np) ** 2)

        # Calculate Structural Similarity Index (SSIM)
        ssim_index, _ = ssim(img_np, pred_img_np, full=True)

        # Display the target image and the prediction side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Target Image: {filename}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Prediction: {prediction}")
        plt.imshow(pred_img, cmap='gray')
        plt.axis('off')
        
        plt.show()

        # Print the results
        print(f"Filename: {filename}")
        print(f"Prediction: {prediction}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Structural Similarity Index (SSIM): {ssim_index}")

        # Log these metrics to Weights & Biases
        wandb.log({
            "Image Filename": filename,
            "Prediction": prediction,
            "MSE": mse,
            "SSIM": ssim_index
        })

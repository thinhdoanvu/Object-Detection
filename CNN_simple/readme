# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize

# Get data 
data_dir='/content/drive/MyDrive/DemHongCau/classifier/data/train'

#----------------DATA----------------------#
class PlayingCardDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data = ImageFolder(data_dir, transform = transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  @property
  def classes(self):
    return self.data.classes

transform = transforms.Compose([Resize((224, 224)), ToTensor()])
dataset = PlayingCardDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3)),  # Change 1 to 3 for RGB images
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(224-6)*(224-6), 53) 
        )
    def forward(self, x): 
      return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to('cpu')                 #cuda
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow 
if __name__ == "__main__": 
  for epoch in range(10): # train for 10 epochs
    for batch in data_loader: 
      X,y = batch 
      yhat = clf(X) 
      loss = loss_fn(yhat, y) 

      # Apply backprop 
      opt.zero_grad()
      loss.backward() 
      opt.step() 
    print(f"Epoch:{epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f: 
      save(clf.state_dict(), f) 

# predict
    #with open('model_state.pt', 'rb') as f: 
    #  clf.load_state_dict(load(f))  

    #img = Image.open('img_3.jpg') 
    #img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    #print(torch.argmax(clf(img_tensor)))

# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.

### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name:Franklin raj G
### Register Number:212223230058

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
        
    # Plot training and validation loss
    print("Name:Prem Kumar G")
    print("Register Number:212223230158")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Train the model
train_model(model,train_loader,test_loader,num_epochs=10)
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="969" height="673" alt="image" src="https://github.com/user-attachments/assets/43da206a-cf13-4b6e-9b67-c0920b4cd4ea" />



## Confusion Matrix
<img width="648" height="519" alt="image" src="https://github.com/user-attachments/assets/dd17a1be-6df6-4d5a-8322-2a83d46f6423" />


## Classification Report
<img width="497" height="228" alt="image" src="https://github.com/user-attachments/assets/3d98eabd-6a62-47c1-96ed-b6383ac5c4bb" />


### New Sample Data Prediction
<img width="404" height="410" alt="image" src="https://github.com/user-attachments/assets/3ea92bb2-15b5-4868-9256-9196e088d8eb" />


## RESULT
Thus, the image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.

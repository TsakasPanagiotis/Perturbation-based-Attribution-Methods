from tqdm import tqdm
import torch

# def create_subset(args, dataset):
#     subset = []
#     class_counts = [0] * 10
#     for image, label in dataset:
#         if class_counts[label] < args.num_images_per_class:
#             subset.append((image, label))
#             class_counts[label] += 1
#         if sum(class_counts) >= args.num_images_per_class * 10:
#             break
#     return subset

def evaluate_model(args,model,dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

def train_model(args,model,trainloader):

    train_loss = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = args.num_epochs

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs,labels in tqdm(trainloader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs).squeeze(dim=1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * labels.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/len(trainloader.dataset))
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss/len(trainloader.dataset)}")

    torch.save(model.state_dict(), "./vgg_" + str(args.experiment) + "_model.pt")
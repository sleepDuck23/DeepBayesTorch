import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def model_eval(model, dataloader, device, return_pred=False):
    """
    Compute the accuracy of a PyTorch model on some data.
    
    :param model: PyTorch model to evaluate.
    :param dataloader: DataLoader containing the test dataset (inputs and labels).
    :param device: Device ('cpu' or 'cuda') on which the evaluation is performed.
    :param return_pred: If True, also returns the predictions.
    :return: Accuracy as a float, and optionally the predictions as a NumPy array.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Move data to the correct device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted classes
            
            # Accumulate correct predictions and total samples
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Collect predictions if required
            if return_pred:
                all_preds.append(outputs.cpu().numpy())
    
    # Compute accuracy
    accuracy = correct / total
    
    if return_pred:
        all_preds = np.concatenate(all_preds, axis=0)
        return accuracy, all_preds
    else:
        return accuracy

# Example usage:
# Assuming X_test and Y_test are NumPy arrays and batch_size is defined.
# dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long))
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# accuracy, predictions = model_eval(model, dataloader, device='cuda', return_pred=True)
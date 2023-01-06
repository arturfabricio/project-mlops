import wandb
from train_model import train

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')

def compute_validation_metrics(model, dataloader):
    ''' The prettiest function 🙂'''
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader)
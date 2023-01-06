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

wandb.init()

learning_rate  =  wandb.config.lr
batch_size = wandb.config.batch_size
epochs = wandb.config.epochs

function = train(train_dataset, test_dataset,'resnet18', batch_size, epochs, learning_rate)

wandb.agent(sweep_id, function=function, count=4)



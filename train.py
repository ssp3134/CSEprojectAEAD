import utils
import models 
import torch 
from time import time
import matplotlib.pyplot as plt 

WINDOW_SIZE = 60 
BATCH_SIZE = 64
DATA_POINTS = 1e7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 40

auto_encoder = models.AEArch(window_size=WINDOW_SIZE)
model_name = auto_encoder.__class__.__name__
model = auto_encoder.to(device=DEVICE)
model = torch.compile(model)

optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=1e-3)
loss_criterion = torch.nn.MSELoss()

train_loader, val_loader = utils.get_time_series_data(
    DATA_POINTS, 
    batch_size=BATCH_SIZE, 
    val_size=0.2
)
train_size = len(train_loader)
val_size = len(val_loader)
train_epoch_loss, val_epoch_loss = [], []

start_time = time()
# early_stopping = utils.EarlyStopping(patience=5, verbose=True)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}")
    epoch_start_time = time()
    running_loss = 0
    for i, batch in enumerate(train_loader):
        batch = batch.to(device=DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE):
            pred = auto_encoder(batch)
            loss = loss_criterion(pred, batch)
            running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            duration = (time() - epoch_start_time)/(i+1)
            print(f"[{i+1}/{train_size}]-> Loss: {loss:.4f}, Duration: {duration:.4f} s/batch")
    mean_running_loss = running_loss/train_size
    train_epoch_loss.append(mean_running_loss)
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device=DEVICE)
            with torch.amp.autocast(device_type=DEVICE):
                val_pred = model(val_batch)
                val_batch_loss = loss_criterion(val_pred, val_batch)
            val_loss += val_batch_loss.item()
    mean_val_loss = val_loss/val_size
    val_epoch_loss.append(mean_val_loss)
    print(f"Training Loss: {mean_running_loss:.4f}; Validation Loss: {mean_val_loss:.4f}")  
    
    # early_stopping(mean_val_loss)
    # if early_stopping.early_stop:
    #     print(f"Stopping training early at epoch {epoch+1}.")
    #     break

plt.plot(train_epoch_loss, label='Training Loss')
plt.plot(val_epoch_loss, label='Validation Loss')
plt.title("Loss Over Epochs")
plt.legend()
plt.savefig(f"{model_name}-{start_time}.jpeg")
torch.save(model.state_dict(), f"{model_name}-{start_time}-{epoch+1}.pt")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch 
import utils  
import models
import matplotlib
matplotlib.use('TkAgg')

def get_model(window_size, device):
    model = models.AEArch(window_size=window_size)
    model = torch.compile(model)
    state_dict = torch.load("AEArch-1726948317.1064928-50.pt", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)

def predict(model, device, threshold):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device):  
            reconstructed = model(data)
            errors = torch.mean((reconstructed - data) ** 2, dim=1)
    max_threshold = errors.mean() + threshold * errors.std()
    min_threshold = errors.mean() - threshold * errors.std()
    anomalies = (errors > max_threshold) | (errors < min_threshold)
    anomaly_indices = torch.where(anomalies)[0].cpu().numpy()
    normal_indices = torch.where(~anomalies)[0].cpu().numpy()
    return reconstructed.cpu().numpy(), anomaly_indices, normal_indices

def plot_data(input_data, reconstructed_data, index, title=""):
    ax.clear()
    ax.plot(input_data[index], label="Input Data", linestyle='--', color='blue')
    ax.plot(reconstructed_data[index], label="Reconstructed Data", color='red')
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.canvas.draw()
    
if __name__ == "__main__":
    WINDOW_SIZE = 60
    NUM_TENSORS = 100 
    THRESHOLD_FACTOR = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    
    data = utils.generate_random_time_series(num=NUM_TENSORS, window_size=WINDOW_SIZE, device=DEVICE)
    model = get_model(window_size=WINDOW_SIZE, device=DEVICE)
    reconst, anomaly_indices, normal_indices = predict(model, device=DEVICE, threshold=THRESHOLD_FACTOR)
    data = data.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)

    anomaly_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])  
    normal_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])  

    anomaly_button = Button(anomaly_button_ax, 'Plot Anomaly')
    normal_button = Button(normal_button_ax, 'Plot Normal')

    current_plot_index = [0]
    
    
    def plot_anomaly(event):
        if len(anomaly_indices) > 0:
            idx = anomaly_indices[current_plot_index[0] % len(anomaly_indices)]
            plot_data(data, reconst, idx, title=f"Anomaly Data at Index {idx}")
            current_plot_index[0] += 1

    def plot_normal(event):
        if len(normal_indices) > 0:
            idx = normal_indices[current_plot_index[0] % len(normal_indices)]
            plot_data(data, reconst, idx, title=f"Normal Data at Index {idx}")
            current_plot_index[0] += 1
    
    anomaly_button.on_clicked(plot_anomaly)
    normal_button.on_clicked(plot_normal)

    plt.show()
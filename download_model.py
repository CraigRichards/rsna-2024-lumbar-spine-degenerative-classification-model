import torch

# URL of the pre-trained model weights
model_url = 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth'

# Download the weights
state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)

save_path = 'models/efficientnet_v2_s-dd5fe13b.pth'

# Save the downloaded weights
torch.save(state_dict, save_path)
print(f"Model weights saved to {save_path}")

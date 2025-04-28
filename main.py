import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from homework4 import CNP # didn't change the original CNP class

class TrajDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the trajectory data
        traj = self.data[idx]
        
        t = traj[:, 0:1]  # time
        ey = traj[:, 1]  # end-effector y
        ez = traj[:, 2]  # end-effector z
        oy = traj[:, 3]  # object y
        oz = traj[:, 4]  # object z
        h = traj[:, 5:6]   # object height
        
        # Stack targets ee_pos and obj_pos
        targets = traj[:, 1:5]
        
        # Sample context and target points
        n_points = len(traj)
        n_context = np.random.randint(1, 50)
        n_target = np.random.randint(1, 50)
        
        # Sample indices for context and target
        context_idx = np.random.choice(n_points, n_context, replace=False)
        target_idx = np.random.choice(n_points, n_target, replace=False)
        
        context_x = t[context_idx] # fancy indexing :)
        context_y = targets[context_idx]
        context_h = h[context_idx]

        target_x = t[target_idx]
        target_y = targets[target_idx]
        target_h = h[target_idx]
        

        observation = np.concatenate([context_x, context_h, context_y], axis=1)# (ey, ez, oy, oz)

        target_query = np.concatenate([target_x,target_h], axis=1)        
        return {
            'observation': torch.tensor(observation, dtype=torch.float32),
            'target': torch.tensor(target_query, dtype=torch.float32),
            'target_y': torch.tensor(target_y, dtype=torch.float32)
        }

def collate_fn(batch):
    # Find max 
    max_obs_len = max(item['observation'].shape[0] for item in batch)
    max_target_len = max(item['target'].shape[0] for item in batch)
    
    # Create tensors with padding
    batch_size = len(batch)
    obs_dim = batch[0]['observation'].shape[1]
    target_dim = batch[0]['target'].shape[1]
    target_y_dim = batch[0]['target_y'].shape[1]
    
    observation_batch = torch.zeros(batch_size, max_obs_len, obs_dim) # zero padding 
    target_batch = torch.zeros(batch_size, max_target_len, target_dim)
    target_y_batch = torch.zeros(batch_size, max_target_len, target_y_dim)
    
    # Create mask tensors
    observation_mask = torch.zeros(batch_size, max_obs_len)
    target_mask = torch.zeros(batch_size, max_target_len)
    
    for i, item in enumerate(batch):
        obs_len = item['observation'].shape[0]
        target_len = item['target'].shape[0]
        # Fill data
        observation_batch[i, :obs_len] = item['observation']
        target_batch[i, :target_len] = item['target']
        target_y_batch[i, :target_len] = item['target_y']
        # Fill masks
        observation_mask[i, :obs_len] = 1
        target_mask[i, :target_len] = 1
    return {
        'observation': observation_batch,
        'target': target_batch,
        'target_y': target_y_batch,
        'observation_mask': observation_mask,
        'target_mask': target_mask
    }

def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            observation = batch['observation']
            target = batch['target']
            target_y = batch['target_y']
            observation_mask = batch['observation_mask']
            target_mask = batch['target_mask']
            
            optimizer.zero_grad()
            loss = model.nll_loss(observation, target, target_y, observation_mask, target_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                observation = batch['observation']
                target = batch['target']
                target_y = batch['target_y']
                observation_mask = batch['observation_mask']
                target_mask = batch['target_mask']
                
                loss = model.nll_loss(observation, target, target_y, observation_mask, target_mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnmp_model.pt')
            print(f'Model saved with val loss: {val_loss:.4f}')
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    return train_losses, val_losses

def main():
    data_path = 'dataset_5000.npz'
    data = np.load(data_path, allow_pickle=True)['data']
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Create datasets
    train_dataset = TrajDataset(train_data)
    val_dataset = TrajDataset(val_data)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    in_shape = (2, 4)  # (query_dim, target_dim)
    hidden_size = 128
    num_hidden_layers = 3
    
    model = CNP(in_shape, hidden_size, num_hidden_layers)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Train the model
    num_epochs = 500
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, num_epochs)
    print('done')

if __name__ == "__main__":
    main()
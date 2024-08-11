import torch


def train(recommendation_model, train_dataset, train_loader, loss_fn, optimizer, model_path, device, epochs=2):
    total_loss = 0
    log_step = 100
    losses = []
    
    print(f'Training on size: {len(train_dataset)}')
    recommendation_model.train()
    
    for epoch_i in range(epochs):
        step_count = 0
        for i, train_data in enumerate(train_loader):
            users = train_data["users"].to(device)
            movies = train_data["movies"].to(device)
    
            output = recommendation_model(users, movies)
            output = output.squeeze()
            
            ratings = train_data["ratings"].to(torch.float32).to(device)
    
            loss = loss_fn(output, ratings)
            total_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            step_count += len(train_data["users"])
    
            if (step_count % log_step == 0 or i == len(train_loader) - 1):
                avg_loss = (total_loss / log_step)
                print(f"epoch {epoch_i} loss at step {step_count} is {avg_loss}")
                losses.append(avg_loss)
                total_loss = 0

    torch.save(recommendation_model.state_dict(), model_path)
    print(f'Model saved to: {model_path}')

    return losses
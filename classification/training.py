import torch
from tqdm import tqdm
from time import perf_counter


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    learning_rate,
    epochs,
    checkpoint_dir
):
    """
    Train and validate a model for a specified number of epochs.
    The latest training state is saved after each epoch as the last
        checkpoint. A separate best checkpoint is saved whenever validation
        loss improves.

    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): DataLoader for the training dataset
        validation_loader (DataLoader): DataLoader for the validation dataset
        device (torch.device): Device on which to run the model
        learning_rate (float): Learning rate used by the optimizer
        epochs (int): Number of training epochs

    Returns:
        dict: Training history: training and validation loss and accuracy
            for each epoch.
    """
    best_validation_loss = float('inf')
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': [],
    }

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    for epoch in range(epochs):
        start = perf_counter()
        train_loss, train_accuracy = train_epoch(
            model, train_loader, device, criterion, optimizer
        )
        validation_loss, validation_accuracy = validate_epoch(
            model, validation_loader, device, criterion
        )
        elapsed = perf_counter() - start

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f'Training loss: {train_loss:.4f}    | '
              f'Training accuracy: {train_accuracy:.4f}')
        print(f'Validation loss: {validation_loss:.4f}  | '
              f'Validation accuracy: {validation_accuracy:.4f}')
        print(f'Time: {elapsed:.1f}')

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        save_checkpoint(
            model,
            epoch,
            optimizer,
            validation_loss,
            validation_accuracy,
            checkpoint_dir / 'last.pt'
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                model,
                epoch,
                optimizer,
                validation_loss,
                validation_accuracy,
                checkpoint_dir / 'best.pt'
            )

    return history


def train_epoch(model, data_loader, device, criterion, optimizer):
    """
    Train a model for one epoch

    Args:
        model (torch.nn.Module): Model to train
        data_loader (torch.utils.data.DataLoader): DataLoader containing
            training data
        device (torch.device): Device on which to perform computations
        criterion (torch.nn.Module): Loss function for model evaluation
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters

    Returns:
        tuple[float, float]: Mean training loss and training accuracy for epoch
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(data_loader, desc='Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        # item() needed to convert out of tensor
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / total_samples
    train_accuracy = total_correct / total_samples
    return train_loss, train_accuracy


def validate_epoch(model, data_loader, device, criterion):
    """
    Evaluate a model for one epoch

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): DataLoader containing
            validation data
        device (torch.device): Device on which to perform computations
        criterion (torch.nn.Module): Loss function used for model evaluation

    Returns:
        tuple[float, float]: Mean validation loss and validation accuracy for
            epoch
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validation', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    validation_loss = total_loss / total_samples
    validation_accuracy = total_correct / total_samples
    return validation_loss, validation_accuracy


def save_checkpoint(
    model, epoch, optimizer, validation_loss, validation_accuracy, path
):
    """
    Save model training state to a checkpoint file

    Args:
        model (torch.nn.Module): Model whose state is saved
        epoch (int): Epoch at which checkpoint is saved
        optimizer (torch.optim.Optimizer): Optimizer whose state is saved
        validation_loss (float): Validation loss for current epoch
        validation_accuracy (float): Validation accuracy for current epoch
        path (str | Path): Path where checkpoint is saved
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
    }
    torch.save(checkpoint, path)

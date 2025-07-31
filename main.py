import argparse
import torch
import torch.nn as nn

from src.data.dataset_loader import (
    MNIST_loader, FMNIST_loader, CIFAR_loader,
    Speech_loader, PIMA_loader
)
from src.models.mnist_net import AE_MNIST
from src.models.cifar_net import AE_CIFAR
from src.models.pima_net import AE_PIMA
from src.models.speech_net import AE_Speech
from src.core.gamma_tune import tune_gamma
from src.core.trainer import DASVDD_trainer
from src.core.tester import DASVDD_test


def get_loader_and_model(dataset_name, train_batch, test_batch, cls):
    if dataset_name == "MNIST":
        train_loader, test_loader, labels = MNIST_loader(train_batch, test_batch, cls)
        model = AE_MNIST(input_shape=28 * 28)
        input_shape = 28 * 28
        code_size = 256
    elif dataset_name == "FMNIST":
        train_loader, test_loader, labels = FMNIST_loader(train_batch, test_batch, cls)
        model = AE_MNIST(input_shape=28 * 28)
        input_shape = 28 * 28
        code_size = 256
    elif dataset_name == "CIFAR":
        train_loader, test_loader, labels = CIFAR_loader(train_batch, test_batch, cls)
        model = AE_CIFAR(input_shape=32 * 32 * 3)
        input_shape = 32 * 32 * 3
        code_size = 256
    elif dataset_name == "Speech":
        train_loader, test_loader, labels = Speech_loader(train_batch, test_batch)
        model = AE_Speech(input_shape=400)
        input_shape = 400
        code_size = model.code_size

    elif dataset_name == "PIMA":
        train_loader, test_loader, labels = PIMA_loader(train_batch, test_batch)
        model = AE_PIMA(input_shape=8)
        input_shape = 8
        code_size = model.code_size
    else:
        raise ValueError("Unsupported dataset")

    return train_loader, test_loader, labels, model, input_shape, code_size


def main():
    parser = argparse.ArgumentParser(description="Run DASVDD on different datasets.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["MNIST", "FMNIST", "CIFAR", "Speech", "PIMA"],
                        help="Dataset to use for training and testing.")
    parser.add_argument("--target_class", type=int, default=0,
                        help="Target class for anomaly detection (ignored for Speech and PIMA).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=200, help="Training batch size.")

    args = parser.parse_args()

    print(f"\n Starting DASVDD on the {args.dataset} dataset...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, labels, model, input_shape, code_size = get_loader_and_model(
        args.dataset, args.batch_size, test_batch=1, cls=args.target_class
    )

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    C = torch.randn(code_size, device=device, requires_grad=True)
    update_center = torch.optim.Adagrad([C], lr=1, lr_decay=0.01)

    print("Tuning gamma...")
    Gamma = tune_gamma(model.__class__, input_shape, criterion, train_loader, device=device, T=10)

    print("Training DASVDD...")
    DASVDD_trainer(
        model, input_shape, code_size, C, train_loader, optimizer, update_center,
        criterion, Gamma, device=device, num_epochs=args.epochs, K=0.9, verbosity=1
    )

    print("\nEvaluating...")
    results = DASVDD_test(model, C, input_shape, Gamma, test_loader, labels, criterion, C)
    print(f"Results: {results:.2f}")


if __name__ == "__main__":
    main()

from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


def svm_loss_naive(scores: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = None
    "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    
    "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    return loss


def svm_loss_vectorized(scores: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss = None
    "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    
    "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    return loss


def cross_entropy_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = None
    "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"

    "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    return loss


def construct_batches(X, y, batch_size):
    X_batch, y_batch = None, None
    "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"

    "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
    return X_batch, y_batch


class LinearBaseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, W_dtype: torch.dtype = torch.float32, device: str = "cpu"):
        super().__init__()
        W = 0.0001 * torch.randn(
            output_dim, input_dim + 1, dtype=W_dtype, device=device
        )
        self.register_parameter("W", nn.Parameter(W))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_pred = None
        "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"

        "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
        return out_pred

    @abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor, mode: str = "vectorized",
             return_scores: bool = False) -> torch.Tensor:
        raise NotImplementedError("TODO: Implement")

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def train_loop(
            self, X: torch.Tensor, y: torch.Tensor, n_iter: int = 1500, batch_size: int = 64, lr: float = 1e-3,
            reg: float = 0.0, verbose: bool = True
    ) -> list:
        if verbose:
            print(f"Training on {len(X)} samples for {n_iter} iterations")
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=reg)
        loss = None
        loss_history = []
        for i in range(n_iter):
            X_batch, y_batch = construct_batches(X, y, batch_size=batch_size)
            "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"

            "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
            if i % 100 == 0 and verbose:
                print(f"Iteration {i} / {n_iter}: Loss = {loss.item()}")
        return loss_history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        cls_pred = torch.empty(len(X), dtype=torch.long, device=X.device)
        "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"

        "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"
        return cls_pred


class SVM(LinearBaseModel):

    def __init__(
            self, input_dim: int, output_dim: int, delta: float = 1.0, device: str = "cpu",
            W_dtype: torch.dtype = torch.float32
    ):
        super().__init__(input_dim, output_dim, W_dtype, device)
        self.delta = delta

    def loss(self, x: torch.Tensor, y: torch.Tensor, mode: str = "vectorized",
             return_scores: bool = False) -> torch.Tensor | tuple:
        if mode == "naive":
            scores = self(x)
            loss = svm_loss_naive(scores, y, self.delta)
        elif mode == "vectorized":
            scores = self(x)
            loss = svm_loss_vectorized(scores, y, self.delta)
        else:
            raise ValueError(f"Invalid mode {mode}")
        if return_scores:
            return scores, loss
        return loss

    def __str__(self):
        return f"svm_classifier"


class SoftmaxClassifier(LinearBaseModel):

    def __init__(self, input_dim: int, output_dim: int, device: str = "cpu", W_dtype: torch.dtype = torch.float32):
        super().__init__(input_dim, output_dim, W_dtype, device)

    def loss(self, x: torch.Tensor, y: torch.Tensor, mode: str = "vectorized",
             return_scores: bool = False) -> torch.Tensor | tuple:
        assert mode == "vectorized", f"Invalid mode {mode}"
        scores = self(x)
        loss = cross_entropy_loss(scores, y)
        if return_scores:
            return scores, loss
        return loss

    def __str__(self):
        return f"softmax_classifier"
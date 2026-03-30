'''
This script implements a dirichlet calibration class
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class which implements Dirichlet Calibration (multiclass) as:
Fit W,b on a calibration set by minimizing NLL.
Predict calibrated scores on new predictions
"""
class DirichletCalibrator(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-12):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.linear = nn.Linear(num_classes, num_classes, bias=True)

        # Optional: initialize close to identity mapping for stability
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(num_classes))
            self.linear.bias.zero_()


    """
    forward pass:
    call with:
        logits: torch.Tensor --> [N, K] unnormalized logits from the base model
    returns: 
        q: torch.Tensor --> calibrated probabilities q: [N, K]
    """
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        x = torch.log(p.clamp_min(self.eps))
        q = F.softmax(self.linear(x), dim=-1)
        return q

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        self.eval()
        preds = self.forward(logits)
        return preds.cpu().detach().tolist()


    """
    Fit on calibration logits/labels.
    Call with:
        logits_cal: torch.Tensor --> [N, K] logits from frozen base model
        y_cal: torch.Tensor --> [N] int64 labels in [0..K-1]
    """
    def fit(
        self,
        logits_cal: torch.Tensor,
        y_cal: torch.Tensor,
        *,
        lr: float = 0.1,
        max_iter: int = 200,
        weight_decay: float = 0.0,
        use_lbfgs: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.to(device)
        logits_cal = logits_cal.to(device)
        y_cal = y_cal.to(device).long()

        self.train()

        # We optimize NLL of calibrated probs, i.e. CE on calibrated logits.
        # Note: our module outputs probs, so we use negative log prob of true class.
        def nll_loss():
            q = self.forward(logits_cal)  # [N,K]
            loss = F.nll_loss(torch.log(q.clamp_min(self.eps)), y_cal)
            if weight_decay > 0:
                # L2 regularization on weights (and optionally bias)
                loss = loss + weight_decay * (self.linear.weight.pow(2).sum())
            return loss

        if use_lbfgs:
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=lr,
                max_iter=max_iter,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = nll_loss()
                loss.backward()
                return loss

            optimizer.step(closure)

        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
            for _ in range(max_iter):
                optimizer.zero_grad(set_to_none=True)
                loss = nll_loss()
                loss.backward()
                optimizer.step()

        self.eval()
        return self




"""
functuon to fit calibrator
call with: 
    y_cal: torch.tensor, --> targets 
    logits_cal: torch.tensor --> logits
"""
def get_cal(y_cal: torch.tensor, 
                     logits_cal: torch.tensor):
    K = logits_cal.size(1)
    cal = DirichletCalibrator(num_classes=K)
    cal.fit(
        logits_cal, y_cal,
        lr=0.5,
        max_iter=300,
        weight_decay=1e-4,   
        use_lbfgs=True,
        device="cpu",    
    )
    return cal
 

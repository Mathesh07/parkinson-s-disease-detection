"""Evidential Deep Learning & Dirichlet Subjective Logic Core Module.

Implements the Trusted Multi-View Classification (TMC) / Evidential Deep Learning (EDL)
mathematical formulation for classification under uncertainty (Han et al., ICLR 2021; Sensoy et al., NeurIPS 2018).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_from_evidence(
    evidence: torch.Tensor,
    num_classes: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert non-negative evidence into Dirichlet distribution parameters and uncertainty.
    
    Mathematical Formulation:
        alpha_k = e_k + 1
        S = sum_{k=1}^K alpha_k
        P_k = alpha_k / S
        u = K / S
        
    Subjective Logic Identity:
        sum_{k=1}^K (e_k / S) + u = 1.0 (Belief + Uncertainty = 1)

    Args:
        evidence: Non-negative tensor of shape (..., num_classes), e_k >= 0.
        num_classes: Number of target classes K (default=2 for HC/PD).

    Returns:
        alpha: Dirichlet concentration parameters of shape (..., num_classes), alpha_k >= 1.
        S: Total Dirichlet strength of shape (..., 1), S >= K.
        probabilities: Expected class probabilities of shape (..., num_classes), sum(P) = 1.
        uncertainty: Total vacuity / predictive uncertainty of shape (..., 1), 0 < u <= 1.
    """
    # Ensure evidence is non-negative
    evidence = F.relu(evidence)
    
    # 1. Dirichlet concentration parameter
    alpha = evidence + 1.0  # (..., K)
    
    # 2. Total Dirichlet strength
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (..., 1)
    
    # 3. Expected class probabilities
    probabilities = alpha / S  # (..., K)
    
    # 4. Vacuity / Predictive Uncertainty (K / S)
    uncertainty = float(num_classes) / S  # (..., 1)
    
    return alpha, S, probabilities, uncertainty


def kl_divergence_dirichlet(
    alpha: torch.Tensor,
    num_classes: int = 2
) -> torch.Tensor:
    """
    Calculate the KL divergence between a Dirichlet distribution Dir(alpha) and the uniform Dirichlet prior Dir(1).
    
    Formula:
        KL(Dir(alpha) || Dir(1)) = ln( Gamma(S) / (Gamma(K) * prod Gamma(alpha_k)) )
                                  + sum_{k=1}^K (alpha_k - 1) * [ psi(alpha_k) - psi(S) ]
                                  
    In log space (numerically stable via torch.lgamma & torch.digamma):
        lgamma(S) - lgamma(K) - sum(lgamma(alpha_k)) + sum((alpha_k - 1) * (psi(alpha_k) - psi(S)))

    Args:
        alpha: Dirichlet parameters of shape (batch_size, num_classes), alpha >= 1.
        num_classes: K classes (default=2).

    Returns:
        kl_div: Tensor of shape (batch_size,) containing per-sample KL divergence >= 0.
    """
    K = float(num_classes)
    ones = torch.ones_like(alpha)
    
    # Sum of alpha
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)
    
    # Log-gamma terms
    first_term = (
        torch.lgamma(S.squeeze(-1))
        - torch.lgamma(torch.tensor(K, device=alpha.device, dtype=alpha.dtype))
        - torch.sum(torch.lgamma(alpha), dim=-1)
    )  # (B,)
    
    # Digamma difference terms
    digamma_S = torch.digamma(S)  # (B, 1)
    digamma_alpha = torch.digamma(alpha)  # (B, K)
    second_term = torch.sum((alpha - ones) * (digamma_alpha - digamma_S), dim=-1)  # (B,)
    
    kl = first_term + second_term
    return F.relu(kl)  # Clamp against any minor negative float imprecision


class EvidentialLoss(nn.Module):
    """
    Evidential Classification Loss (Trusted Multi-View Classification / EDL).
    
    Combines:
    1. Expected Cross-Entropy (ACE) with Digamma function:
       L_ace = sum_{k=1}^K y_k * ( psi(S) - psi(alpha_k) )
       
    2. KL Divergence Regularization on misleading evidence:
       alpha_tilde = y + (1 - y) * alpha
       L_kl = KL(Dir(alpha_tilde) || Dir(1))
       
    3. Total Loss with Annealing:
       L_total = L_ace + lambda_t * L_kl
       where lambda_t = min(1.0, current_epoch / annealing_epochs)
    """

    def __init__(
        self,
        num_classes: int = 2,
        annealing_epochs: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "ace"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.class_weights = class_weights
        self.loss_type = loss_type

    def forward(
        self,
        evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute evidential loss.
        
        Args:
            evidence: Non-negative evidence tensor of shape (batch_size, num_classes) or (batch_size,).
            target: Ground truth class indices of shape (batch_size,) or one-hot (batch_size, num_classes).
            epoch: Current training epoch (0-indexed or 1-indexed) for KL annealing.
            
        Returns:
            loss: Scalar PyTorch loss tensor for backpropagation.
            loss_dict: Dictionary with decomposed loss components ('loss', 'ace_loss', 'kl_loss', 'lambda_t').
        """
        # Ensure 2D shape (B, K)
        if evidence.dim() == 1:
            evidence = evidence.unsqueeze(-1)
            
        batch_size = evidence.size(0)
        device = evidence.device
        
        # 1. Convert integer targets to one-hot if needed
        if target.dim() == 1 or target.size(-1) == 1:
            target_indices = target.squeeze().long()
            y_one_hot = F.one_hot(target_indices, num_classes=self.num_classes).float()
        else:
            y_one_hot = target.float()

        # 2. Dirichlet Parameters
        alpha = evidence + 1.0  # (B, K)
        S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)

        # 3. Fitting Loss: Expected Cross-Entropy (ACE)
        # L_ace = sum( y_k * ( psi(S) - psi(alpha_k) ) )
        psi_S = torch.digamma(S)  # (B, 1)
        psi_alpha = torch.digamma(alpha)  # (B, K)
        ace_per_sample = torch.sum(y_one_hot * (psi_S - psi_alpha), dim=-1)  # (B,)

        # Apply class weights if specified
        if self.class_weights is not None:
            weights = self.class_weights.to(device)
            # Sample weight based on true class
            sample_weights = torch.sum(y_one_hot * weights, dim=-1)
            ace_per_sample = ace_per_sample * sample_weights

        ace_loss = torch.mean(ace_per_sample)

        # 4. KL Regularization on Misleading Evidence
        # alpha_tilde removes evidence from the true class: alpha_tilde_k = y_k + (1 - y_k) * alpha_k
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha  # (B, K)
        kl_per_sample = kl_divergence_dirichlet(alpha_tilde, num_classes=self.num_classes)  # (B,)
        kl_loss = torch.mean(kl_per_sample)

        # 5. Annealing Coefficient lambda_t
        if self.annealing_epochs > 0:
            lambda_t = min(1.0, float(epoch) / float(self.annealing_epochs))
        else:
            lambda_t = 1.0

        total_loss = ace_loss + lambda_t * kl_loss

        return total_loss, {
            "loss": float(total_loss.item()),
            "ace_loss": float(ace_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "lambda_t": float(lambda_t)
        }

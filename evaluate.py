import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import ToPILImage

def calculate_inception_score(generated_images, batch_size=64, splits=10):
    """
    Calculate the Inception Score of generated images.

    Args:
        generated_images (torch.Tensor): Generated images as a tensor of shape (N, C, H, W).
        batch_size (int, optional): Batch size for feeding images to the Inception network.
        splits (int, optional): Number of splits to divide the generated images for evaluation.

    Returns:
        float: Inception Score.
    """
    # Load the Inception-v3 model pretrained on ImageNet
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = nn.Sequential(*list(inception_model.children())[:-1]).eval().to(device)

    # Resize and normalize generated images
    generated_images = F.interpolate(generated_images, size=(299, 299), mode='bilinear', align_corners=True)
    generated_images = (generated_images + 1) / 2  # Unnormalize to [0, 1]

    # Calculate the activations of generated images using Inception-v3
    activations = []
    num_images = generated_images.shape[0]
    num_batches = (num_images - 1) // batch_size + 1
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            batch = generated_images[start_idx:end_idx].to(device)
            activations.append(inception_model(batch).squeeze())

    activations = torch.cat(activations, dim=0)

    # Calculate the marginal distribution (p(y)) by averaging the softmax outputs
    marginal_probs = torch.softmax(activations, dim=1).mean(dim=0)

    # Calculate the conditional distribution (p(y|x)) by taking the softmax of the max logits
    conditional_probs = torch.softmax(activations, dim=1)
    conditional_probs_max = conditional_probs.max(dim=1)[0]
    
    # Calculate the Inception Score
    kl_divs = conditional_probs * (torch.log(conditional_probs) - torch.log(marginal_probs))
    kl_divs = kl_divs.sum(dim=1)
    kl_divs_mean = kl_divs.mean()

    # Calculate the Inception Score for each split
    split_scores = []
    for _ in range(splits):
        perm = torch.randperm(num_images)
        subset = activations[perm[:batch_size]]
        subset_probs = torch.softmax(subset, dim=1)
        subset_probs_max = subset_probs.max(dim=1)[0]
        kl_divs_subset = subset_probs * (torch.log(subset_probs) - torch.log(marginal_probs))
        kl_divs_subset = kl_divs_subset.sum(dim=1)
        kl_divs_subset_mean = kl_divs_subset.mean()
        split_scores.append(torch.exp(kl_divs_subset_mean - kl_divs_mean))

    # Calculate the Inception Score
    inception_score = torch.stack(split_scores).mean().item()
    return inception_score

def torch_covariance(x, rowvar=False, bias=False):
    """
    Compute the covariance matrix of a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D).
        rowvar (bool, optional): If True, each row represents a variable; otherwise, each column represents a variable.
        bias (bool, optional): If True, the biased covariance matrix will be calculated.

    Returns:
        torch.Tensor: Covariance matrix of shape (D, D) if rowvar is False, or (N, N) if rowvar is True.
    """
    # Calculate the mean along the specified axis
    mean = torch.mean(x, dim=0, keepdim=True) if rowvar else torch.mean(x, dim=1, keepdim=True)

    # Subtract the mean from the input tensor
    x_centered = x - mean

    # Compute the covariance matrix
    if rowvar:
        cov = torch.mm(x_centered, x_centered.t())
    else:
        cov = torch.mm(x_centered.t(), x_centered)

    # Normalize the covariance matrix
    factor = x.shape[0] - int(bias)
    cov /= factor

    return cov


def calculate_fid(real_images, generated_images, batch_size=64):
    """
    Calculate the Frechet Inception Distance (FID) between real and generated images.

    Args:
        real_images (torch.Tensor): Real images as a tensor of shape (N, C, H, W).
        generated_images (torch.Tensor): Generated images as a tensor of shape (N, C, H, W).
        batch_size (int, optional): Batch size for feeding images to the Inception network.

    Returns:
        float: Frechet Inception Distance (FID).
    """
    # Load the Inception-v3 model pretrained on ImageNet
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = nn.Sequential(*list(inception_model.children())[:-1]).eval().to(device)

    # Resize and normalize real and generated images
    real_images = F.interpolate(real_images, size=(299, 299), mode='bilinear', align_corners=True)
    real_images = (real_images + 1) / 2  # Unnormalize to [0, 1]
    generated_images = F.interpolate(generated_images, size=(299, 299), mode='bilinear', align_corners=True)
    generated_images = (generated_images + 1) / 2  # Unnormalize to [0, 1]

    # Calculate the activations of real and generated images using Inception-v3
    real_activations = []
    generated_activations = []
    num_images = real_images.shape[0]
    num_batches = (num_images - 1) // batch_size + 1
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            real_batch = real_images[start_idx:end_idx].to(device)
            generated_batch = generated_images[start_idx:end_idx].to(device)
            real_activations.append(inception_model(real_batch).squeeze())
            generated_activations.append(inception_model(generated_batch).squeeze())

    real_activations = torch.cat(real_activations, dim=0)
    generated_activations = torch.cat(generated_activations, dim=0)

    # Calculate the mean and covariance of real and generated activations
    real_mean = real_activations.mean(dim=0)
    generated_mean = generated_activations.mean(dim=0)
    real_cov = torch_covariance(real_activations, rowvar=False)
    generated_cov = torch_covariance(generated_activations, rowvar=False)

    # Calculate the squared Euclidean distance between means
    mean_distance = torch.sum((real_mean - generated_mean) ** 2)

    # Calculate the trace of the product of covariances
    cov_trace = torch.trace(real_cov + generated_cov - 2 * torch.sqrt(real_cov @ generated_cov))

    # Calculate the Frechet Inception Distance (FID)
    fid = mean_distance + cov_trace

    return fid.item()

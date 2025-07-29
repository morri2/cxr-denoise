import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray',
'darkred', 'darkgreen', 'darkblue', 'orange', 'pink', 'purple', 'brown',
'teal', 'lime', 'olive', 'navy', 'maroon', 'aqua', 'fuchsia', 'gold',
'silver', 'indigo', 'violet', 'coral', 'chocolate', 'crimson', 'darkcyan',
'darkorange', 'darkviolet', 'deeppink', 'forestgreen', 'hotpink', 'khaki',
'lavender', 'lightblue', 'lightgreen', 'mediumpurple', 'midnightblue', 'orchid', 'peru', 'plum', 'rosybrown', 'saddlebrown', 'salmon', 'sienna',
'skyblue', 'slateblue', 'slategray', 'springgreen', 'steelblue', 'tan',
'tomato', 'turquoise', 'wheat', 'yellowgreen']

# ------ CXR Image Plotting ------

def plot_cxr_images(images, titles, figsize=(12, 4), vmin=0.0, vmax=1.0):
    """
    Plots CXR images (as PyTorch tensors) in grayscale side by side with titles.
    
    Parameters:
    - images: List of 2D (or 3D) torch.Tensor objects (CXR images).
    - titles: List of strings, one for each image.
    - figsize: Tuple (width, height) for the matplotlib figure.
    """
    if len(images) != len(titles):
        raise ValueError("The number of images must match the number of titles.")
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = images[i]
        
        img = img.detach().cpu().numpy()
        
        ax.imshow(img.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.tight_layout(pad=0.5)
    plt.show()

def plot_cxr_res(clean,noisy,output, residual=None, figsize=(12, 8), vmin=0.0, vmax=1.0):
    titles = ["Clean", "Noisy", "Residual", "Restored"]
    if residual is None:
        residual = output - noisy

    true_residual = clean - noisy

    residual_diff = residual - true_residual
    
    fig, axes = plt.subplots(2,3, figsize=figsize)
    
    # --- imgs ---
    axes[0, 0].imshow(clean.detach().cpu().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Clean", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy.detach().cpu().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Noisy", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(output.detach().cpu().squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Restored", fontsize=10)
    axes[0, 2].axis('off')

    # --- res ---
    rvmax = max(abs(residual.min().item()), abs(residual.max().item()))
    rvmin = -rvmax

    axes[1, 0].imshow(true_residual.detach().cpu().squeeze(), cmap='seismic', vmin=rvmin, vmax=rvmax)
    axes[1, 0].set_title("True Residual", fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(residual.detach().cpu().squeeze(), cmap='seismic', vmin=rvmin, vmax=rvmax)
    axes[1, 1].set_title("Residual", fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(residual_diff.detach().cpu().squeeze(), cmap='seismic', vmin=rvmin, vmax=rvmax)
    axes[1, 2].set_title("Diff", fontsize=10)
    axes[1, 2].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.tight_layout(pad=0.5)
    plt.show()


# ------ ROC Curve Plotting ------

def setup_roc_plot(title="ROC Curve"):
    """Create a new ROC plot with title and axis labels."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.grid(True)
    return fig, ax

def add_roc_curve(ax: Axes, fpr_tpr , label=None, color=None, linestyle="-", lw=2, alpha=0.8, auc_in_label=False):
    """Add a ROC curve to an existing plot. fpr_tpr is a tuple of (fpr, tpr)"""
    fpr, tpr = fpr_tpr
    if auc_in_label:
        label = f"{label} (AUC={auc(fpr, tpr):.2f})"
    ax.plot(fpr, tpr, label=label, color=color, linestyle=linestyle, lw=lw, alpha=alpha)

def finalize_roc_plot(ax, loc="lower right", ncol=1):
    """Finalize and show a ROC plot."""
    ax.legend(loc=loc, ncol=ncol)
    # run plt.show() seperate
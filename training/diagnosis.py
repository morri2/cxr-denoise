import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torchvision.transforms as transforms


def to_range(cxr, from_min, from_max, to_min, to_max):
    """Scales images in [from_min, from_max] to be in the range [to_min, to_max]."""
    cxr = ((cxr - from_min) / (from_max - from_min)) * (to_max - to_min) + to_min
    cxr = cxr.clamp(to_min, to_max)
    return cxr

def norm_to_xrv(cxr: torch.Tensor):
    """Scales images to be [-1024 1024]. the range used in xrv."""
    return to_range(cxr, 0.0, 1.0, -1024, 1024)

class ROCs:
    def __init__(self, preds, labels, targets=None):
        self.preds = preds
        self.labels = labels

        self.targets = targets

    def roc(self, idx):
        return roc_curve(self.labels[:,idx], self.preds[:,idx])
    
    def roc_fpr_tpr(self, idx):
        fpr, tpr, _ = roc_curve(self.labels[:,idx], self.preds[:,idx])
        return (fpr, tpr)
    
    def auc_score(self, idx):
        return roc_auc_score(self.labels[:,idx], self.preds[:,idx])   

    def total_auc_score(self):
        return roc_auc_score(self.labels, self.preds, average='micro')
    
    def avg_auc_score(self):
        return roc_auc_score(self.labels, self.preds, average='macro')

class NihTester:
    """
    512x512 Tester, for nih dataset (only looks at 14 pathologies). 
    downscale: "2x2mean" / "scale_nearest" / "scale_bilinear" / "keep"
    
    """
    def __init__(self, cxr_label_dataset: Dataset, device=torch.device("cuda"), downscale="keep"):
        self.device = device
        self.diagnosis_model = xrv.models.ResNet(weights="resnet50-res512-all").to(self.device)
      
        self.dataloader = DataLoader(cxr_label_dataset, batch_size=16, shuffle=False)

        self.downscale = nn.Identity()

        if downscale == "2x2mean":
            self.downscale = nn.AvgPool2d(2).to(self.device)
        elif downscale == "keep":
            self.downscale = nn.Identity().to(self.device)
        elif downscale == "scale_nearest":
            self.downscale = transforms.Resize(512, transforms.InterpolationMode.NEAREST).to(self.device)
        elif downscale == "scale_bilinear":
            self.downscale = transforms.Resize(512, transforms.InterpolationMode.BILINEAR).to(self.device)
        else:
            print("Invalid downscale settings!")


    def rocs(self, model, preproc: nn.Module = nn.Identity()):
        
        all_outputs = []
        all_labels = []

        try:
            preproc_device = next(preproc.parameters()).device
        except:
            preproc_device = "cpu"

        self.diagnosis_model.eval()
        preproc.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Making ROCs"):

                inputs = preproc(inputs.to(preproc_device))
                
                
                inputs = self.downscale(inputs.to(self.device))

                inputs = norm_to_xrv(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.diagnosis_model(inputs)[:, :14]  # NIH has 14 labels
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_outputs, all_labels = torch.cat(all_outputs, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

        return ROCs(all_outputs, all_labels, targets=self.diagnosis_model.targets)
    

class DiagnosticLoss(nn.Module):
    """
    Diagnostic loss for CXR images.
    This loss is used to train a model to predict the presence of certain diseases in chest X-ray images.
    The loss is based on the binary cross-entropy loss between the predicted and target labels.
    """
    
    def __init__(self, num_classes=14, device="cuda"):
        super(DiagnosticLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device

        self.diagnosis_model = xrv.models.ResNet(weights="resnet50-res512-all").to(self.device)
        self.diagnosis_model.eval()
        for p in self.diagnosis_model.parameters():
            p.requires_grad = False

        self.criterion = nn.BCELoss()


    def forward(self, outputs, targets):
        """
        Compute the diagnostic loss.
        
        Args:
            outputs (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth labels.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Ensure the outputs and targets are on the same device
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)

        if outputs.shape[-1] == 512:
            outputs = F.avg_pool2d(outputs, 2)
        if targets.shape[-1] == 512:
            targets = F.avg_pool2d(targets, 2)

        with torch.no_grad():
            # Rescale [0, 1] -> [-1024, 1024]
            targets = norm_to_xrv(targets)
            targets_diagnosis = self.diagnosis_model(targets)

        outputs = norm_to_xrv(outputs)
        outputs_dignosis = self.diagnosis_model(outputs)
        
        # Compute the binary cross-entropy loss
        diagnosis_loss = self.criterion(outputs_dignosis, targets_diagnosis)
        
        return diagnosis_loss

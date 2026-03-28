import os, io, base64, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import timm, cv2
from torchvision.transforms.functional import to_pil_image

app = Flask(__name__)

# ── Paths modèles ──
BASE = os.path.dirname(os.path.abspath(__file__))
MDL  = os.path.join(BASE, 'Modèles')

PATHOLOGIES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural Thickening','Hernia'
]
N_CLASSES = 14
DEVICE    = torch.device('cpu')

# ── Transforms ──
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ── Définitions architectures ──
class SimpleCNN(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n)
        )
    def forward(self, x): return self.classifier(self.features(x))

class ResNetChest(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048,512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, n)
        )
    def forward(self, x): return self.backbone(x)

class ViTChest(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.vit  = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0, in_chans=1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.vit.embed_dim), nn.Dropout(0.4),
            nn.Linear(self.vit.embed_dim, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, n)
        )
    def forward(self, x):
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        return self.head(self.vit(x))

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(128,64,3,stride=2,padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,128,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,3,stride=2,padding=1,output_padding=1), nn.Tanh()
        )
    def forward(self, x): return self.decoder(self.encoder(x))

class ImageOnlyModel(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,n))
    def forward(self, img, txt=None): return self.fc(self.cnn(img))

VOCAB_NIH = {'<pad>':0,'<unk>':1,'atelectasis':2,'cardiomegaly':3,'consolidation':4,
             'edema':5,'effusion':6,'emphysema':7,'fibrosis':8,'finding':9,'hernia':10,
             'infiltration':11,'mass':12,'no':13,'nodule':14,'pleural':15,
             'pneumonia':16,'pneumothorax':17,'thickening':18}
VOCAB_SIZE = len(VOCAB_NIH)

class TextOnlyModel(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB_SIZE, 64, padding_idx=0)
        self.lstm = nn.LSTM(64,128,batch_first=True,bidirectional=True)
        self.fc   = nn.Sequential(nn.Dropout(0.4), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,n))
    def forward(self, img, txt):
        _, (hn,_) = self.lstm(self.emb(txt))
        return self.fc(torch.cat([hn[0],hn[1]],dim=1))

class MultimodalModel(nn.Module):
    def __init__(self, n=14):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.img_proj = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3))
        self.emb      = nn.Embedding(VOCAB_SIZE, 64, padding_idx=0)
        self.lstm     = nn.LSTM(64,128,batch_first=True,bidirectional=True)
        self.txt_proj = nn.Sequential(nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.3))
        self.fusion   = nn.Sequential(nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,n))
    def forward(self, img, txt):
        img_f     = self.img_proj(self.cnn(img))
        _,(hn,_)  = self.lstm(self.emb(txt))
        txt_f     = self.txt_proj(torch.cat([hn[0],hn[1]],dim=1))
        return self.fusion(torch.cat([img_f,txt_f],dim=1))

# ── Chargement modèles ──
def load_model(cls, path, **kwargs):
    m = cls(**kwargs)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

cnn_model   = load_model(SimpleCNN,      os.path.join(MDL, 'CNN_scratch_best.pth'))
resnet_model= load_model(ResNetChest,    os.path.join(MDL, 'ResNet50_finetune_best.pth'))
vit_model   = load_model(ViTChest,       os.path.join(MDL, 'ViT_small_best.pth'))
ae_model    = load_model(ConvAutoencoder,os.path.join(MDL, 'autoencoder_best.pth'))
img_model   = load_model(ImageOnlyModel, os.path.join(MDL, 'Multimodèle', 'MM_ImageOnly_best.pth'))
txt_model   = load_model(TextOnlyModel,  os.path.join(MDL, 'Multimodèle', 'MM_TextOnly_best.pth'))
mm_model    = load_model(MultimodalModel,os.path.join(MDL, 'Multimodèle', 'MM_Multimodal_best.pth'))
print("✅ Tous les modèles chargés")

# ── Grad-CAM ──
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output): self.activations = output.detach()
    def _save_gradient(self, _, __, grad_output): self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(64,64), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

gradcam = GradCAM(cnn_model, cnn_model.features[-3])

def img_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def tokenize_text(text, max_len=20):
    words  = text.lower().replace('_',' ').replace('|',' ').split()
    tokens = [VOCAB_NIH.get(w, 1) for w in words[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file: return jsonify({'error': 'Aucune image'}), 400

    img_pil = Image.open(file.stream).convert('RGB')
    img_64  = transform(img_pil).unsqueeze(0)

    results = {}
    with torch.no_grad():
        for name, model in [('CNN', cnn_model), ('ResNet50', resnet_model), ('ViT', vit_model)]:
            probs = torch.sigmoid(model(img_64)).squeeze().numpy()
            results[name] = {
                'predictions': [
                    {'label': PATHOLOGIES[i], 'score': float(probs[i])}
                    for i in np.argsort(probs)[::-1]
                ]
            }
        # Score anomalie
        recon     = ae_model(img_64)
        ae_score  = float(F.mse_loss(recon, img_64).item())
        ae_max    = 0.006
        ae_pct    = min(100, (ae_score / ae_max) * 100)
        recon_img = to_pil_image((recon.squeeze() * 0.5 + 0.5).clamp(0,1))

    # Grad-CAM
    img_gc  = img_64.clone().requires_grad_(True)
    top_cls = int(np.argmax([results['CNN']['predictions'][i]['score']
                             for i in range(N_CLASSES)]))
    cam     = gradcam.generate(img_gc, top_cls)
    orig_np = np.array(img_pil.resize((64,64)).convert('L'))
    orig_np = (orig_np / 255.0 * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig_3ch= cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_3ch, 0.6, heatmap, 0.4, 0)
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return jsonify({
        'models'      : results,
        'anomaly'     : {'score': ae_score, 'percent': ae_pct},
        'original_b64': img_to_base64(img_pil.resize((256,256))),
        'recon_b64'   : img_to_base64(recon_img.resize((256,256))),
        'gradcam_b64' : img_to_base64(overlay_pil.resize((256,256))),
    })

@app.route('/multimodal', methods=['POST'])
def multimodal():
    file = request.files.get('image')
    text = request.form.get('text', '')
    if not file: return jsonify({'error': 'Aucune image'}), 400

    img_pil = Image.open(file.stream).convert('RGB')
    img_64  = transform(img_pil).unsqueeze(0)
    tokens  = tokenize_text(text)

    with torch.no_grad():
        p_img = torch.sigmoid(img_model(img_64)).squeeze().numpy()
        p_txt = torch.sigmoid(txt_model(img_64, tokens)).squeeze().numpy()
        p_mm  = torch.sigmoid(mm_model(img_64, tokens)).squeeze().numpy()

    def fmt(probs):
        return [{'label': PATHOLOGIES[i], 'score': float(probs[i])}
                for i in np.argsort(probs)[::-1][:5]]

    return jsonify({
        'image_only' : fmt(p_img),
        'text_only'  : fmt(p_txt),
        'multimodal' : fmt(p_mm),
        'original_b64': img_to_base64(img_pil.resize((256,256))),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

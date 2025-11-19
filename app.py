from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import pickle

app = Flask(__name__)
CORS(app)

# ============ DEFINIÇÃO DO MODELO (copiada do notebook) ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar configurações
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

num_frutas = config['num_frutas']
alpha = config['alpha']

print(f"📦 Carregando modelo...")
print(f"   Frutas: {config['frutas_lista']}")
print(f"   Alpha: {alpha}")

# Arquitetura EXATA do teu modelo
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_feats = backbone.fc.in_features
backbone.fc = nn.Identity()  # Remove FC layer

# Congelar layers (opcional - igual ao notebook)
for name, p in backbone.named_parameters():
    if not name.startswith('layer4'):
        p.requires_grad = False

# Cabeças multi-task
head_fruit = nn.Linear(in_feats, num_frutas)
head_fresh = nn.Linear(in_feats, 2)

class MultiHead(nn.Module):
    def __init__(self, backbone, hf, hs):
        super().__init__()
        self.backbone = backbone
        self.hf = hf
        self.hs = hs
    
    def forward(self, x):
        feats = self.backbone(x)
        return self.hf(feats), self.hs(feats)

model = MultiHead(backbone, head_fruit, head_fresh)
model.load_state_dict(torch.load('modelo_fresh_continente.pth', map_location=device))
model.to(device)
model.eval()

print(f"✓ Modelo carregado com sucesso no {device}")
# =================================================================

# Transformações (mesmas do treino)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber imagem
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        # Preprocessar
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predição
        with torch.no_grad():
            logits_fruit, logits_fresh = model(image_tensor)
            
            # Probabilidades
            prob_fruit = torch.nn.functional.softmax(logits_fruit, dim=1)
            prob_fresh = torch.nn.functional.softmax(logits_fresh, dim=1)
            
            # Predições
            pred_fruit_idx = logits_fruit.argmax(1).item()
            pred_fresh_idx = logits_fresh.argmax(1).item()
            
            confidence_fruit = prob_fruit[0][pred_fruit_idx].item() * 100
            confidence_fresh = prob_fresh[0][pred_fresh_idx].item() * 100
        
        # Nomes
        fruit_name = config['frutas_lista'][pred_fruit_idx]
        freshness = 'Fresh' if pred_fresh_idx == 0 else 'Rotten'
        is_fresh = pred_fresh_idx == 0
        
        return jsonify({
            'fruit': fruit_name.capitalize(),
            'fruit_confidence': round(confidence_fruit, 2),
            'freshness': freshness,
            'freshness_confidence': round(confidence_fresh, 2),
            'is_fresh': is_fresh,
            'recommendation': '✓ OK para venda' if is_fresh else '⚠️ REMOVER DO STOCK',
            'color': 'green' if is_fresh else 'red'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model_loaded': True,
        'frutas_suportadas': config['frutas_lista'],
        'device': str(device)
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de Detecção de Frescura - Continente',
        'endpoints': {
            '/health': 'GET - Status da API',
            '/predict': 'POST - Analisar imagem (form-data: image)'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 API de Detecção de Frescura - Continente")
    print("="*60)
    print(f"📦 Frutas suportadas: {', '.join(config['frutas_lista'])}")
    print(f"🔧 Device: {device}")
    print(f"⚖️ Alpha (peso frutas): {alpha}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)

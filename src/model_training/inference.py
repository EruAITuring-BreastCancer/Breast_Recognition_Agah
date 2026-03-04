import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

from models import get_model
from dataset import prepare_and_split_data

# --- YAPILANDIRMA ---
CONFIG = {
    'model_name': 'resnet',
    'model_size': '50',
    'num_classes': 4,
    'image_size': 224,
    'cc_model_path': '/home/agah/PycharmProjects/Breast_Recognition_Agah/outputs_CC_results/resnet_CC_best_model.pth',
    'mlo_model_path': '/home/agah/PycharmProjects/Breast_Recognition_Agah/outputs_MLO_results/resnet_MLO_best_model.pth',
    'csv_path': '/media/agah/Sata/Breast_veriler/Etiketler/teknofest_final_master.csv',
    'class_names': ['BIRADS-1', 'BIRADS-2', 'BIRADS-4', 'BIRADS-5'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}


def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_trained_model(model_path, device):
    model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        model_size=CONFIG['model_size'],
        pretrained=False
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_case_id(filepath):
    """
    Dosya adından CC veya MLO kısmını atarak vaka ID'sini çıkarır.
    Örnek: 822670189_LCC.png -> 822670189_L
           822670189_LMLO.png -> 822670189_L
    Böylece aynı memenin iki açısı eşleşmiş olur.
    """
    name = os.path.basename(filepath).replace('.png', '').replace('.jpg', '')
    if name.endswith('CC'):
        return name[:-2]  # Sondaki 'CC' yi atar, geriye 'L' veya 'R' kalır
    elif name.endswith('MLO'):
        return name[:-3]  # Sondaki 'MLO' yu atar, geriye 'L' veya 'R' kalır
    return name


def evaluate_test_set():
    print("=" * 70)
    print(" TOPLULUK (ENSEMBLE) MODELİ TEST SETİ DEĞERLENDİRMESİ ")
    print("=" * 70)

    device = CONFIG['device']
    transform = get_transforms(CONFIG['image_size'])

    # 1. Modelleri Yükle
    print("[1/3] Eğitilmiş modeller yükleniyor...")
    cc_model = load_trained_model(CONFIG['cc_model_path'], device)
    mlo_model = load_trained_model(CONFIG['mlo_model_path'], device)

    # 2. Test Verilerini Çek (Eğitimde kullanılan splitin aynısı olmak zorunda)
    print("\n[2/3] Test veri seti hazırlanıyor...")
    _, _, _, _, test_paths_cc, test_labels_cc = prepare_and_split_data(
        csv_path=CONFIG['csv_path'], view_type='CC', random_state=CONFIG['seed'])

    _, _, _, _, test_paths_mlo, test_labels_mlo = prepare_and_split_data(
        csv_path=CONFIG['csv_path'], view_type='MLO', random_state=CONFIG['seed'])

    # Vakaları sözlükte grupla
    cases = {}

    for path, label in zip(test_paths_cc, test_labels_cc):
        case_id = get_case_id(path)
        cases[case_id] = {'CC_path': path, 'label': label}

    for path, label in zip(test_paths_mlo, test_labels_mlo):
        case_id = get_case_id(path)
        if case_id in cases:
            cases[case_id]['MLO_path'] = path
        else:
            # Sadece MLO'su olan (CC'si test setine düşmeyen) vakalar atlanır
            pass

    # Hem CC hem MLO görüntüsü test setine düşen tam vakaları filtrele
    valid_cases = {k: v for k, v in cases.items() if 'CC_path' in v and 'MLO_path' in v}
    print(f"\n[BİLGİ] Test setinde hem CC hem MLO açısı bulunan eşleşmiş tam vaka sayısı: {len(valid_cases)}")

    if len(valid_cases) == 0:
        print(
            "[HATA] Test setinde eşleşen vaka bulunamadı. Veri setinizde her hastanın iki açısı olduğundan emin olun.")
        return

    # 3. Tahminleri Yap
    print("\n[3/3] Ensemble tahminleri yapılıyor...")
    y_true = []
    y_pred_soft = []
    y_pred_worst = []

    for case_id, data in tqdm(valid_cases.items()):
        true_label = data['label']
        y_true.append(true_label)

        # Görüntüleri Yükle
        img_cc = Image.open(data['CC_path']).convert('RGB')
        img_mlo = Image.open(data['MLO_path']).convert('RGB')

        tensor_cc = transform(img_cc).unsqueeze(0).to(device)
        tensor_mlo = transform(img_mlo).unsqueeze(0).to(device)

        with torch.no_grad():
            out_cc = cc_model(tensor_cc)
            out_mlo = mlo_model(tensor_mlo)

            prob_cc = F.softmax(out_cc, dim=1).squeeze(0)
            prob_mlo = F.softmax(out_mlo, dim=1).squeeze(0)

        # Strateji 1: Soft Voting (Olasılık Ortalaması)
        avg_probs = (prob_cc + prob_mlo) / 2
        pred_soft = torch.argmax(avg_probs).item()
        y_pred_soft.append(pred_soft)

        # Strateji 2: Klinik Öncelik (En Kötü Senaryo)
        pred_cc = torch.argmax(prob_cc).item()
        pred_mlo = torch.argmax(prob_mlo).item()
        pred_worst = max(pred_cc, pred_mlo)
        y_pred_worst.append(pred_worst)

    # 4. Sonuçları Yazdır
    print("\n" + "=" * 50)
    print(" SONUÇLAR: STRATEJİ 1 - SOFT VOTING (ORTALAMA)")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred_soft):.4f}")
    print(classification_report(y_true, y_pred_soft, target_names=CONFIG['class_names']))

    print("\n" + "=" * 50)
    print(" SONUÇLAR: STRATEJİ 2 - KLİNİK ÖNCELİK (EN KÖTÜ DURUM)")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred_worst):.4f}")
    print(classification_report(y_true, y_pred_worst, target_names=CONFIG['class_names']))


if __name__ == "__main__":
    evaluate_test_set()
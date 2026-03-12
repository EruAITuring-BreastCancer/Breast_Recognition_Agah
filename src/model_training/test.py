import torch
from models import get_model
from dataset import prepare_and_split_data, CustomImageDataset, get_val_transforms
from train import test_model
from torch.utils.data import DataLoader
from pathlib import Path


def run_test():
    CONFIG = {
        'model_name': 'swin',
        'model_size': 'tiny',
        'num_classes': 3,
        'pretrained': False,  # Ağırlıkları biz yükleyeceğimiz için False
        'image_size': 224,
        'batch_size': 16,
        'num_workers': 4,
        'csv_path': '/media/agah/Sata/Breast_veriler/Etiketler/uclu_siniflandirma_etiketleri.csv',
        'output_dir': 'outputs',
        'seed': 42
    }

    print("=" * 60)
    print("SADECE TEST İŞLEMİ BAŞLATILIYOR")
    print("=" * 60)

    # VRAM doluluğunu önlemek için cihazı seç (Eğitim GPU'yu dolduruyorsa CPU'da test edilebilir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n[1/3] Test verisi hazırlanıyor...")
    _, _, _, _, test_paths, test_labels = prepare_and_split_data(
        csv_path=CONFIG['csv_path'],
        val_size=0.15,
        test_size=0.15,
        random_state=CONFIG['seed']
    )

    test_dataset = CustomImageDataset(
        test_paths,
        test_labels,
        transform=get_val_transforms(CONFIG['image_size'])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    print(f"\n[2/3] Model oluşturuluyor ve ağırlıklar yükleniyor...")
    model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        model_size=CONFIG['model_size'],
        pretrained=CONFIG['pretrained']
    ).to(device)

    best_model_path = Path(CONFIG['output_dir']) / 'best_model.pth'
    if not best_model_path.exists():
        print(f"\n[!] HATA: '{best_model_path}' bulunamadı. Model henüz kaydedilmemiş olabilir.")
        return

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ En iyi ağırlıklar başarıyla yüklendi (Kaydedilen Epoch: {checkpoint.get('epoch', 0) + 1})")

    print(f"\n[3/3] Test başlatılıyor...")
    class_names = ['Sinif 0 (BI-RADS 0)', 'Sinif 1 (BI-RADS 1-2)', 'Sinif 2 (BI-RADS 4-5)']

    test_results = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=CONFIG['output_dir']
    )

    print("\n✓ Test tamamlandı. Sonuçlar 'outputs' klasörüne kaydedildi.")


if __name__ == "__main__":
    run_test()
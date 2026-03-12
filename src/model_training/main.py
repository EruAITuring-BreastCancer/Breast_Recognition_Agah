import torch
from models import get_model
# Yeni fonksiyonumuzu da import ediyoruz
from dataset import create_dataloaders, prepare_and_split_data, CustomImageDataset, get_val_transforms
from train import Trainer, test_model
from torch.utils.data import DataLoader
from pathlib import Path


def main():
    CONFIG = {
        'model_name': 'densenet',
        'model_size': '121',
        'num_classes': 3,
        'pretrained': True,

        'image_size': 224,
        'batch_size': 32,
        'num_workers': 6,
        'use_weighted_sampler': True,

        'num_epochs': 75,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,

        'csv_path': '/media/agah/Sata/Breast_veriler/Etiketler/uclu_siniflandirma_etiketleri.csv',  # CSV dosyasının yolu

        'output_dir': 'outputs',
        'seed': 42
    }

    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    print("=" * 70)
    print("BILGISAYARLI GÖRÜ PROJESİ - EĞİTİM PIPELINE")
    print("=" * 70)

    print("\n[1/4] Veri hazırlanıyor...")

    # dataset.py içindeki yeni fonksiyonumuzu çağırıyoruz
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_and_split_data(
        csv_path=CONFIG['csv_path'],
        val_size=0.15,
        test_size=0.15,
        random_state=CONFIG['seed']
    )

    # Train ve Val DataLoader'larını oluştur
    train_loader, val_loader, data_info = create_dataloaders(
        train_image_paths=train_paths,
        train_labels=train_labels,
        val_image_paths=val_paths,
        val_labels=val_labels,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['image_size'],
        use_weighted_sampler=CONFIG['use_weighted_sampler']
    )

    # Test DataLoader'ını oluştur
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

    print(f"\n[2/4] Model oluşturuluyor...")

    model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        model_size=CONFIG['model_size'],
        pretrained=CONFIG['pretrained']
    )

    print(f"✓ Model: {CONFIG['model_name'].upper()} ({CONFIG['model_size']})")
    print(f"✓ Pretrained: {CONFIG['pretrained']}")
    print(f"✓ Sınıf sayısı: {CONFIG['num_classes']}")

    print(f"\n[3/4] Eğitim başlatılıyor...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=data_info['num_classes'],
        class_weights=data_info['class_weights'],
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        output_dir=CONFIG['output_dir']
    )

    history = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_best=True
    )

    print(f"\n[4/4] Model test ediliyor...")

    best_model_path = Path(CONFIG['output_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ En iyi model yüklendi (Epoch {checkpoint['epoch'] + 1})")

    # Modeli test et
    class_names = ['Sinif 0 (BI-RADS 0)', 'Sinif 1 (BI-RADS 1-2)', 'Sinif 2 (BI-RADS 4-5)']
    test_results = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=CONFIG['output_dir']
    )

    print("\n" + "=" * 70)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"\nEn iyi validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Test accuracy: {test_results['accuracy']:.2f}%")
    print(f"f1_macro: {test_results['f1_macro']:.2f}%")
    print(f"f1_weighted: {test_results['f1_weighted']:.2f}%")

    print("\nSınıf Bazlı F1 Skorları:")
    for class_name, f1_val in test_results['f1_per_class'].items():
        print(f"  {class_name}: {f1_val:.2f}%")

    print(f"\nSonuçlar '{CONFIG['output_dir']}' klasörüne kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
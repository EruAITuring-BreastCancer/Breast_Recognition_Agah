import torch
from models import get_model
from dataset import create_dataloaders, prepare_and_split_data, CustomImageDataset, get_val_transforms
from train import Trainer, test_model
from torch.utils.data import DataLoader
from pathlib import Path


def main():
    CONFIG = {
        'model_name': 'mobilenet',
        'model_size': 'small',
        'num_classes': 4,  # 4 Sınıflı orijinal yapı (1, 2, 4, 5)
        'pretrained': True,

        'image_size': 224,
        'batch_size': 32,
        'num_workers': 8,
        'use_weighted_sampler': False,

        'num_epochs': 30,
        'learning_rate': 1e-3,  # SGD için ayarlanmış yüksek LR
        'weight_decay': 1e-4,

        'csv_path': '/media/agah/Sata/Breast_veriler/Etiketler/teknofest_final_master.csv',
        'seed': 42
    }

    # Sabitlemeler
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    print("=" * 70)
    print("BİLGİSAYARLI GÖRÜ PROJESİ - AÇI TABANLI (VIEW-SPECIFIC) EĞİTİM")
    print("=" * 70)

    # Eğitilecek Açıların Listesi
    views_to_train = ['CC', 'MLO']

    for view in views_to_train:
        print(f"\n\n{'*' * 50}")
        print(f" {view} GÖRÜNTÜLERİ İÇİN EĞİTİM BAŞLIYOR ")
        print(f"{'*' * 50}")

        # Her açı için farklı bir çıktı klasörü (Örn: outputs_CC_results)
        current_output_dir = f"outputs_{view}_results"
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        print("\n[1/4] Veri hazırlanıyor...")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_and_split_data(
            csv_path=CONFIG['csv_path'],
            view_type=view,  # Filtreleme burada devreye girer
            val_size=0.15,
            test_size=0.15,
            random_state=CONFIG['seed']
        )

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

        test_dataset = CustomImageDataset(
            test_paths, test_labels, transform=get_val_transforms(CONFIG['image_size'])
        )
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers']
        )

        print(f"\n[2/4] Model oluşturuluyor...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Her döngüde modeli sıfırdan oluşturuyoruz ki önceki açının ağırlıklarını hatırlamasın
        model = get_model(
            model_name=CONFIG['model_name'],
            num_classes=CONFIG['num_classes'],
            model_size=CONFIG['model_size'],
            pretrained=CONFIG['pretrained']
        )

        print(f"\n[3/4] {view} için Eğitim başlatılıyor...")

        # Dosya isimlerine açıyı ekliyoruz (Örn: resnet_CC_best_model.pth)
        dynamic_model_name = f"{CONFIG['model_name']}_{view}"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=data_info['num_classes'],
            model_name=dynamic_model_name,
            class_weights=None,
            device=device,
            learning_rate=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
            output_dir=current_output_dir,
            num_epochs=CONFIG['num_epochs']
        )

        history = trainer.train(num_epochs=CONFIG['num_epochs'], save_best=True)

        print(f"\n[4/4] {view} modeli test ediliyor...")

        best_model_filename = f"{dynamic_model_name}_best_model.pth"
        best_model_path = Path(current_output_dir) / best_model_filename

        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ {view} için en iyi model yüklendi (Epoch {checkpoint['epoch'] + 1})")
        else:
            print(f"✗ Uyarı: {best_model_filename} bulunamadı!")

        class_names = ['BIRADS-1', 'BIRADS-2', 'BIRADS-4', 'BIRADS-5']
        test_results = test_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            output_dir=current_output_dir
        )

        # --- EKSİK OLAN ÖZET YAZDIRMA KISMI EKLENDİ ---
        print(f"\n[{view} AÇISI İÇİN ÖZET SONUÇLAR]")
        print(f"En iyi Validation Accuracy: {trainer.best_val_acc:.2f}%")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
        print(f"F1-Macro: {test_results['f1_macro']:.2f}%")
        print(f"F1-Weighted: {test_results['f1_weighted']:.2f}%")

        print("Sınıf Bazlı F1 Skorları:")
        for class_name, f1_val in test_results['f1_per_class'].items():
            print(f"  {class_name}: {f1_val:.2f}%")

        print(f"\n✓ {view} açısı eğitimi tamamlandı. Sonuçlar '{current_output_dir}' klasöründe.")

        # Döngü bitişi
    print("\n" + "=" * 70)
    print("TÜM AÇILARIN EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
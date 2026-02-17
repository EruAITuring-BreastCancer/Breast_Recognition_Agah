import torch
from models import get_model
from dataset import create_dataloaders
from train import Trainer, test_model


def main():

    CONFIG = {
        'model_name': 'convnext',
        'model_size': 'tiny',
        'num_classes': 4,
        'pretrained': True,

        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        'use_weighted_sampler': True,

        'num_epochs': 30,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,

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


    # Gerçek yolları ekleyeceğim
    train_image_paths = []
    train_labels = []
    val_image_paths = []
    val_labels = []
    test_image_paths = []
    test_labels = []

    from pathlib import Path
    data_dir = Path('your_dataset')

    for class_idx, class_name in enumerate(sorted(data_dir.iterdir())):
        if class_name.is_dir():
            images = list(class_name.glob('*.jpg'))

            train_split = int(len(images) * 0.7)
            val_split = int(len(images) * 0.85)

            train_imgs = images[:train_split]
            val_imgs = images[train_split:val_split]
            test_imgs = images[val_split:]

            train_image_paths.extend([str(img) for img in train_imgs])
            train_labels.extend([class_idx] * len(train_imgs))

            val_image_paths.extend([str(img) for img in val_imgs])
            val_labels.extend([class_idx] * len(val_imgs))

            test_image_paths.extend([str(img) for img in test_imgs])
            test_labels.extend([class_idx] * len(test_imgs))



    train_loader, val_loader, data_info = create_dataloaders(
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        val_image_paths=val_image_paths,
        val_labels=val_labels,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        image_size=CONFIG['image_size'],
        use_weighted_sampler=CONFIG['use_weighted_sampler']
    )

    from dataset import CustomImageDataset, get_val_transforms
    from torch.utils.data import DataLoader

    test_dataset = CustomImageDataset(
        test_image_paths,
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


    class_names = [f'Class_{i}' for i in range(CONFIG['num_classes'])]
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
    print(f"f1_per_class: {test_results['f1_per_class:']:.2f}%")
    print(f"classification_report: {test_results['classification_report']:.2f}%")
    print(f"confusion_matrix: {test_results['confusion_matrix']:.2f}%")
    print(f"\nSonuçlar '{CONFIG['output_dir']}' klasörüne kaydedildi:")
    print(f"  - best_model.pth (en iyi model)")
    print(f"  - last_model.pth (son model)")
    print(f"  - training_history.png (eğitim grafikleri)")
    print(f"  - confusion_matrix.png (karmaşıklık matrisi)")
    print(f"  - test_results.json (detaylı test sonuçları)")
    print("=" * 70)


if __name__ == "__main__":
    main()
import torch
from models import get_model
from dataset import create_dataloaders, prepare_and_split_data, CustomImageDataset, get_val_transforms
from train import Trainer, test_model
from torch.utils.data import DataLoader
from pathlib import Path


def main():
    CONFIG = {
        'model_name': 'paper_hybrid',
        'model_size': '50',  # Bu otomatik olarak Hybrid modele yönlendirecek
        'num_classes': 4,
        'pretrained': True,
        'image_size': 224,
        'batch_size': 8,

        'num_workers': 4,

        'use_weighted_sampler': False,
        'num_epochs': 75,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'csv_path': '/media/agah/Sata/Breast_veriler/Etiketler/inbreast_final_master.csv',
        'seed': 42
    }

    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    print("=" * 70)
    print("BİLGİSAYARLI GÖRÜ PROJESİ - BİRLEŞTİRİLMİŞ (UNIFIED) EĞİTİM V2")
    print("=" * 70)

    current_output_dir = "outputs_UNIFIED_results"
    Path(current_output_dir).mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Veri hazırlanıyor (Tüm açılar bir arada)...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_and_split_data(
        csv_path=CONFIG['csv_path'],
        view_type='ALL',  # <-- Tümü aynı anda eğitilecek
        val_size=0.15,
        test_size=0.15,
        random_state=CONFIG['seed']
    )

    train_loader, val_loader, data_info = create_dataloaders(
        train_image_paths=train_paths, train_labels=train_labels,
        val_image_paths=val_paths, val_labels=val_labels,
        batch_size=CONFIG['batch_size'], num_workers=CONFIG['num_workers'],
        image_size=CONFIG['image_size'], use_weighted_sampler=CONFIG['use_weighted_sampler']
    )

    test_dataset = CustomImageDataset(test_paths, test_labels, transform=get_val_transforms(CONFIG['image_size']))
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'])

    print(f"\n[2/4] Hibrit Model oluşturuluyor...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        model_size=CONFIG['model_size'],
        pretrained=CONFIG['pretrained']
    )"""

    model = get_model('paper_hybrid', num_classes=data_info['num_classes']).to(device)

    print(f"\n[3/4] Birleşik (Unified) Eğitim başlatılıyor...")
    dynamic_model_name = f"{CONFIG['model_name']}_UNIFIED"

    # ==========================================
    # AŞAMA 1: Feature Adaptation Phase
    # ==========================================
    print("\n--- STAGE 1: Feature Adaptation (Backbone Frozen) ---")
    for param in model.backbone.parameters(): param.requires_grad = False

    for param in model.projection.parameters(): param.requires_grad = True
    model.pos_embedding.requires_grad = True
    for param in model.transformer.parameters(): param.requires_grad = True
    for param in model.classifier.parameters(): param.requires_grad = True

    # GÜNCELLENDİ: Makaleye göre ilk aşama sadece 4 epoch
    stage1_epochs = 4
    trainer_stage1 = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        num_classes=data_info['num_classes'], model_name="HybridResNet_Stage1",
        device=device, learning_rate=1e-4,
        weight_decay=CONFIG.get('weight_decay', 1e-4),
        output_dir=current_output_dir, num_epochs=stage1_epochs
    )

    trainer_stage1.train(num_epochs=stage1_epochs, save_best=True)

    # ==========================================
    # AŞAMA 2: Fine-Tuning Phase
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning (Unfreezing Layer 3 & 4) ---")
    for param in model.backbone[6].parameters(): param.requires_grad = True
    for param in model.backbone[7].parameters(): param.requires_grad = True

    # GÜNCELLENDİ: Makaleye göre ince ayar 60 epoch ve 1e-5 learning rate
    stage2_epochs = 60
    stage2_lr = 1e-5

    # İkinci aşama için optimizer'ı yeni LR ile kur
    trainer_stage1.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage2_lr, weight_decay=CONFIG.get('weight_decay', 1e-4)
    )

    # GÜNCELLENDİ: İkinci aşama için Scheduler'ı da baştan kur (ReduceLROnPlateau)
    trainer_stage1.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer_stage1.optimizer, mode='max', factor=0.5, patience=2
    )

    # GÜNCELLENDİ: Early stopping sayaçlarını yeni aşama için sıfırla
    trainer_stage1.early_stop_triggered = False
    trainer_stage1.epochs_no_improve = 0

    trainer_stage1.model_name = "HybridResNet_Stage2"

    # Eğitime Stage 2 ayarlarla devam et
    trainer_stage1.train(num_epochs=stage2_epochs, save_best=True)


    print(f"\n[4/4] Unified model test ediliyor...")
    best_model_path = Path(current_output_dir) / f"{dynamic_model_name}_best_model.pth"

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ En iyi model yüklendi (Epoch {checkpoint['epoch'] + 1})")

    test_results = test_model(
        model=model, test_loader=test_loader, device=device,
        class_names=['BIRADS-1', 'BIRADS-2', 'BIRADS-4', 'BIRADS-5'],
        output_dir=current_output_dir
    )

    print(f"\n✓ Unified eğitimi tamamlandı. Sonuçlar '{current_output_dir}' klasöründe.")


if __name__ == "__main__":
    main()
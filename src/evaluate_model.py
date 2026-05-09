import matplotlib
matplotlib.use('Agg')  # Terminalde çalışması için GUI olmadan kaydet

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def cross_validate_models(X, y, save_path=None):
    """5-Fold Stratified Cross-Validation ile overfitting kontrolü.
    Her katlamada sınıf oranları korunur (Stratified)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000),
    }

    print("\n" + "=" * 60)
    print("  5-FOLD STRATIFIED CROSS-VALIDATION (Overfitting Kontrolü)")
    print("=" * 60)

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        cv_results[name] = scores
        print(f"\n  {name}:")
        print(f"    Fold Skorları : {[f'{s:.4f}' for s in scores]}")
        print(f"    Ortalama      : {scores.mean():.4f}")
        print(f"    Std Sapma     : {scores.std():.4f}")
        print(f"    Min/Max       : {scores.min():.4f} / {scores.max():.4f}")

    # Cross-Validation sonuçlarını görselleştir
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels = []
    colors = ['#2196F3', '#FF5722']
    for i, (name, scores) in enumerate(cv_results.items()):
        bp = ax.boxplot(scores, positions=[i], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=colors[i], alpha=0.7),
                        medianprops=dict(color='black', linewidth=2))
        # Her fold skoru da nokta olarak göster
        ax.scatter([i] * len(scores), scores, color='black', zorder=5, s=40, alpha=0.6)
        positions.append(i)
        labels.append(name)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Stratified Cross-Validation Sonuçları')
    ax.set_ylim(0.98, 1.002)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  → CV grafiği kaydedildi: {save_path}")
    plt.close()

    # Sonuçları dosyaya kaydet
    if save_path:
        txt_path = save_path.replace('.png', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== 5-FOLD STRATIFIED CROSS-VALIDATION SONUÇLARI ===\n\n")
            for name, scores in cv_results.items():
                f.write(f"{name}:\n")
                f.write(f"  Fold Skorları : {[f'{s:.4f}' for s in scores]}\n")
                f.write(f"  Ortalama      : {scores.mean():.4f}\n")
                f.write(f"  Std Sapma     : {scores.std():.4f}\n")
                f.write(f"  Min/Max       : {scores.min():.4f} / {scores.max():.4f}\n\n")
            f.write("YORUM: Eğer standart sapma düşükse (< 0.005) ve tüm fold skorları\n")
            f.write("birbirine yakınsa, model overfitting YAPMIYOR demektir.\n")
        print(f"  → CV raporu kaydedildi: {txt_path}")

    return cv_results


def evaluate(model, X_test, y_test, model_name="Model"):
    """Modeli değerlendir: classification report ve confusion matrix."""
    y_pred = model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"{model_name} - Classification Report:")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return y_pred, accuracy_score(y_test, y_pred)


def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """Confusion matrix görselleştir."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Grafik kaydedildi: {save_path}")
    plt.close()


def compare_models(results, save_path=None):
    """İki modelin accuracy'sini karşılaştır."""
    names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=['#2196F3', '#FF5722'])
    plt.title('Model Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1.0)

    # Bar üstüne değer yaz
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Grafik kaydedildi: {save_path}")
    plt.close()

def plot_feature_importance(model, vectorizer, top_n=20, model_name="Model", save_path=None):
    """Lineer modeller için en önemli kelimeleri (feature importance) görselleştir."""
    if not hasattr(model, 'coef_'):
        print(f"{model_name} modeli feature importance desteklemiyor.")
        return
        
    coefs = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    
    # AI (Sınıf 1) için en etkili kelimeler
    top_ai_idx = np.argsort(coefs)[-top_n:]
    top_ai_features = [feature_names[i] for i in top_ai_idx]
    top_ai_coefs = coefs[top_ai_idx]
    
    # İnsan (Sınıf 0) için en etkili kelimeler
    top_human_idx = np.argsort(coefs)[:top_n]
    top_human_features = [feature_names[i] for i in top_human_idx]
    top_human_coefs = abs(coefs[top_human_idx]) # Mutlak değer olarak göster
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.barh(top_ai_features, top_ai_coefs, color='#FF5722')
    plt.title(f'AI Sınıfı İçin En Önemli Kelimeler ({model_name})')
    plt.xlabel('Coefficient Değeri')
    
    plt.subplot(1, 2, 2)
    plt.barh(top_human_features, top_human_coefs, color='#2196F3')
    plt.title(f'İnsan Sınıfı İçin En Önemli Kelimeler ({model_name})')
    plt.xlabel('Mutlak Coefficient Değeri')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Feature importance grafiği kaydedildi: {save_path}")
    plt.close()

def perform_error_analysis(y_test, y_pred, texts_test, model_name="Model", save_path=None):
    """Yanlış tahmin edilen metinleri analiz et ve raporla."""
    import pandas as pd
    
    df_errors = pd.DataFrame({
        'Gerçek_Sınıf': y_test,
        'Tahmin_Edilen': y_pred,
        'Metin': texts_test
    })
    
    # Gerçekle tahminin uyuşmadığı satırlar
    errors = df_errors[df_errors['Gerçek_Sınıf'] != df_errors['Tahmin_Edilen']]
    
    # 0 = Human, 1 = AI
    # False Positives: Gerçek Human (0), Tahmin AI (1)
    fps = errors[(errors['Gerçek_Sınıf'] == 0.0) & (errors['Tahmin_Edilen'] == 1.0)]
    
    # False Negatives: Gerçek AI (1), Tahmin Human (0)
    fns = errors[(errors['Gerçek_Sınıf'] == 1.0) & (errors['Tahmin_Edilen'] == 0.0)]
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"=== ERROR ANALYSIS: {model_name} ===\n")
            f.write(f"Toplam Hata: {len(errors)}\n")
            f.write(f"False Positives (İnsan yazmış ama model AI sanmış): {len(fps)}\n")
            f.write(f"False Negatives (AI yazmış ama model İnsan sanmış): {len(fns)}\n\n")
            
            f.write("--- ÖRNEK 5 FALSE POSITIVE ---\n")
            for idx, row in fps.head(5).iterrows():
                f.write(f"- {row['Metin'][:500]}...\n\n")
                
            f.write("--- ÖRNEK 5 FALSE NEGATIVE ---\n")
            for idx, row in fns.head(5).iterrows():
                f.write(f"- {row['Metin'][:500]}...\n\n")
                
        print(f"  → Error analysis raporu kaydedildi: {save_path}")
    else:
        print(f"  -> Error analysis: {len(errors)} hata tespit edildi ({len(fps)} FP, {len(fns)} FN)")

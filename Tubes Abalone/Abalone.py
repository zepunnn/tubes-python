import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

abalone_data = pd.read_csv('abalone.csv')

def main_menu():
    print("\n=== Abalone Dataset Manager ===")
    print("1. Tampilkan data sebagai tabel")
    print("2. Buat grafik atau diagram")
    print("3. Tambahkan data baru")
    print("4. Hapus data")
    print("5. Edit data")
    print("6. Analisis Deskriptif")
    print("7. Korelasi dan Heatmap")
    print("8. Decision Tree")
    print("9. Analisis Kolom 'Rings'")
    print("10. Proses Data dengan Minimal 20 Cincin")
    print("11. Keluar")
    
    choice = input("Pilih opsi (1-11): ")
    return choice

def display_table(data):
    print("\n=== Data Abalone ===")
    total_rows = len(data)
    rows_per_page = 100 
    start = 0

    while start < total_rows:
        end = min(start + rows_per_page, total_rows)
        print(data.iloc[start:end].to_string(index=True))
        start += rows_per_page
        
        if start < total_rows:
            more = input("\nTampilkan lebih banyak? (y/n): ").strip().lower()
            if more != 'y':
                break

def create_plot(data):
    print("\n=== Pilih Grafik atau Diagram ===")
    print("1. Histogram distribusi cincin (umur)")
    print("2. Scatter plot: Panjang vs Berat Keseluruhan")
    print("3. Diagram Boxplot untuk Berat Cangkang")
    print("4. Pairplot untuk semua fitur")
    print("5. Histogram untuk fitur numerik")
    
    choice = input("Pilih opsi (1-5): ")
    if choice == '1':
        plt.hist(data['rings'], bins=15, color='skyblue', edgecolor='black')
        plt.title("Distribusi Jumlah Cincin")
        plt.xlabel("Jumlah Cincin")
        plt.ylabel("Frekuensi")
        plt.show()
    elif choice == '2':
        plt.scatter(data['length'], data['whole-weight'], alpha=0.6, c='orange')
        plt.title("Panjang vs Berat Keseluruhan")
        plt.xlabel("Panjang (mm)")
        plt.ylabel("Berat Keseluruhan (g)")
        plt.show()
    elif choice == '3':
        plt.boxplot(data['shell-weight'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        plt.title("Boxplot Berat Cangkang")
        plt.ylabel("Berat Cangkang (g)")
        plt.show()
    elif choice == '4':
        sns.pairplot(data, hue='sex')
        plt.title("Pairplot untuk Semua Fitur")
        plt.show()
    elif choice == '5':
        data.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
        plt.suptitle("Histogram untuk Fitur Numerik")
        plt.show()
    else:
        print("Pilihan tidak valid!")

def add_data(data):
    print("\n=== Tambahkan Data Baru ===")
    try:
        new_row = {
            'sex': input("Jenis kelamin (M/F/I): "),
            'length': float(input("Panjang (mm): ")),
            'diameter': float(input("Diameter (mm): ")),
            'height': float(input("Tinggi (mm): ")),
            'whole-weight': float(input("Berat keseluruhan (g): ")),
            'shucked-weight': float(input("Berat daging (g): ")),
            'viscera-weight': float(input("Berat isi perut (g): ")),
            'shell-weight': float(input("Berat cangkang (g): ")),
            'rings': int(input("Jumlah cincin: "))
        }
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        print("Data berhasil ditambahkan!")
    except ValueError:
        print("Input tidak valid. Data gagal ditambahkan.")
    return data

def delete_data(data):
    print("\n=== Hapus Data ===")
    try:
        index = int(input("Masukkan indeks baris yang ingin dihapus: "))
        if index in data.index:
            data = data.drop(index)
            print("Data berhasil dihapus!")
        else:
            print("Indeks tidak ditemukan!")
    except ValueError:
        print("Input tidak valid. Data gagal dihapus.")
    return data

def edit_data(data):
    print("\n=== Edit Data ===")
    try:
        index = int(input("Masukkan indeks baris yang ingin diedit: "))
        if index in data.index:
            column = input("Masukkan nama kolom yang ingin diedit: ")
            if column in data.columns:
                new_value = input("Masukkan nilai baru: ")
                data.at[index, column] = float(new_value) if column not in ['sex'] else new_value
                print("Data berhasil diperbarui!")
            else:
                print("Kolom tidak ditemukan!")
        else:
            print("Indeks tidak ditemukan!")
    except ValueError:
        print("Input tidak valid. Data gagal diperbarui.")
    return data

def descriptive_analysis(data):
    print("\n=== Analisis Deskriptif ===")
    print(data.describe(include='all'))
    print("\nJumlah nilai unik per kolom:")
    print(data.nunique())

def correlation_heatmap(data):
    print("\n=== Korelasi dan Heatmap ===")
    plt.figure(figsize=(10, 8))
    
    # Pilih hanya kolom numerik untuk menghitung korelasi
    numeric_data = data.select_dtypes(include=[np.number])
    
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title("Matriks Korelasi")
    plt.show()

def analyze_rings(data):
    print("\n=== Analisis Kolom 'Rings' ===")
    print("Statistik Deskriptif untuk Jumlah Cincin:")
    print(data['rings'].describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data['rings'], bins=20, kde=True, color='skyblue')
    plt.title("Distribusi Jumlah Cincin")
    plt.xlabel("Jumlah Cincin")
    plt.ylabel("Frekuensi")
    plt.show()

def process_min_20_rings(data):
    print("\n=== Data dengan Jumlah Cincin Minimal 20 ===")
    filtered_data = data[data['rings'] >= 20]
    if filtered_data.empty:
        print("Tidak ada data dengan jumlah cincin minimal 20.")
    else:
        display_table(filtered_data)
        # Tambahkan analisis tambahan jika diperlukan

def train_decision_tree(data):
    print("\n=== Decision Tree Regression ===")
    
    # Encode kolom 'sex' menggunakan One-Hot Encoding
    data_encoded = pd.get_dummies(data, columns=['sex'])
    
    # Memisahkan fitur dan target
    X = data_encoded.drop('rings', axis=1)
    y = data_encoded['rings']
    
    # Mengisi nilai kosong jika ada
    X = X.fillna(0)
    
    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat dan melatih model Decision Tree Regressor dengan parameter yang dituning
    model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_leaf=4)
    model.fit(X_train, y_train)
    
    # Memprediksi set pengujian
    y_pred = model.predict(X_test)
    
    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error pada set pengujian: {mse:.2f}")
    print(f"R^2 Score pada set pengujian: {r2:.2f}")
    
    # Plot Actual vs Predicted Rings
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='red')
    plt.title("Actual vs Predicted Rings")
    plt.xlabel("Actual Rings")
    plt.ylabel("Predicted Rings")
    plt.show()

# Main program loop
current_data = abalone_data.copy()
while True:
    user_choice = main_menu()
    if user_choice == '1':
        display_table(current_data)
    elif user_choice == '2':
        create_plot(current_data)
    elif user_choice == '3':
        current_data = add_data(current_data)
    elif user_choice == '4':
        current_data = delete_data(current_data)
    elif user_choice == '5':
        current_data = edit_data(current_data)
    elif user_choice == '6':
        descriptive_analysis(current_data)
    elif user_choice == '7':
        correlation_heatmap(current_data)
    elif user_choice == '8':
        train_decision_tree(current_data)
    elif user_choice == '9':
        analyze_rings(current_data)
    elif user_choice == '10':
        process_min_20_rings(current_data)
    elif user_choice == '11':
        print("Keluar dari program. Sampai jumpa!")
        break
    else:
        print("Pilihan tidak valid, coba lagi.")
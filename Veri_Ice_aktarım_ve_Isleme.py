import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Dosya yolunu belirtin
folder_path = 'D:\\EEG\\PythonApplication1\\dataset'

# EEG kanallari
columns = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']

def load_csv_files(folder_path):
    # Tum CSV dosyalarini listele
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Tum CSV dosyalarini birlestirmek icin bos bir DataFrame olusturun
    combined_data = pd.DataFrame()
    
    # Her bir CSV dosyasini oku ve birlestir
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path, header=None, names=columns)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    return combined_data

def preprocess_and_reduce_data_tsne(data):
    # Eksik verileri doldurma
    data.ffill(inplace=True)
    
    # Veriyi normalize etme
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Normalizasyon sonrasi veriyi DataFrame'e donusturme
    normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
    
    # t-SNE ile boyut indirgeme
    tsne = TSNE(n_components=3, random_state=42)
    reduced_data = tsne.fit_transform(normalized_data)
    
    # Boyut indirgenmiþ veriyi DataFrame'e dönüþtürme
    reduced_data_df = pd.DataFrame(reduced_data, columns=['Dim1', 'Dim2', 'Dim3'])
    
    # Boyut indirgenmiþ veriyi .txt dosyasina yazdirma
    reduced_data_df.to_csv('reduced_data_tsne.txt', sep='\t', index=False)
    
    return reduced_data_df

def main():
    # Verileri yukleme
    combined_data = load_csv_files(folder_path)
    print("Veri yuklendi ve birlestirildi.")
    
    # Verileri on isleme ve boyut indirgeme (t-SNE)
    reduced_data = preprocess_and_reduce_data_tsne(combined_data)
    print("Veri on islendi ve normalize edildi. Boyut indirgenmis veri 'reduced_data_tsne.txt' dosyasina yazildi.")

if __name__ == "__main__":
    main()

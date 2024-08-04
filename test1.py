import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

# Dalga üretme fonksiyonu
def generate_wave(freq, amp, duration, sample_rate=100):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

# Dalga görüntüsü oluşturma fonksiyonu
def create_wave_image(wave):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(wave)
    ax.axis('off')
    plt.tight_layout()
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    img = np.mean(img, axis=2).astype(np.float32) / 255.0
    return img

# Veri seti oluşturma fonksiyonu
def create_dataset(n_samples=1000):
    X, y = [], []
    wave_types = ['Beta', 'Alpha', 'Theta', 'Delta']
    for _ in range(n_samples):
        wave_type = np.random.choice(wave_types)
        if wave_type == 'Beta':
            freq = np.random.uniform(12, 30)
            amp = np.random.uniform(0.1, 0.5)
        elif wave_type == 'Alpha':
            freq = np.random.uniform(8, 12)
            amp = np.random.uniform(0.5, 1)
        elif wave_type == 'Theta':
            freq = np.random.uniform(4, 8)
            amp = np.random.uniform(1, 1.5)
        else:  # Delta
            freq = np.random.uniform(1, 4)
            amp = np.random.uniform(1.5, 2)
        
        wave = generate_wave(freq, amp, duration=1)
        img = create_wave_image(wave)
        X.append(img)
        y.append(wave_types.index(wave_type))
    
    return np.array(X), np.array(y)

# Veri seti oluştur ve modeli eğit
X, y = create_dataset(2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1), 
          y_train, epochs=3, batch_size=32, 
          validation_split=0.2, verbose=1)

# Gerçek zamanlı analiz için gerekli değişkenler
wave_types = ['Beta', 'Alpha', 'Theta', 'Delta']
current_wave = generate_wave(3, 0.8, duration=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 8))
line, = ax1.plot(current_wave)
ax1.set_ylim(-2, 2)
ax1.set_title('Gerçek Zamanlı Beyin Dalgası')
bar_container = ax2.bar(wave_types, [0, 0, 0, 0])
ax2.set_ylim(0, 1)
ax2.set_title('Dalga Tipi Olasılıkları')

# Animasyon güncelleme fonksiyonu
def update(frame):
    global current_wave
    
    # Yeni dalga üret
    wave_type = np.random.choice(wave_types)
    if wave_type == 'Beta':
        freq, amp = np.random.uniform(12, 30), np.random.uniform(0.1, 0.5)
    elif wave_type == 'Alpha':
        freq, amp = np.random.uniform(8, 12), np.random.uniform(0.5, 1)
    elif wave_type == 'Theta':
        freq, amp = np.random.uniform(4, 8), np.random.uniform(1, 1.5)
    else:  # Delta
        freq, amp = np.random.uniform(1, 4), np.random.uniform(1.5, 2)
    
    new_wave = generate_wave(freq, amp, duration=0.1)
    current_wave = np.concatenate((current_wave[len(new_wave):], new_wave))
    
    # Grafiği güncelle
    line.set_ydata(current_wave)
    
    # Tahmin yap
    img = create_wave_image(current_wave)
    prediction = model.predict(img.reshape(1, img.shape[0], img.shape[1], 1))[0]
    
    # Tahmin çubuklarını güncelle
    for rect, h in zip(bar_container, prediction):
        rect.set_height(h)
    
    return line, bar_container

np.save('EEG.npy')

# Animasyonu başlat
ani = FuncAnimation(fig, update, frames=200, interval=100, blit=False)
plt.tight_layout()
plt.show()
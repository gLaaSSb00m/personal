import numpy as np
import librosa
import os
from scipy.linalg import norm
import soundfile as sf
import shutil

# Configuration
voice_dir = 'Dataset & code/speech'
noise_dir = 'Dataset & code/noise'
sample_rate = 8000
frame_length = 8064
hop_length_frame = 8064
n_fft = 255
hop_length = 63
SNR = 0

# Get file lists
list_noise_files = os.listdir(noise_dir)
list_voice_files = os.listdir(voice_dir)

import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss/1024/1024:.2f} MB")


def audio_to_audio_frame_stack(audio, frame_length, hop_length):
    total_samples = len(audio)
    frames = []
    for start in range(0, total_samples - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)
    return np.vstack(frames)

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame):
    list_sound_array = []
    for file in list_audio_files:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
    return np.vstack(list_sound_array)

def mixed_voice_with_noise(voice, noise, nb_samples, frame_length, SNR):
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        prod_voice[i, :] = voice[i, :]
        prod_noise[i, :] = noise[i, :]/norm(noise[i, :])*10**(-SNR/20)*norm(voice[i, :])
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]
    return prod_voice, prod_noise, prod_noisy_voice

def extract_stft_features_to_disk(numpy_audio, dim_square_spec, n_fft, hop_length_fft, output_dir, batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    
    nb_audio = numpy_audio.shape[0]
    
    for i in range(0, nb_audio, batch_size):
        batch = numpy_audio[i:i + batch_size]
        mag_batch = []
        phase_real_batch = []
        phase_imag_batch = []
        
        for audio in batch:
            try:
                stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
                mag_audio, phase_audio = librosa.magphase(stftaudio)
                
                # Extract magnitude and split phase into components
                mag_batch.append(mag_audio[:dim_square_spec, :dim_square_spec].astype(np.float32))
                phase_real_batch.append(phase_audio[:dim_square_spec, :dim_square_spec].real.astype(np.float32))
                phase_imag_batch.append(phase_audio[:dim_square_spec, :dim_square_spec].imag.astype(np.float32))
            except Exception as e:
                print(f"Error processing audio batch {i}: {str(e)}")
                continue
        
        # Save each component separately
        try:
            np.save(os.path.join(output_dir, f'mag_batch_{i//batch_size}.npy'), np.array(mag_batch))
            np.save(os.path.join(output_dir, f'phase_real_batch_{i//batch_size}.npy'), np.array(phase_real_batch))
            np.save(os.path.join(output_dir, f'phase_imag_batch_{i//batch_size}.npy'), np.array(phase_imag_batch))
        except OSError as e:
            print(f"Error saving batch {i//batch_size}: {str(e)}")
            # Try with smaller batch size if error persists
            if batch_size > 10:
                print(f"Retrying with smaller batch size {batch_size//2}")
                return extract_stft_features_to_disk(numpy_audio, dim_square_spec, n_fft, hop_length_fft, output_dir, batch_size//2)
            else:
                raise
def main():
    # Clean up previous runs
    if os.path.exists('sound'):
        shutil.rmtree('sound')
    os.makedirs('sound')
    
    if os.path.exists('spectrogram_data'):
        shutil.rmtree('spectrogram_data')
    os.makedirs('spectrogram_data')

    try:
        # Load and process voice and noise
        print("Loading voice files...")
        voice = audio_files_to_numpy(voice_dir, list_voice_files, sample_rate, frame_length, hop_length_frame)
        print_memory_usage()
        
        print("Loading noise files...")
        noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate, frame_length, hop_length_frame)
        print_memory_usage()

        # Ensure equal lengths
        l = np.min([voice.shape[0], noise.shape[0]])
        voice = voice[:l]
        noise = noise[:l]

        # Calculate square spectrogram dimensions
        dim_square_spec = int(n_fft / 2) + 1
        print(f"Spectrogram dimensions: {dim_square_spec}x{dim_square_spec}")

        # Create mixed signals
        print("Creating mixed signals...")
        prod_voice, prod_noise, prod_noisy = mixed_voice_with_noise(voice, noise, l, frame_length, SNR)
        del voice, noise  # Free memory
        print_memory_usage()

        # Save raw waveforms
        print("Saving raw waveforms...")
        voice_flat = prod_voice.reshape(1, l * frame_length)
        noise_flat = prod_noise.reshape(1, l * frame_length)
        noisy_flat = prod_noisy.reshape(1, l * frame_length)
        
        sf.write('sound/voice.wav', voice_flat[0, :], samplerate=sample_rate)
        sf.write('sound/noise.wav', noise_flat[0, :], samplerate=sample_rate)
        sf.write('sound/noisy.wav', noisy_flat[0, :], samplerate=sample_rate)

        # Process and save spectrograms in batches
        print("Processing voice spectrograms...")
        extract_stft_features_to_disk(prod_voice, dim_square_spec, n_fft, hop_length, 'spectrogram_data/voice', batch_size=50)
        del prod_voice
        print_memory_usage()
        
        print("Processing noise spectrograms...")
        extract_stft_features_to_disk(prod_noise, dim_square_spec, n_fft, hop_length, 'spectrogram_data/noise', batch_size=50)
        del prod_noise
        print_memory_usage()
        
        print("Processing noisy spectrograms...")
        extract_stft_features_to_disk(prod_noisy, dim_square_spec, n_fft, hop_length, 'spectrogram_data/noisy', batch_size=50)
        del prod_noisy
        print_memory_usage()

        print("Processing complete!")
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        raise
if __name__ == "__main__":
    main()
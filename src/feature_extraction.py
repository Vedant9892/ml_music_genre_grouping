import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class AudioFeatureExtractor:
    """Extract 5 key features from audio files for clustering."""
    
    def __init__(self, audio_dir):
        self.audio_dir = Path(audio_dir)
        
    def extract_features(self, file_path):
        """Extract tempo, energy, loudness, valence, and danceability."""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, duration=30)
            
            # 1. Tempo (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # 2. Energy (RMS mean)
            rms = librosa.feature.rms(y=y)
            energy = np.mean(rms)
            
            # 3. Loudness (RMS mean - same as energy for GTZAN)
            loudness = np.mean(rms)
            
            # 4. Valence (approximated: harmony + spectral centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            harmonic, _ = librosa.effects.hpss(y)
            harmonic_mean = np.mean(np.abs(harmonic))
            valence = (np.mean(spectral_centroid) / 4000 + harmonic_mean) / 2
            
            # 5. Danceability (approximated: tempo + zero crossing rate)
            zcr = librosa.feature.zero_crossing_rate(y)
            danceability = (tempo / 200 + np.mean(zcr)) / 2
            
            return {
                'filename': file_path.name,
                'genre': file_path.parent.name,
                'tempo': float(tempo),
                'energy': float(energy),
                'loudness': float(loudness),
                'valence': float(valence),
                'danceability': float(danceability)
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def extract_all_features(self):
        """Extract features from all audio files in dataset."""
        features_list = []
        
        # Get all audio files
        audio_files = list(self.audio_dir.glob('**/*.wav'))
        
        print(f"Extracting features from {len(audio_files)} audio files...")
        
        for audio_file in tqdm(audio_files):
            features = self.extract_features(audio_file)
            if features:
                features_list.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle missing values
        df.fillna(0, inplace=True)
        
        return df
    
    def save_features(self, output_path):
        """Extract and save features to CSV."""
        df = self.extract_all_features()
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        return df


# Usage example
if __name__ == "__main__":
    extractor = AudioFeatureExtractor('data/gtzan/genres_original')
    features_df = extractor.save_features('data/processed/features_selected.csv')
    print(f"Extracted {len(features_df)} songs with 5 features each")

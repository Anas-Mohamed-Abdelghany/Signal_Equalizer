import numpy as np
from utils.logger import get_logger
from ai.demucs_wrapper import spectral_separate
from ai.ai_config import load_mode_bands

logger = get_logger(__name__)

try:
    import librosa
    from sklearn.decomposition import FastICA
    _ICA_AVAILABLE = True
except ImportError:
    _ICA_AVAILABLE = False
    logger.warning("librosa or sklearn not installed — falling back to spectral masking for ECG.")

def ecg_ica_separate(signal: np.ndarray, sr: int, source_bands: list) -> list[dict]:
    """
    Separates ECG arrhythmias using Independent Component Analysis (ICA),
    the textbook Machine Learning algorithm for biomedical signal separation.
    """
    if not _ICA_AVAILABLE:
         return spectral_separate(signal, sr, source_bands)
         
    logger.info("Running FastICA Machine Learning separation for ECG mode")
    try:
        n_components = len(source_bands)
        
        # Use STFT magnitude as multi-dimensional observations for ICA
        S = librosa.stft(signal, n_fft=1024, hop_length=256)
        X, phase = librosa.magphase(S)
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=200, tol=0.01)
        # Transpose X to shape (n_samples, n_features) for processing by ICA
        S_ = ica.fit_transform(X.T)  
        A_ = ica.mixing_  

        results = []
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        
        assigned_masks = []
        for i in range(n_components):
            # Reconstruct the component magnitude
            component_mag = np.outer(S_[:, i], A_[:, i]).T
            component_mag = np.abs(component_mag)
            
            energies = []
            for band in source_bands:
                energy = 0
                for (low, high) in band.get("ranges", []):
                    idx = (freqs >= low) & (freqs <= high)
                    energy += np.sum(component_mag[idx, :])
                energies.append(energy)
            assigned_masks.append((i, energies))
            
        used_components = set()
        for b_idx, band in enumerate(source_bands):
             best_c = -1
             best_energy = -1
             for c_idx, energies in assigned_masks:
                  if c_idx not in used_components and energies[b_idx] > best_energy:
                       best_energy = energies[b_idx]
                       best_c = c_idx
             
             if best_c != -1:
                 used_components.add(best_c)
                 component_mag = np.outer(S_[:, best_c], A_[:, best_c]).T
                 component_mag = np.abs(component_mag)
                 Y = component_mag * phase
                 y_reconstructed = librosa.istft(Y, hop_length=256, length=len(signal))
                 results.append({"label": band["label"], "signal": y_reconstructed})
             else:
                 results.append({"label": band["label"], "signal": np.zeros(len(signal))})
                 
        return results
        
    except Exception as e:
        logger.error(f"ICA separation failed: {e}")
        try:
            source_bands = load_mode_bands("ecg")
        except Exception:
            pass
        return spectral_separate(signal, sr, source_bands)
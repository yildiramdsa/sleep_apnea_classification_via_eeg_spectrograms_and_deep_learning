# Sleep Apnea Classification via EEG Spectrograms and Deep Learning

**Tech Stack:** ![ResNet50](https://img.shields.io/badge/ResNet50-EE4C2C?logo=pytorch&logoColor=white) ![YOLOv5](https://img.shields.io/badge/YOLOv5-FF9900?logo=pytorch&logoColor=white) ![YOLOv8](https://img.shields.io/badge/YOLOv8-0072C6?logo=pytorch&logoColor=white)

## Methodology

1. **Data Acquisition & Preprocessing**  
   - EEG signals were extracted from polysomnography (PSG) recordings.  
   - A 30-second sliding window was applied to segment the signals by apnea–hypopnea index (AHI) values.  
   - Short-time Fourier transform (STFT) generated spectrograms for each segment, which serve as model inputs.

2. **Model Architectures**  
   - **YOLOv8**: Chosen for its balance of accuracy and speed, with reduced parameter count and fewer layers.  
   - **ResNet64**: A standard convolutional baseline known for strong EEG-based classification performance.  
   - **YOLOv5**: Included for comparison, demonstrating effective event detection but with higher computational cost.

3. **Training & Evaluation**  
   - Four-class classification labels: Healthy, Mild, Moderate, Severe apnea.  
   - Balanced-class reference level set at 25% per class, ensuring equal sample representation.  
   - Metrics: Total correct classification (TCC) ratio across classes.

## Experiments & Results

| Model     | TCC (%) | Notes                                                     |
|-----------|---------|-----------------------------------------------------------|
| YOLOv8    | 93.7    | Highest accuracy; fastest inference; fewest parameters    |
| ResNet64  | 93.0    | Comparable accuracy; higher parameter count               |
| YOLOv5    | 88.2    | Good detection but slower and more computationally heavy  |

- **Key Finding:** YOLOv8 matches or exceeds ResNet64’s performance while reducing model size and inference time.  
- **Novelty:** Parameter-reduction strategies in four-class EEG classification have been underexplored; this study demonstrates their viability.

## Conclusion

This work introduces a lightweight, high-accuracy pipeline for multi-severity sleep apnea classification using EEG spectrograms, and positions YOLOv8 as a competitive new tool in this domain.

## References

Tanci, K., & Hekim, M. (2025). *Classification of sleep apnea syndrome using the spectrograms of EEG signals and YOLOv8 deep learning model*. *PeerJ Computer Science, 11*, e2718. https://doi.org/10.7717/peerj-cs.2718

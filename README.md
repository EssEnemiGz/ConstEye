# Constellation Eye (ConstEye)
## Brief Introduction to the Project
I present Constellation Eye (ConstEye), the interactive platform for exoplanet discovery using NASA resources. The project combines classic machine learning techniques (Random Forest and Convolutional Neural Networks) with astrophysics techniques like the transit method to classify changes in light curves and predict potential exoplanets.

## ConstEye's Motivation
My purpose with this project is to implement modern AI techniques combined with the large volumes of public information shared by NASA, to automate the process of classifying exoplanets into three main types (non-exoplanet, exoplanet, candidate). This will significantly speed up discovery and motivate other young people around the world to delve into data analysis and astrophysics. This entire project is free and Open Source, and it's designed to be accessible in terms of both hardware and internet connection.

# 1. Introduction to Exoplanets
### What is the motivation for searching for exoplanets?
Exoplanets are any planet outside our solar system, usually orbiting their own stars. The distance between the closest exoplanets and Earth leads one to ask, "Why do we search for exoplanets? If we can't even inhabit them." Exoplanet research has many motivations, including that by discovering new exoplanets, we understand more about the formation of our own solar system, the composition of other planets, similarities to Earth, and of course, investigating if there are any planets that favor the proliferation of life.

### Importance of automation with AI
The amount of data collected by modern telescopes reaches terabytes daily—a massive volume of data. The only viable way to analyze such an amount of data is with Deep Learning techniques, due to how fast and efficient AI is for these tasks. It reduces false positives, and AI is also more sensitive to subtle patterns like faint dips in light. All of this motivates space agencies to automate the exoplanet detection and classification process.

# 2. History
## 2.1 Discovery of Exoplanets
### Transit Method
The Transit Method is the primary technique for discovering exoplanets, and it consists of **detecting the periodic and subtle decrease in a star's brightness** when a planet crosses in front of it from our perspective. This slight dimming of light, called a **transit**, allows astronomers to measure the **planet's size** and its **orbital period**. The pattern must repeat regularly to confirm that the object is a planet orbiting the star, and its success has been key to modern astrophysics.

### Missions
The Transit Method has been implemented by crucial space missions that require high precision and continuous observation. The [**Kepler Mission**](https://science.nasa.gov/mission/kepler/) was a pioneer, observing over 150,000 stars in a small region of the sky and confirming thousands of exoplanets, demonstrating that planets are common. Its extension, **K2**, continued the work by observing different fields. The current mission, [**TESS**](https://science.nasa.gov/mission/tess/) (Transiting Exoplanet Survey Satellite), is the successor and focuses on scanning most of the sky for planets orbiting bright, nearby stars, providing a wealth of candidates for follow-up studies with more powerful telescopes, such as the [James Webb](https://science.nasa.gov/mission/webb/).

## 2.2 Machine Learning in Astronomy
### AI Models Used in Astronomy
Modern astronomy, with its flood of data from telescopes, has become a data-driven science, and AI is the key tool. Machine Learning models are used to classify galaxies, predict supernovae, and, as you've been doing, **detect exoplanets**.

| Model Type             | Examples                                                                     | Advantages                                                                                                                                                                        | Challenges                                                                                         |
| :--------------------- | :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **Classic ML**         | **Random Forests**, Support Vector Machines (SVM)                            | **Accessibility:** They are fast to train and do not require powerful hardware (you can use your own laptop!). Ideal for **tabular data** (extracted numerical features).      | Require a **human expert** to extract relevant features *before* training.                       |
| **Deep Learning (DL)** | **Convolutional Neural Networks (CNNs)**, Recurrent Neural Networks (RNNs)   | **Power:** They learn to extract features directly from **raw data** (images, time series). Achieve higher accuracy in complex tasks.                                           | Require **large datasets** and **specialized hardware** (GPUs) for training.                     |

If you're looking to get into this field, **Classic ML, like Random Forests, is the perfect starting point.**

Models like Random Forest work incredibly well with pre-processed **tabular data** from the [Kepler Dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative). In this project (the first version), it has been shown that you can achieve a high level of accuracy (F1 of up to 0.78) without relying on expensive equipment (the first model was trained and tested on a MacBook Air M2 with 8GB of RAM; any equivalent is feasible). The field of exoplanet detection has many already extracted and clean features (such as orbital period, transit duration, or signal-to-noise ratio), making it the ideal scenario for a simple scikit-learn model (like the one shown in the project's early commits) to **demonstrate its value without the need for a supercomputer**. Take advantage of the accessibility of these techniques to make your first discoveries!

# 3. Data
Rigorous analysis in exoplanet detection is based on a precise understanding of the structure of datasets generated by transit missions and the application of standard preprocessing methodologies.

## 3.1 Kepler Dataset

The [Kepler Dataset](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) represents the tabular database resulting from the initial processing of light curves captured by the Kepler Mission. This format facilitates the application of classic Machine Learning algorithms due to its structured nature.

- **Structure:** The dataset is fundamentally **tabular**, where each entry (row) corresponds to an identified stellar transit event, labeled as a Kepler Object of Interest (KOI).
- **Features:** Columns represent previously extracted **astrophysical and signal parameters** from light curve analysis. Crucial features for discrimination include **Orbital Period** (P), **Transit Depth** (δ), **Signal-to-Noise Ratio** (SNR), and various diagnostic metrics designed to identify false positive phenomena (e.g., eclipsing binary stars).
- **Confirmed Exoplanets:** The historical validity of the dataset lies in its inclusion of thousands of validated candidates, providing the gold standard labels for training and evaluating classification models.

### 3.2 Light Curve Datasets

**Light Curve Datasets** (generated by missions such as Kepler, K2, and TESS) constitute the time series of stellar brightness, being the primary input for Deep Learning (DL) models, such as Convolutional Neural Networks (CNNs).

- **Preprocessing Steps:** To isolate the transit signal, essential preprocessing steps are required:
    1. **Trend Correction:** Removing intrinsic stellar variability (e.g., starspot activity) to flatten the light curve baseline. This is commonly done by Spline fitting or low-frequency component removal.
    2. **Normalization:** Scaling the light curve so that the brightness flux unaffected by the transit is set to a base unit (e.g., 1.0), standardizing the data for model input.
    3. **Handling Missing Data (Gaps):** Interruptions in observation (caused by recalibrations or instrumental errors) lead to gaps in the data. Management includes **interpolating** lost points (imputation) or **segmenting** the light curve into contiguous blocks to avoid artifacts introduced by the gap.

### 3.3 Data Labeling
The quality of labels is critical for training AI models. The detection process is typically modeled as a multi-class classification problem:

- **Label 1: Confirmed:** Assigned to events that have been fully validated by multiple methods or follow-up analysis, representing **true positives**.
- **Label 2: Candidate:** Assigned to events that show a high probability of being a planetary transit but **have not yet been fully verified**. These are priority targets for follow-up validation.
- **Label 0: Non-Exoplanet / False Positive:** Assigned to transit events attributed to non-planetary astrophysical phenomena (e.g., eclipsing binary stars, stellar blends) or instrumental artifacts. Robust labeling in this class is fundamental to **mitigate the false positive rate** of the AI model.

### 3.4 Preprocessing
#### Feature Extraction for Random Forests
Exoplanet detection often relies on the **transit method**, where a star's light dims slightly when a planet passes in front of it. A **Random Forest** is like a team of many "judges" (decision trees) who vote on whether a signal is an exoplanet or a "false positive" (like two eclipsing stars). For these judges to decide, we don't give them all the raw light curve data (which is thousands of points), but rather key **features** extracted, such as the **orbital period** (how often the light dip repeats), the **transit depth** (how much the star dims), and the **transit duration**. By converting the complex light curve into a few numerical and descriptive features, the Random Forest model can very efficiently learn to identify the patterns that truly indicate the presence of an exoplanet with high accuracy.

#### Light Curve Normalization for Convolutional Neural Networks (CNNs)
**Convolutional Neural Networks (CNNs)** are a type of machine learning algorithm excellent at finding patterns in structured data, such as images or, in this case, the "image" of the **light curve** (the graph of a star's brightness over time). However, each star is different: some are intrinsically brighter or noisier than others. **Normalization** is a crucial preprocessing step that standardizes all light curves. Imagine it's like adjusting the brightness and contrast of all photos to the same level before showing them to the CNN. This ensures that the neural network focuses on the *shape* of the dip (the exoplanet transit signature) and is not distracted by the star's overall brightness or telescope noise. Typically, this involves setting the average brightness to zero and the variation to one.

#### Handling Imbalanced Datasets (SMOTE, Augmentation)
In the search for exoplanets, the problem of **class imbalance** is enormous: we have millions of stars that **do not have a detectable exoplanet** (the majority class) for every few that **do** have a confirmed exoplanet (the minority class). If we train a machine learning model with this data as is, it might simply learn to always predict "no exoplanet" and still achieve high accuracy, but it would be useless for discovery! Techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** and **data augmentation** address this. [SMOTE](https://medium.com/@thecontentfarmblog/smote-a-powerful-technique-for-handling-imbalanced-data-2375ad46103c) generates new synthetic samples of the minority class (exoplanets) by interpolating between existing ones, which helps balance the dataset. Data augmentation, on the other hand, could create more examples of exoplanet light curves by applying small transformations (like slight noise changes), giving the model enough "material" to learn to recognize the rare but important pattern of planetary transit.

# 4. Methodology Used
## 4.1 Machine Learning Models

### Random Forest

**Features Used:** The Random Forest model uses 11 astrophysical features previously extracted from the Kepler dataset, including orbital period (`koi_period`), transit duration (`koi_duration`), transit depth (`koi_depth`), planetary radius (`koi_prad`), impact parameter (`koi_impact`), stellar radius (`koi_srad`), stellar effective temperature (`koi_steff`), equilibrium temperature (`koi_teq`), stellar mass (`koi_smass`), insolation (`koi_insol`), and model signal-to-noise ratio (`koi_model_snr`). These features represent fundamental physical parameters describing both the candidate planet's properties and those of its host star.

**Training and Validation:** The model is trained using a Random Forest with 100 decision trees and a fixed random seed to ensure reproducibility. The dataset is split into 80% for training and 20% for testing, and the SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the classes and avoid bias towards majority classes. The model is saved in pickle format for later use in predictions.

**Metrics (Precision, Recall, F1-score):** The model generates a complete classification report that includes precision, recall, and F1-score for each of the three classes: "Non-Exoplanet," "Confirmed Exoplanet," and "Candidate." According to the project documentation, the model achieves an F1-score of up to 0.78, demonstrating solid performance in classifying exoplanets using only pre-processed tabular features.

### CNN

**Input Representation (Flow/Time Arrays, 1D Convolution):** The CNN processes light curves as one-dimensional time sequences, where each point represents the stellar flux at a specific time. Curves are normalized by z-score (zero mean, one standard deviation) and padded or truncated to a fixed length of 2000 points to maintain consistency in input size. The model uses 1D convolution to capture temporal patterns in the light curves, allowing it to automatically identify distinctive features of planetary transits.

**Architecture (Layers, Activation Functions):** The ExoCNN architecture consists of three sequential convolutional blocks, each with a 1D convolutional layer followed by ReLU and MaxPooling. The first block uses 8 filters with a kernel size of 9, the second 16 filters with a kernel size of 5, and the third 32 filters with a kernel size of 5. After the convolutional layers, global adaptive pooling and a fully connected classifier with two linear layers (32→64→3) interspersed with ReLU and 50% dropout for regularization are applied.

**Training Procedure:** The model is trained for a maximum of 50 epochs using the Adam optimizer with a learning rate of 0.001 and weight decay of 1e-5. Early stopping with a patience of 5 epochs is implemented to prevent overfitting, and a scheduler that reduces the learning rate when the loss plateaus is used. Training includes data augmentation with temporal shifts, Gaussian noise, and synthesis of additional transits to improve the model's generalization.

**Evaluation Metrics:** The model is evaluated using multiple comprehensive metrics including a classification report with precision, recall, and F1-score, a confusion matrix, ROC and Precision-Recall curves with areas under the curve (AUC), and calibration curves to assess the reliability of predicted probabilities. Additionally, visualizations of misclassified cases are generated for detailed analysis of the model's performance on different types of light curves.

### Get Motivated to Use It and Hunt for Exoplanets
These AI models do not require supercomputers or expensive specialized hardware. Both the Random Forest and the CNN have been designed and optimized to run perfectly on accessible hardware like a MacBook Air M2 with only 8GB of RAM (or any equivalent). This means you can train, run, and experiment with exoplanet detection using the same computer you use for daily work. The democratization of space science is here: you no longer need access to supercomputing centers to make significant astronomical discoveries. Your laptop can be your personal exoplanet lab.

## 4.2 Interactive Platform

The ConstEye project offers you an intuitive and exciting web experience where you can become a planet hunter from the comfort of your computer. The platform is designed to be as easy to use as uploading a photo to social media, but with the power of advanced artificial intelligence. Simply drag and drop your astronomical data file, and watch the magic of science unfold before your eyes with clear visual indicators and a modern interface that makes space exploration feel like a game.

Users can upload new light curves, run predictions, and visualize transits: Do you have data from space telescopes? It's your time to shine! Upload NPZ or CSV files with light curves and let our AI do the heavy lifting. In seconds, you'll get an accurate prediction of whether you've found a real exoplanet, a promising candidate, or just a false positive. The platform shows you not only the final answer but also how confident the AI is in its prediction, giving you the confidence to know if you've made a genuine discovery or need more observations.

Transit Visualization: See the magic in action! Our visualization tool allows you to see exactly what the AI is analyzing: interactive and beautiful graphs of light curves that reveal the subtle "blinks" of stars as a planet passes in front. You can zoom, explore different sections of the data, and see in real-time how transit patterns become evident. It's like having a virtual telescope that allows you to see the universe in a completely new way, where every dip in stellar brightness could be the signature of a distant world waiting to be discovered.

# 5. Model Performance and Capabilities (CNN)
## 5.1 Performance Metrics
The CNN model achieves an average accuracy of **78-82%** in classifying three classes (Non-Exoplanet, Confirmed Exoplanet, Candidate). This is a solid metric considering the complexity of the exoplanet detection task.

- **Non-Exoplanet (Class 0):** Accuracy of **85-90%** - The model is excellent at identifying false positives.
- **Confirmed Exoplanet (Class 1):** Accuracy of **75-80%** - Good performance in detecting real exoplanets.
- **Candidate (Class 2):** Accuracy of **65-70%** - Area for improvement, confuses some candidates with other classes.

## 5.2 Model Strengths

**Excellent at False Positives:** The model has an outstanding ability to correctly identify cases that are NOT exoplanets, which is crucial to avoid false alarms in astronomical research. This is especially valuable because it significantly reduces follow-up time on expensive telescopes.

**Robustness on Accessible Hardware:** The model maintains its performance even on modest hardware like a MacBook Air M2, democratizing exoplanet research without the need for supercomputers.

**Clear Visualization:** ROC curves show areas under the curve (AUC) greater than 0.85 for the main classes, indicating good separability between classes.

## 5.3 Weaknesses and Limitations

**Confusion in Candidates:** The model has consistent difficulty distinguishing between candidates and confirmed exoplanets. This is understandable since the difference between these classes often requires follow-up observations that are not available in the initial light curves.

**Preprocessing Dependence:** The model is sensitive to the normalization and padding of light curves. Changes in preprocessing can significantly affect performance, limiting its robustness under different data conditions.

**Dataset Limitations:** Performance is limited by the quality and representativeness of the training dataset. Light curves of highly variable stars or with atypical patterns may be misclassified.

## 5.4 Training Method Used
The model uses an innovative strategy that combines **6,000 synthetic light curves** (2,000 per class) generated algorithmically with a limited number of real samples from the Kepler telescope. This hybrid approach allows for training robust models even in environments with limited internet connectivity (as I was in such an environment during this hackathon), democratizing access to exoplanet research.

The system generates synthetic light curves that faithfully mimic real planetary transit patterns. For confirmed exoplanets (class 1), transits with depths between 0.001-0.01 and widths of 5-50 points are created, while candidates (class 2) exhibit more subtle transits (0.0005-0.005) with additional noise to simulate observational uncertainty. "Non-Exoplanet" curves (class 0) maintain constant flux with minimal stochastic variations.

The dataset includes carefully selected real samples of known objects such as Kepler-9, Kepler-7, Kepler-11, and other confirmed exoplanets, along with KOI (Kepler Objects of Interest) candidates and known variable stars such as KIC 8462852 (the famous "Tabby's Star"). These real samples provide the "ground truth" that anchors the model to physical reality.

### Advantages of the Hybrid Approach

This method allows researchers anywhere in the world to train exoplanet detection models without relying on massive data downloads. Once the initial real samples (approximately 100-200 curves) are downloaded, the rest of the training is based on locally generated synthetic data.

The combination of synthetic and real data creates a model that generalizes better to new data. Synthetic data provides controlled variability and consistent patterns, while real samples ensure the model captures the complexities and artifacts of the real world.

The system can easily generate more synthetic data as needed, allowing experiments with different dataset sizes without bandwidth limitations. This is especially valuable for researchers in institutions with limited internet resources.

By generating synthetic data, full control over the quality and characteristics of the light curves is achieved, allowing the model to be trained on specific scenarios or extreme cases that might be rare in real data.

This innovative strategy makes exoplanet research truly accessible, allowing anyone with a laptop to contribute to the discovery of new worlds, regardless of their geographical location or connectivity resources.

# 6. Model Limitations and Areas for Improvement

## 6.1 Identified Limitations

**Data Limitations:** The current model relies primarily on synthetic data (6,000 curves) with only ~100 real samples, which can create a bias towards idealized patterns. Synthetic curves, while useful, do not fully capture the complexity of instrumental noise, intrinsic stellar variability, and processing artifacts that characterize real space telescope data.

**Simplified Architecture:** The current CNN uses only 3 convolutional layers with a relatively simple architecture (8→16→32 filters). This structure, while efficient, may be insufficient to capture complex and subtle patterns in light curves with multiple overlapping transits or complex planetary systems.

**Candidate Classification:** The model consistently struggles to distinguish between candidates (class 2) and confirmed exoplanets (class 1), with an accuracy of 65-70% on candidates. This limitation is critical as candidates represent the most important area for the discovery of new exoplanets.

**Preprocessing Dependence:** The model is sensitive to the normalization and padding of light curves. Changes in preprocessing can significantly affect performance, limiting its robustness under different data conditions.

## 6.2 Improvement Techniques Awaiting Implementation

**Ensemble Learning:** Implement a voting system that combines the Random Forest (excellent for tabular features) with the CNN (superior for temporal patterns). This combination can leverage the strengths of both approaches, improving overall accuracy and reducing variance in predictions.

**Advanced Data Augmentation:** Expand data augmentation techniques to include more realistic variations such as specific Kepler/K2 instrumental noise, stellar variability of different spectral types, and effects of multiple transiting planets. This would help close the gap between synthetic and real data.

**Deeper Architectures:** Experiment with more sophisticated architectures such as ResNet-1D, Transformer for time series, or attention networks that can capture long-range dependencies in light curves. These architectures could significantly improve the detection of complex patterns.

**Transfer Learning:** Utilize pre-trained models on larger astronomical datasets or adapt successful architectures from other time series domains. This could accelerate training and improve performance with less data.

**Incorporating Metadata:** Integrate contextual information such as stellar spectral type, apparent magnitude, and estimated orbital parameters directly into the model. This additional information could help resolve classification ambiguities.

**Advanced Regularization Techniques:** Implement adaptive dropout, batch normalization, and specific regularization techniques for time series that reduce overfitting and improve generalization.

**Temporal Cross-Validation:** Use cross-validation that respects the temporal nature of the data, avoiding data leakage that can occur with simple random splits.

**Specialized Evaluation Metrics:** Develop specific metrics for exoplanet detection that consider the cost of false positives (wasted telescope time) versus false negatives (missed exoplanets).

# 7. Conclusion
## Summary of Achievements

**Democratization of Space Science**: ConstEye has made exoplanet detection accessible to anyone with a laptop, removing the barrier of access to expensive supercomputers. The project demonstrates that it is possible to train sophisticated AI models on modest hardware like a MacBook Air M2, achieving accuracies of 78-82% in exoplanet classification.

**Innovation in Data Methodology**: A unique hybrid strategy was developed that combines 6,000 synthetic light curves with carefully selected real samples, allowing for robust training even in low-connectivity environments. This approach solves the problem of access to large astronomical datasets and democratizes research.

**Complete Interactive Platform**: A modern and intuitive web interface was created that allows non-expert users to upload astronomical data, run real-time predictions, and visualize results in a comprehensible way. The platform includes interactive visualizations, confidence indicators, and robust error handling.

**Optimized Dual Models**: Both Random Forest (F1-score 0.78) and CNN (accuracy 78-82%) were successfully implemented, optimized for different types of analysis, with Random Forest excellent for quick screening and CNN superior for detailed analysis of temporal patterns.

## Future Work

**Real-time Data Integration**: The next evolutionary step includes direct integration with APIs from space telescopes like TESS, Kepler, and future missions like PLATO. This will allow automatic analysis of new data as soon as it becomes available, transforming ConstEye into a real-time discovery tool.

**Expansion to Additional Missions**: Plans include incorporating data from multiple space missions including K2, TESS, and ground-based data from surveys like NGTS and WASP. This diversification will improve model robustness and allow for the detection of exoplanets around different types of stars and orbital configurations.

**More Robust Models**: The roadmap includes implementing advanced architectures such as Transformers for time series, more sophisticated ensemble systems, and transfer learning techniques. The goal is to achieve 90%+ accuracies and significantly reduce false positives, making the system competitive with traditional detection methods.

**Advanced Features**: Capabilities for multi-planetary system detection, automatic orbital parameter estimation, and exoplanet type classification will be developed. Astrophysical metadata will also be integrated to improve accuracy and provide additional scientific context.

# 8. References
### Datasets and Space Missions
- Kepler Dataset: NASA Exoplanet Archive - Main database of Kepler Objects of Interest (KOI) and confirmed exoplanets
- LightKurve Library: Python tool for downloading and processing light curves from Kepler and K2 missions
- Kepler Mission: NASA Kepler mission data for exoplanet detection by transit method
### Specific Astronomical Objects
- Confirmed Exoplanets: Kepler-9, Kepler-7, Kepler-5, Kepler-11, Kepler-12, Kepler-17, Kepler-20, Kepler-22, Kepler-37, Kepler-62, Kepler-69, Kepler-78, Kepler-90, Kepler-186, Kepler-452
- KOI Candidates: KOI-102, KOI-87, KOI-94, KOI-1428, KOI-314, KOI-7016, KOI-2124, KOI-268, KOI-292, KOI-3158, KOI-217, KOI-2418, KOI-3512
- Variable Stars/False Positives: KIC 6278683, KIC 9832227, KIC 1026957, KIC 12557548, KIC 8462852 (Tabby's Star), KIC 6933899, KIC 3241344, KIC 3427720, KIC 7671081, KIC 12356914, KIC 9705459, KIC 5113061, KIC 10118816, KIC 2997455
### Technologies and Libraries
- PyTorch: Deep learning framework for CNN implementation
- scikit-learn: Machine learning library for Random Forest and evaluation metrics
- FastAPI: Web framework for the API backend
- React + TypeScript: Frontend technologies for the web interface
- Recharts: Visualization library for interactive charts
- SMOTE (Synthetic Minority Oversampling Technique): Class balancing technique from the imbalanced-learn package
### Metrics and Evaluation
- Multi-class Classification: Precision, Recall, F1-score for three classes (Non-Exoplanet, Confirmed Exoplanet, Candidate)
- ROC and Precision-Recall Curves: Performance analysis with areas under the curve (AUC)
- Confusion Matrix: Detailed analysis of classification errors
- Calibration Curves: Evaluation of the reliability of predicted probabilities
### Hardware and Accessibility
- MacBook Air M2 (8GB RAM): Reference platform to demonstrate training accessibility.
- Low Connectivity Environments: Synthetic data strategy to democratize access to research. The connectivity where the project was developed was 1.03 Mbit/s (download) and 1.82 Mbit/s (upload) according to Ookla's SpeedTest, with the nearest provider at 3.79 km and a ping of 139.317ms.

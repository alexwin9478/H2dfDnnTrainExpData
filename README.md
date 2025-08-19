# Deep Neural Network (DNN) Training for H2DF Engine Models

This repository contains MATLAB® code to train deep neural networks (DNNs) and gated recurrent unit (GRU)-based models for the **Hydrogen–Diesel Dual-Fuel (H₂DF) project at the University of Alberta (UofA)**.  
The models are based on experimental data from the **4.5 L Hydrogen–Diesel Engine** at the **MECE Engine Lab, Edmonton**, where **pseudo-random binary sequence (PRBS)** signals were used to excite the actuators.  

> **PRBS** inputs are widely used in system identification as they provide persistently exciting excitation patterns, enabling reliable training data collection across the full operating space.  

The trained models are later integrated into **nonlinear model predictive control (NMPC)** frameworks for advanced combustion control.  

**Authors:** 
- **Alexander Winkler**(alexander.winkler@rwth-aachen.de)
- Vasu Sharma(vasu.sharma@rwth-aachen.de)
- David Gordon(dgordon@ualberta.ca)
- Armin Norouzi(arminnorouzi2016@gmail.com)                     

---

## 🚀 Getting started

The repository contains two main directories at the root level:

├── H2DFmodel
│ ├── Scripts # Training scripts (run from here, MATLAB root)
│ ├── Functions # Helper functions for preprocessing and training
│ ├── Plots # Generated plots (matlab2tikz / .fig / .png)
│ └── Results # Trained models, performance metrics, evaluation plots
└── data # Experimental datasets (concatenated and split into train/val/test)


---

## 📊 Data Handling

- Multiple experimental datasets are **concatenated** and then **split into training, validation, and test sets**.  
- Splits follow best practices for supervised deep learning to ensure model generalization.  
- The datasets include actuator commands and measured engine responses under PRBS excitation.  

---

## 🧠 Model Variants

Two main models are trained and provided:  

1. **Model A (Feedback Model)**  
   - Includes IMEP (Indicated Mean Effective Pressure) feedback as an input feature.  
   - Represents the NMPC variant with closed-loop feedback integration.  

2. **Model B (No-Feedback Model)**  
   - Trained without IMEP feedback.  
   - Represents the NMPC variant with purely feedforward structure.  

Both model configurations are documented and later used in the **DNN/GRU-based NMPC** implementation (see [other repository](LINK)).  

---

## ⚙️ Usage

1. Verify `matlab2tikz` location in `setpath`.  
2. Execute `setpath` (drag and drop into MATLAB).  
3. Navigate to the **`\H2DFmodel\Scripts\`** folder and run all training scripts **from this directory (MATLAB root here)**.  
4. Training starts automatically using the **MATLAB Deep Learning Toolbox**.  
5. Training results (trained networks, performance plots, and evaluation metrics) are stored in the **`/H2DFmodel/Results/`** folder.  

---

## 📂 Results

- Trained models are included in the `/H2DFmodel/Results/` folder.  
- Each model includes:
  - Network architecture and weights  
  - Training/validation performance metrics  
  - Evaluation plots (loss curves, prediction vs. ground truth)  

---

## 📦 Dependencies

- MATLAB **R2024a or newer**  
- [Deep Learning Toolbox](https://de.mathworks.com/products/deep-learning.html)  
- [matlab2tikz](https://github.com/matlab2tikz/matlab2tikz) for exporting plots  

---

## 📑 Cite us

If you are using this code or the trained models, please cite the publications:  

- [Dummy1]  
- [Dummy2]  
- The experimental dataset publication on **Zenodo**  

---

## 📜 License

This project is licensed under the  
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).  
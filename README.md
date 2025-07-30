# PDTremorCode
This repository contains the code used for our second paper, focusing on tremor classification in Parkinson's disease using smartwatch sensor data (PADS dataset). The implementation includes data preprocessing, model training, ablation experiments, and visualization.

## Our code is located under code/second_paper.

## Project Structure
<pre markdown="1">
code/
└── second_paper/
    ├── checkpoint/                      # Trained model checkpoints
    ├── filternet/, layers/, models/    # Supporting model components
    ├── test/                            # Test scripts
    ├── two_model_confu/                 # Main training/testing logic and core modules
        ├── paifilter/, tcanet/, timenet_xiugai/, tps/
        ├── main_two.py                  # Main training entry
        ├── test_model.py                # Model evaluation script
        ├── model.py                    # Model assembly, config and ablation control
        ├── t_sne.py                    # Visualization of learned features (e.g., t-SNE)
        └── ...                         # Other utility scripts for ablation
    ├── load_data.py                   # Dataset loader (PADS dataset)
    ├── earlystoping.py, Nadam.py      # Early stopping, optimizer definition
    └── ...                            # Configs, device setup, utilities, etc.
</pre>

## Dataset
1.We used the PADS – Parkinson’s Disease Smartwatch dataset. PADS is a Parkinson’s disease research dataset collected using smartwatches and smartphones, covering a wide range of PD patients, individuals with other movement disorders, and healthy controls. Participants wore smartwatches on both wrists and performed 11 interactive movement tasks designed by neurologists to elicit subtle motor abnormalities. The devices synchronously recorded acceleration and rotation sensor signals, resulting in 5,159 measurement instances from 469 individuals. The dataset also includes detailed annotations on demographics, medical history, and non-motor symptoms of PD, making it well-suited for training and validating sensor-based models for movement disorder analysis.

2.You can download it from the following website: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

3.In addition to the original partial processing code included with the dataset, you can find a `.ipynb` file under `code/second_paper` that contains our additional data processing. You only need to modify the file paths accordingly.

## Setup
The code runs on Windows 11.

It is compatible with Python 3.8.

Other dependencies can be installed using:
pip install -r code/second_paper/requirements.txt

## Run
First, you need to navigate to the `code/second_paper` directory, then you can perform the following operations:

- In the directory `code\second_paper\two_model_confu`, there is a file named `main_two.py`. Running this file will execute the complete training process.

- In the directory `code\second_paper\two_model_confu`, there is a file named `test_model.py` which can be used for model testing.

- In the directory `code\second_paper\two_model_confu`, the file `model.py` contains the implementation of various modules mentioned in the paper as well as their parameter configurations. Additionally, you can add or remove modules in this file to perform ablation experiments.

- Within the directory `code\second_paper\two_model_confu`, there are four folders:  
  `paifilter`, `tcanet`, `timenet_xiugai`, and `tps`, each containing the full implementation code for the respective modules.

- The file `load_data.py` located in the `code/second_paper` directory handles dataset loading and also includes code for early stopping and optimizer settings.

- The directory `code\second_paper\two_model_confu` also contains some visualization code for the ablation experiments in the paper, such as `t_sne.py`.


### The full execution process is as follows:：
Step 1: Change directory
cd `code/second_paper`

Step 2: Run training
python `two_model_confu/main_two.py`

Step 3: Run model testing
python `two_model_confu/test_model.py`

Ablation & Modular Design
The file two_model_confu/model.py defines the overall model structure.
You can enable/disable specific modules or change configurations to perform ablation experiments.
Key modules (with full implementation) are located in:
`two_model_confu/paifilter`
`two_model_confu/tcanet`
`two_model_confu/timenet_xiugai`
`two_model_confu/tps`

Visualization
The script `two_model_confu/t_sne.py` provides t-SNE visualization for evaluating feature separability in embedding space.
Other utility scripts are available for analyzing similarity matrices and class attention (e.g., 计算相似性矩阵.py, 类别权重.py).

## Acknowledgments
This project is based on the PADS dataset from PhysioNet. We thank the original data contributors.
Varghese, J., Brenner, A., Plagwitz, L., van Alen, C., Fujarski, M., & Warnecke, T. (2024). PADS - Parkinsons Disease Smartwatch dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/m0w9-zx22

Varghese, J., Brenner, A., Fujarski, M., van Alen, C.M., Plagwitz, L., & Warnecke, T. (2024). Machine Learning in the Parkinson's disease smartwatch (PADS) dataset. npj Parkinsons Dis. 10, 9.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.





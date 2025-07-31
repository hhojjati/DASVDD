# DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection

DASVDD is an anomaly detection framework that combines deep autoencoders with Support Vector Data Description (SVDD). It supports multiple datasets and is designed for robust one-class classification tasks.


## 🚀 Features

- Dataset-specific autoencoders (e.g., PIMA, Speech)
- SVDD-inspired loss for anomaly detection
- Easy to modify and extend for other datasets

## 🔧 Installation

```
git clone https://github.com/Armanfard-Lab/DASVDD.git
cd DASVDD

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


## 🧪 Usage

Run DASVDD on the MNIST dataset:

```
python3 main.py --dataset NAME_OF_THE_DATASET --targest_class TARGET_CLASS_NO --epochs NUM_EPOCHS
```
Supported datasets: `MNIST`,`FMNIST`,`CIFAR`,`PIMA`,`Speech`.

| Argument        | Description                        | Default     |
| --------------- | ---------------------------------- | ----------- |
| `--dataset`     | Dataset to use:                    | *required*  |
| `--target_class`| Only for MNIST, FMNIST, CIFA       | `0`         |
| `--epochs`      | Number of training epochs          | `30`        |
| `--batch_size`  | Batch size for training            | `128`       |


## 📝 Citation

You can find the preprint of our paper on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10314785).

Please cite our paper if you use the results of our work.

```
@ARTICLE{DASVDD,
  author={Hojjati, Hadi and Armanfard, Narges},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection}, 
  year={2024},
  volume={36},
  number={8},
  pages={3739-3750},
  keywords={Anomaly detection;Training;Task analysis;Support vector machines;Image reconstruction;Data models;Benchmark testing;Anomaly detection;deep autoencoder;deep learning;support vector data descriptor},
  doi={10.1109/TKDE.2023.3328882}}

```

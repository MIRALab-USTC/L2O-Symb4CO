# Rethinking Branching on Exact Combinatorial Optimization Solver: The First Deep Symbolic Discovery Framework
This is the code of paper **Rethinking Branching on Exact Combinatorial Optimization Solver: The First Deep Symbolic Discovery Framework**. [[paper]](https://openreview.net/forum?id=jKhNBulNMh&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))

## Installation

### Revidsed scip 6.0.1

We revise the official scip (version 6.0.1) to add interfaces for Symb4CO. To install the revised scip:
1. Download the official scipoptsuite 6.0.1 from [here](https://scipopt.org/download.php?fname=scipoptsuite-6.0.1.tgz).
2. Replace the folder `./scipoptsuite-6.0.1/scip` (i.e., the folder of scip) with our revised version provided [here](https://github.com/MIRALab-USTC/scip/tree/symb4co-iclr2024). 
> Attention! The revised scip is in the `symb4co-iclr2024` branch, **not the `main`**!
3. Installing the revised scip 6.0.1 via cmake following the official [instructions](https://scipopt.org/doc/html/md_INSTALL.php).

P.S.: You can refer to the git commits of our revised scip 6.0.1 to identify the changes, compared with the original scip 6.0.1 and the revised version provided in [learn2branch](https://github.com/ds4dm/learn2branch/blob/master/scip_patch/vanillafullstrong.patch).

### Revised PySCIPOpt

We revise the PySCIPOpt to add interfaces for Symb4CO. To install the revised PySCIPOpt:
1. Specify the installation path of scip 6.0.1
```bash
export SCIPOPTDIR='/path/to/where/you/installation'
```
2. Install the revised PySCIPOpt
```bash
pip install git+https://github.com/MIRALab-USTC/PySCIPOpt.git@symb4co-iclr2024
```
> Attention! The revised PySCIPOpt is in the `symb4co-iclr2024` branch, **not the `master`**!

P.S.: You can refer to the last git commit of our revised PySCIPOpt to identify what has been revised, compared with the version provided in [learn2branch](https://github.com/ds4dm/PySCIPOpt/tree/ml-branching).

### Python Environment

We list the required python 3.9 packages in `./requirement.txt`. 

P.S.: We found inconsistent Pytorch might lead to unknown RuntimeError.

## Instructions for Execution

### Generating Instances and Datasets
The instance and the dataset generation is based on the [codes](https://github.com/pg2455/Hybrid-learn2branch) implemented by Gupta et al. To generate them, run

```bash
# generate instances
python 01_generate_instances.py cauctions
# generate the dataset
python 02_generate_dataset.py cauctions
# cauctions can be replaced to indset, setcover, and facilities
```

### Inference

If you do not want to train new symbolic policies, you can directly use our trained ones in `./results/expressions` via
```bash
python 05_evaluate.py instance_kwargs.instance_type=cauctions
```
You can also change the default evaluation settings via modifying `./settings/eval.yaml`.

To test the policies on larger instances, run
```bash
# medium
python 05_evaluate.py instance_kwargs.instance_type=cauctions 'instance_kwargs.dataset_type="200_1000"'
# hard
python 05_evaluate.py instance_kwargs.instance_type=cauctions 'instance_kwargs.dataset_type="300_1500"'
```

### Training and Validation

To train symbolic policies, run
```bash
python 03_train.py instance_kwargs.instance_type=cauctions
```

Then, to deploy learned symbolic policies to validation sets to select the best performed one, run
```bash
python 05_evaluate.py instance_kwargs.instance_type=cauctions exp_name=valid
```

### Test on Collected Dataset

To test the imitation learning accuracy on collected datasets, run
```bash
python 04_test.py cauctions
```

# Chisco 

**[Chisco: An EEG-based BCI dataset for decoding of imagined speech](https://www.nature.com/articles/s41597-024-04114-1)**

## **Abstract**
The rapid advancement of deep learning has enabled Brain-Computer Interfaces (BCIs) technology, particularly neural decoding techniques, to achieve higher accuracy and deeper levels of interpretation. Interest in decoding imagined speech has significantly increased because its concept akin to ``mind reading''. However, previous studies on decoding neural language have predominantly focused on brain activity patterns during human reading. The absence of imagined speech electroencephalography (EEG) datasets has constrained further research in this field. We present the *Chinese Imagined Speech Corpus* (Chisco), including over 20,000 sentences of high-density EEG recordings of imagined speech from healthy adults. Each subject's EEG data exceeds 900 minutes, representing the largest dataset per individual currently available for decoding neural language to date. Furthermore, the experimental stimuli include over 6,000 everyday phrases across 39 semantic categories, covering nearly all aspects of daily language. We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

## **Supplements**
In addition to the three participants mentioned in the paper, we collected and validated data from two additional participants. The data were acquired using the same experimental paradigm and are accessible via the same Chisco link.

## **Reproducing the Paper Results**

To reproduce the results from the paper, please follow the configurations provided below:

### **Model Configuration：**
```
python -u EEGclassify.py --rand_guess 0 --lr1 5e-4 --epoch 100 --layer 1 --pooling mean --dataset imagine_decode --sub "01" --cls 39 --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --subset_ratio 1
```
### **SBATCH Parameters:**
We ran our code on a SLURM cluster server. The following details may not be critical for reproducing the results presented in the paper, but they are provided here for reference if needed.
```bash
#!/bin/zsh
#SBATCH -p compute 
#SBATCH -N 1                                  # Request 1 node
#SBATCH --ntasks-per-node=1                   # 1 process per node
#SBATCH --cpus-per-task=4                     # Use 4 CPU cores per task
#SBATCH --gres=gpu:a100-pcie-80gb:1           # Request 1 A100 GPU
#SBATCH --mem=100G                            # Allocate 100GB memory
source ~/.zshrc
```

## **Citation**

```bibtex
@article{Zhang2024,
  author = {Zihan Zhang and Xiao Ding and Yu Bao and Yi Zhao and Xia Liang and Bing Qin and Ting Liu},
  title = {Chisco: An EEG-based BCI dataset for decoding of imagined speech},
  journal = {Scientific Data},
  volume = {11},
  number = {1},
  pages = {1265},
  year = {2024},
  month = {November},
  doi = {10.1038/s41597-024-04114-1},
  url = {https://doi.org/10.1038/s41597-024-04114-1},
  issn = {2052-4463}
}
```

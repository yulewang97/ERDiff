# Extraction and recovery of spatio-temporal structure in latent dynamics alignment with diffusion model (ERDiff)

## **Installation Guide**
If using conda, [placeholder for conda installation instructions]

If using venv, set up a new virtual environment by running `python3 -m venv erdiff`

Then initialize it with `source erdiff/bin/activate`

Install the required packages in the new environment. Requirements are specified in setup.py. Run with: 

```markdown
$ python3 -m pip install --upgrade pip
$ python3 -m pip install -e .
```

 
  
## **Train & Align**



### 1. **Source Domain: Cooperative Training**

```markdown
$ python3 VAE_Diffusion_CoTrain.py
```



### 2. Target Domain: Maximum Likelihood Alignment

```markdown
$ python3 MLA.py
```

### 

## **Visualization**

###  ![results](images/results_aligned.png)





## **Citation**


[arxiv](https://arxiv.org/abs/2306.06138)

```markdown
@article{wang2023extraction,
  title={Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model},
  author={Wang, Yule and Wu, Zijing and Li, Chengrui and Wu, Anqi},
  journal={arXiv preprint arXiv:2306.06138},
  year={2023}
}
```

### 

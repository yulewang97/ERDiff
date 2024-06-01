<h2>Extraction and recovery of spatio-temporal structure in latent dynamics alignment with diffusion models [NeurIPS'2023 Spotlight]</h2>

<div align='center' ><font size='4'>Yule Wang, Zijing Wu, Chengrui Li, and Anqi Wu</font></div>

<div align='center' ><font size='5'>Georgia Institute of Technology</font></div>

<div align='center' ><font size='5'>Atlanta, GA, USA</font></div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                 <img src="images/GTVertical_RGB.png" alt="GTVertical_RGB" width="150" /><img src="images/127633222.png" alt="GTVertical_RGB" width="125" />

## 

<div align=center><img src="images/ERDiff_main_github.png", width="650"></div>

## **Environment Setup**

To install the required dependancies using conda, run:

```markdown
$ conda create --name erdiff --file requirements.txt
```

To install the required dependancies using Python virtual environment, run:
```markdown
$ python3 -m venv erdiff
$ source erdiff/bin/activate
$ python3 -m pip install --upgrade pip
$ python3 -m pip install -e .
```

 

## **Training & Alignment**



### 1. **Source Domain: Training**

```markdown
$ python3 VAE_Diffusion_CoTrain.py
```



### 2. Target Domain: Maximum Likelihood Alignment

```markdown
$ python3 MLA.py
```

### 

## **Latent Dynamics Visualization**

###  ![results](images/results_aligned.png)


## **Cited as**

```markdown
@article{wang2024extraction,
  title={Extraction and recovery of spatio-temporal structure in latent dynamics alignment with diffusion model},
  author={Wang, Yule and Wu, Zijing and Li, Chengrui and Wu, Anqi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

### **ERDiff Poster for NeurIPS 2023**

###  ![results](images/ERDiff_NeurIPS23_Poster_Final.png)


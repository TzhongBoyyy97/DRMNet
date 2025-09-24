# Domain Generalization for Pulmonary Nodule Detection via Distributionally-Regularized Mamba

This paper has been accepted by MICCAI 2025. And our code is based on [SANet](https://github.com/mj129/SANet), [NoduleNet](https://github.com/uci-cbcl/NoduleNet), and [SGDA](https://github.com/Ruixxxx/SGDA).

We propose the FASS module to incorporate global features by leveraging the spatial relationship between pulmonary nodules and vascular structures to integrate global features. 

We introduce the RADA module further to align pulmonary nodule features from different source domains, enabling effective generalization to the target domain. 

We construct a domain generalization dataset GPND for pulmonary nodule detection. It contains two private datasets and two public datasets. 

Comparison of PN9, LUNA16, PONSD, and GGO. The first row represents some nodule samples, and the second row indicates the diameter distributions in four datasets. Four subsets introduce clear domain shifts:
<img width="1739" height="834" alt="fig1" src="https://github.com/user-attachments/assets/a57f45c5-34fe-4e45-9e92-718fae970971" />

An overview of the proposed DRMNet. DRMNet is an end-to-end neural network with two key components: the FASS and RADA modules:
<img width="1593" height="862" alt="fig2" src="https://github.com/user-attachments/assets/ea7675e8-004e-4b0f-90c5-7e2c6bd2e6fb" />


This code and our data are licensed for non-commerical research purpose only.

<!-- If you are using the code/model/data provided here in a publication, please consider citing -->

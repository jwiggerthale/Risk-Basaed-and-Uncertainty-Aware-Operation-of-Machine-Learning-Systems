# Risk-Based and Uncertainty Aware Operation of Machine Learning Systems
This repo contains the code for my PhD Thesis "Risk Based and Uncertainty Aware Operation of Machine Learning Systems in High-Stakes Environments". It is divided in three folders (Calibration, Risk, SMiLE), each representing one chapter from my thesis. 


# Preface
The repo emerged from different projects conducted throughout my thesis. Many scripts are repititive and it would be possible to combine them in a lean codebase combining all experiments and datasets in few scripts. I refrained from doing so since it would potentially have led to structural inconsistency. I wan not willing to take that risk during my studies. 

# Outline
The thesis investigaed how ML based systems in critial decision systems (CDS, the socio-technical system of human end users and ML models) can be developed and operated in such a way that safety is maximized (i.e. probability of costly errors minimized). The thesis was divided in four main Chapters:
1)  Regulatory requirements imposed on ML-based systems in CDS
2) Comparison of Uncertainty Qunatification (UQ) methods with regard to calibration quality
3) Comparison of Cost Sensitive Learning (CSL) methods to a concept called Risk-Based Decision making (BD)
4) Intoroduction and evaluation of the Safe Machine Learning (SMiLE) framework

All experiments were conducted using three identical datasets, namely
1) the CIFAR-100 dataset and the CIFAR-100C dataset as domain shift (DS) dataset,
2) the Severstal dataset and augmented as well as corrupted versions of the dataset as instances of DS data, and 
3) the diabetes dataset as proposed by Garnder et al. (https://tableshift.org/).


# Comparison of Uncertainty Qunatification Methods
Comparison of UQ methods covered 
1) softmax outputs
2) Monte Carlo (MC) dropout
3) Ensembles
4) Evidential Deep Learning (EDL)

  Methods (1) - (3) were implemented using a heteroscedastic loss function to capture Aleatoric Uncertainty (AU) as well as Epistemic Uncertainty (EU). Results indicate that ensembles provide best calibrated predictive distributions and highest discriminative potential of uncertainty scores across datasets. 

  

  # Comparison of Cost Sensitive Learning Methods
  The thesis introduced the concept of Risk-Based decsion making (RBD). The main idea is to apply Bayesian decison theory to ML model predictions. Given a cost matrix quantifying the cost of different misclassifications and calibrated model predictions, the Expected Prediction Risk (EPR) (product of probailty and severity) of each chosen label can be calculated. The label is chosen to minimize EPR. 

  The thesis compared how the concept of RBD reduces operational risk measured via cost sensitive objectives compared to CSL. Results show that RBD usually provides comparable or superior reduction in operational risk compared to RBD while maintaining accuracy. 

  # Evaluation of the Safe Machine Learning Framework
  Based on the insights on calibration and CSL, the thesis propes the SMiLe framework as a lifecycle oriented framework for development and deplyoment of ML-based systems. The framework composes of 7 phases: 

## Business understanding
- Objectives: understand the aim of the ML-based system to be created as well as underlying mechanisms affecting data quality and potential asymmetric cost structures; definition of performance criteria
- Core activities: define intended use, misuse, and risk boundaries; examine data generation process with regard to bias, noise or related issues; set acceptance criteria; define oversight concept

## Data engineering
- Objective: gather representative data covering all anticipated scenarios
- Core activities: label and curate data with expert oversight; potentially use HiL approach; perform quality checks on annotations; document data sources, labeling procedures, and any biases or limitations in the dataset; data augmentation to fill gaps; ensure personal or sensitive data handling complies with regulations

## Model engineering
- Objective: design and train model(s)
- Core activities: design and train model(s); train alternative models for redundancy; perform internal testing with risk-aware metrics; verify models' uncertainty estimate is reliable; implement DS mechanisms

## Model validation
- Objective: assess system performance under assumption of abstention policies on held-out test data
- Core activities: ensure, global model explanations are stringent with domain knowledge; define abstention policies; evaluate the trained models on held-out test data and simulation scenarios that mimic real-world conditions; HiL trials: deployment in a shadow mode or pilot phase where a human expert reviews every model decision or a statistically significant sample of decisions;  generate a list of failure modes or conditions where the model struggled; feed this information back to development if needed
 
## Model verification
- Objective: define the model's  ODD or conditions under which it is approved to operate
- Core activities: verify through targeted tests that perform boundary analysis and test DS mechanisms within the ODD; ensure all regulatory checks are complete


## Deployment
- Objective: roll out the model into the production environment
- Core activities: set up monitoring hooks; implement decision threshold based warnings and human oversight triggers; ensure transparency; training for end users; implement real time diagnostics

## Monitoring and maintenance
- Objective: detect and mitigate performance degradation
- Core activities: continuous performance monitoring; gather user feedback and incident reports; retrain or fine tune if required; incident management


## Validation of the framework
The framework was validated exemplarily on all three datasets. Results indicate that the framework can efficiently reduce operational risks of ML-based systems and increase accuracy. Especially combination of two models and uncertainty based abstention policies provide largest marginal gains. Notably, inproved performance is purchased by higher abstention rates, i.e. more workload for (human) fallback mechanisms. 

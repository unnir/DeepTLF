# DeepTLF - Robust Deep Learning on Tabular Data by Transforming Heterogeneous Data into Homogeneous Data

---------

![alt text](pipeline.png)



---------
## Abstract
Although deep neural networks (DNNs) constitute the state of the art in many tasks based on visual, audio, or text data,
their performance on heterogeneous, tabular data is typically inferior to that of decision tree ensembles. To bridge the gap
between the difficulty of DNNs to handle tabular data and leverage the flexibility of deep learning under input heterogeneity,
we propose DeepTLF, a framework for deep tabular learning. The core idea of our method is to transform the heterogeneous
input data into homogeneous data to boost the performance of DNNs considerably. For the transformation step, we develop
a novel knowledge distillations approach, TreeDrivenEncoder, which exploits the structure of decision trees trained on the
available heterogeneous data to map the original input vectors onto homogeneous vectors that a DNN can use to improve
the predictive performance. Within the proposed framework, we also address the issue of the multimodal learning, since
it is challenging to apply decision tree ensemble methods when other data modalities are present. Through extensive and
challenging experiments on various real-world datasets, we demonstrate that the DeepTLF pipeline leads to higher predictive
performance. On average, our framework shows 19.6% performance improvement in comparison to DNNs.


---------
## How to use? 
DeepTLF follows the scikit-learn API: 

```python
from src import DeepTFL

dtlf_model = DeepTFL(n_est=23, max_depth=3, drop=0.23, n_layers=4, task='class')
dtlf_model.fit(X_train,y_train)

dtlf_y_hat = dtlf_model.predict(X_test)
```
---------
## Citation
If you use this codebase, please cite our work:

```bibtex
@article{borisov2022deeptlf,
  title={DeepTLF: robust deep neural networks for heterogeneous tabular data},
  author={Borisov, Vadim and Broelemann, Klaus and Kasneci, Enkelejda and Kasneci, Gjergji},
  journal={International Journal of Data Science and Analytics},
  pages={1--16},
  year={2022},
  publisher={Springer}
}
```

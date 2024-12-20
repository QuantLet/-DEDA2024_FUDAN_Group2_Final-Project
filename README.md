<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
======================== DEDA-Fudan-Group2: Final Project ========================

Name of QuantLet: DEDA_Final_Project_DEDA_Fudan_Group2

Published in: DEDA class

Description: 'Financial Risks across Stock and Bond Markets — Analyses with Machine Learning Methods'

Keywords: cluster analysis, minumum spanning tree, machine learning, graph neural networks

Authors: Wang Sukun, Huang Xiwen, Chen Zehao, Li Siyuan, Wang He

Submitted: Wed, October 23 2024 by Huang Xiwen

The data used in the project are all sourced from: Xiaojian Niu, Jiahao Gong, Sukun Wang, Haofan Qiang. Research on the Cascading Interdependent Mechanism of Upside and Downside Financial Risks across Stock and Bond Markets: Cascading Interdependent Multiplex Networks Approach from an Industry Perspective, 2024, Working Paper.

The project is divided into three parts, described as follows: 

1 Clustering (Path: FinalProject/1_Clustering)

Entities: 29 industries (classified by Shenwan first-level industry categories);

Time and Frequency: June 5, 2008, to December 30, 2022, a total of 710 network adjacency matrices. Rolling window calculation is used, with a window of 250 days and a step size of one week (5 trading days).

Clustering metric: The total intensity of risk linkage for industry j, Risk_j. This is obtained by summing the risk contagion intensity (row j) and risk bearing intensity (column j) in the bond risk linkage network adjacency matrix. The adjacency matrix of the bond risk linkage network is calculated using the "Nonlinear Granger Causality Test Based on Leave-One-Out and Data Sharpening" on bond realized volatility (RV).

1.2.1 Spectral Clustering; Three display methods: Spectral clustering graph (graph_spectral_clustering.jpg); t-SNE graph (graph_tsne.jpg); UMAP graph (graph_umap.jpg)

1.2.2 Hierarchical Clustering One display method: Iris clustering tree (graph_iris_clustering.jpg)

2 Minimum Spanning Tree (Path: FinalProject/2_MST)

Graph nodes: 58 nodes, representing 29 stock nodes and 29 bond nodes from different industries.

Graph links: The risk linkage intensity between different asset types within industries.

This section analyzes the risk linkage networks of the following four phases: Stock bull market (July 2014 - June 2015) Stock bear market (June 2015 - January 2016) Bond bull market (November 2013 - October 2016) Bond bear market (October 2016 - November 2017)

Image results: See “MST_vol_bad_stock_bull_nonlinear_network.jpg” and three other images.

Industry names: The industry names corresponding to node numbers in the images (in Chinese) can be found in “industry_mapping_rule.txt”; the industry names (in English) can be found in “label_vol_bad_bond_bear_nonlinear_network.txt”.

Result Interpretation: The minimum spanning tree retains the most critical part of the risk linkage network. If certain industries appear in all four phases, it indicates that these industries play an important role in risk linkage. Especially the industries occupying central positions, which highlights their critical status.

3 Machine Learning Prediction (Path: FinalProject/3_ML)

Data structure: The downside risk linkage network in the stock market is represented by an adjacency matrix, composed of 29 industry stock asset nodes and their links.

Prediction approach: The following periods are designated as financial crisis periods, with the network characteristics during these periods labeled as 1, and those outside as 0. The model is trained to predict future financial crisis periods and evaluate its performance: (datetime(2007, 7, 26), datetime(2009, 12, 31)), # Global financial crisis (datetime(2013, 6, 7), datetime(2013, 12, 31)), # Bank liquidity crisis (datetime(2015, 6, 15), datetime(2016, 12, 20)) # Stock market crash

Hidden layers: hidden_channels=65; Classification types: num_classes=2; Node features: num_node_features=1; Link features: num_edge_features=1; Training iterations: epoch = 350.

Training process accuracy time series graph: See “graph_accuracy_curve.jpg”.

Warning signal graph for the training set: See “graph_prediction_results.jpg”.

Sequential training results of Accuracy, F1-score, and F-Alarm: See “accuraciesGGNN.csv”.

```

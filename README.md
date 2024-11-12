<h1 align="center"> Experiments Introduction for Penalty-based Simple Bilevel Optimization</h1>

## Penalty-based Simple Bilevel Optimization
This document introduces the MATLAB codes for the penalty-based methods (PB-APG, aPB-APG, PB-APG-sc,and aPB-APG-sc) that solve the simple bilevel optimization problems, as described in the article "Penalty-based Methods for Simple Bilevel Optimization under Hölderian Error Bounds".

All simulations are implemented using MATLAB R2023a on a PC running Windows 11 with an AMD (R) Ryzen (TM) R7-7840H CPU (3.80GHz) and 16GB RAM. It is necessary to note that the code requires adding the appropriate paths before execution.

We consider the following two problems in the experiments.

## Logistic Regression Problem (LRP)
To run the code, you should first download the LIBSVM package from https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download and use the "make.m" file in the LIBSVM Matlab package to generate the "libsvmread.mexw64" file. Then, add the appropriate paths to the code. Next, you should download the data file "a1a.t" from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t, rename it as "a1a.t" and place it in the "Logistic Regression Problem (LRP)" folder (We have already included the necessary files and data in our folder.). Then, run the file "main_Logistic_a1a.m" in the same folder.

## Least Squares Regression Problem (LSRP)
Before running the code, you should download the YearPredictionMSD dataset from https://archive.ics.uci.edu/dataset/203/yearpredictionmsd, rename it as "YearPredictionMSD.txt" and place it in the "Least Squares Regression Problem (LSRP)" folder. Then, run the file "main_LeastSquare.m" in the same folder.

## Detailed Parameters Setting
For the two experiments in Section 4, no code modification is required in the "main_Logistic_a1a.m" and "main_LeastSquare.m" files.

To conduct the experiments in Appendix F.3, set the parameter 'gamma_tot' in the 'PB-APG' section of the "main_Logistic_a1a.m" and "main_LeastSquare.m" files to either '20000' or '500000'. 

To conduct the experiments in Appendix F.4, set the parameter 'epsilon' in the 'PB-APG' section of the "main_Logistic_a1a.m" and "main_LeastSquare.m" files to either '1e-4' or '1e-7'. Correspondingly, set the parameter 'tol_vec' in the 'aPB-APG' section of the same files to either '[1e-0;1e-1;1e-2;1e-3;1e-4]' or '[1e-3;1e-4;1e-5;1e-6;1e-7]'. Tip: If you have adjusted the parameter 'gamma_tot' for the experiments in Appendix F.3, please ensure that you update it to '100000' before CONDUCTING the experiments in Appendix F.4.

# Factor Model and Autoregressive Analysis

## Overview
This repository contains MATLAB code for analyzing various statistical models, focusing on factor modeling and autoregressive processes. The analysis involves data manipulation, dimensionality reduction, and model evaluation using criteria like BIC and AIC.

## Tasks

### Task 5.1: Prepare a Factor Model
- **Data Loading**: Loaded dataset from 'Report5 1'.
- **Factor Estimation**: Estimated 20 factors from the matrix Y.
- **Variance Plot**: Created a bar plot displaying the variances explained by the first 20 factors.
- **Optimal Factors**: Employed criteria to assess the optimal number of factors and interpreted the results, noting potential discrepancies in conclusions based on different criteria.

### Task 5.2: Estimate an Autoregressive Model with Lasso Estimator
- **Data Loading**: Loaded dataset from 'Report5 2'.
- **Model Specification**: Defined an autoregressive model.
- **Lasso Estimation**: Estimated using a logarithmic grid of lambda.
- **Optimal Lambda Selection**: 
  - Selected optimal lambda using BIC.
  - Conducted cross-validation (CV) with 10 folds.
  - Implemented a corrected version of CV with 10 folds.
- **Model Variables**: Identified and recognized the significant variables retained in the final models for each estimation method.

### Task 5.3: Dimension Reduction
#### Part 1: Factor Modeling with Electricity Price Data
1. **Data Loading**: Loaded 'POLEX.csv' containing electricity price data.
2. **Subset Selection**: Focused on data corresponding to a specific month based on the first letter of the surname.
3. **Model Construction**: Defined model.
4. **Factor Estimation**: Estimated 10 factors from the independent variables matrix.
5. **Loading Plots**: Plotted the first three loadings.
6. **Optimal Factors Assessment**: Employed criteria to determine the optimal number of factors and interpreted results.
7. **OLS Estimation**: Estimated OLS model with the optimal number of factors and calculated BIC.
8. **Lasso Estimation**: Estimated Lasso with a default grid of lambda.
9. **Optimal Lambda Selection**: 
   - Selected optimal lambda using AIC.
   - Conducted CV with 5 folds and a corrected version of CV.
10. **Final Model Variables**: For each of the 5 models (2 PCA + 3 LASSO), counted retained variables and calculated BIC, compiling results into a table.

#### Part 2: Extended Analysis with Multiple Variables
1. **Data Loading**: Loaded the same 'POLEX.csv' file containing electricity price, load, renewable generation, EUA price, and natural gas price data.
2. **Subset Selection**: Similar to part 1, focusing on data corresponding to a specific month.
3. **Correlation Testing**: Analyzed correlation between price and load, as well as price and renewable energy generation, determining the nature of correlation.
4. **Normality Testing**: Used the Lilliefors test to check the normality of the price series.
5. **Model Construction**:
   - Defined initial autoregressive model.
   - Created a matrix including other relevant variables.
   - Combined the autoregressive matrix with additional variables to form a complete X matrix.
6. **Linear Regression Estimation**: Estimated the model using linear regression and calculated BIC.
7. **T-test for Variable Relevance**: Performed a t-test to identify irrelevant variables, excluding them and re-estimating the regression model.
8. **Factor Estimation**: Estimated 20 factors from the independent variables matrix.
9. **IPC Criterion Assessment**: Used the IPC criterion to assess the optimal number of factors.
10. **OLS Model with Factors**: Estimated OLS model with the optimal number of factors, calculating BIC.
11. **Lasso Estimation**: Estimated Lasso using the default grid of lambda and 7 folds of CV.
12. **Optimal Lambda Selection**: Used corrected CV to determine optimal lambda and calculated BIC for the model.
13. **Final Variables Table**: Prepared a table showing BIC and the number of retained variables for each modeling approach (linear regression, post-t-test linear regression, PCA, and LASSO).
14. **Residual Analysis**: Conducted autocorrelation checks with Q-test for 20 lags and assessed homoscedasticity with the Breusch-Pagan LM test for the model with the lowest BIC.

## Programming Language
This project was implemented in **MATLAB**.

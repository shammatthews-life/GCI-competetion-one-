# GCI-competetion-one-
my attend in GCI

## Competition Trials

- `Trial_1/data_cleaning_correlations.ipynb`: exploratory analysis, missing-value handling, outlier diagnostics, feature engineering, correlations, and preprocessing.
- `Trial_2/gci_0_762337_credit_term.py`: credit-term experiment (`AMT_CREDIT / AMT_ANNUITY`); its header identifies approximately 0.762337 as a local OOF/holdout AUC, not the public champion score.
- `Trial_3/gci_0_76370_reconstruction.py`: reconstruction of the feature/model family associated with the reported 0.76370 public result; it is not guaranteed to reproduce the original submission bit-for-bit.

The competition input and output data are not included. The two scripts expect an `input/` directory containing the provided CSV files in the current working directory. They write generated submissions under `output/`.

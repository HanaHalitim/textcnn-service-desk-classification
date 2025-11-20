# Data Directory

This directory contains the synthetic IT service desk ticket dataset.

## Files

- `words.json`: Vocabulary list of all unique tokens (sorted alphabetically)
- `text.json`: Tokenized ticket texts (list of lists of token strings)
- `labels.npy`: NumPy array of integer labels (0-4) corresponding to ticket classes

## Class Labels

- `0`: Access/Login Issues
- `1`: Software Bugs/Errors
- `2`: Hardware/Device Failure
- `3`: Network/System Outage
- `4`: New Request/Feature

## Generation

To regenerate the dataset, run from the project root:

```bash
python generate_data.py
```

This will create/overwrite the three data files in this directory.



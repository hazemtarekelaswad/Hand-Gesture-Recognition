# Hand Gesture Recognition

<!-- CMPN450 - PATTERN RECOGNITION Project -->

## Run on test images

1. Put your test images in `data`.
2. Run 
    ```bash
    python src/main.py ../data ../results
    ```
3. See the results in `results`

> It is better to rename images as 1, 2, 3 and so on.

## Requirements

```bash
pip install -r requirements.txt
```

## Directory Structure
```
Hand-Gesture-Recognition/
├───models
├───features
├───dataset
├───results
│   ├───results.txt
│   ├───time.txt
├───src
│   ├───main.py
│   ├───feature_extraction
│   ├───model_training
│   ├───pre-process
│   └───utils
│      └───common_functions.py
│      └───constants.py
├───.gitignore
├───project_requirements.pdf
├───project_report.pdf
├───README.md
├───requirements.txt
```

## Dataset

This [drive directory](https://drive.google.com/drive/folders/1o9wzwaJVfrbpCFJ0rIyed1QvARh0JAtn?usp=sharing) contains:
- `Dataset_0-5.zip`: containes Men and Women directories in which you find a directory of images for each number from 0 to 5.
- `dataset_sample.zip`: sample images from the original dataset.

*Download and move `Dataset_0-5.zip` after unzipping it into a directory `dataset`. If it does not exist, create it.* 


## Members

- [Ahmed Yasser](https://github.com/AhmedYasser155)
- [Arwa Ibrahim](https://github.com/ArwaShamardal)
- [Hazem Tarek](https://github.com/hazemtarekelaswad)
- [Saifeleslam Abdelrahman](https://github.com/Saif-El-Eslam)


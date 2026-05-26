# ML Logistic Regression - Internship Acceptance

This project solves the task from the PDF file:

- train a Logistic Regression model;
- predict whether a student will be accepted for a SoftServe internship;
- use features: Experience, Grade, EnglishLevel, Age, EntryTestScore;
- target: Accepted;
- store EnglishLevel as text values: Elementary, Pre-Intermediate, Intermediate, Upper-Intermediate, Advanced;
- build a plot of acceptance probability by English level and entry test score.

## How to run in Visual Studio Code

1. Create/open a folder for the project.
2. Put these files into the folder:
   - `main.py`
   - `requirements.txt`
   - `internship_candidates_cefr_final.csv`
3. Open the folder in VS Code.
4. Open Terminal in VS Code:
   - `Terminal -> New Terminal`
5. Create a virtual environment:

```bash
python -m venv .venv
```

6. Activate it.

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use this command once:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activation again.

On macOS/Linux:

```bash
source .venv/bin/activate
```

7. Install libraries:

```bash
pip install -r requirements.txt
```

8. Run the program:

```bash
python main.py
```

## Expected result

The program will print:

- model accuracy;
- confusion matrix;
- classification report;
- example prediction for one student.

It will also create this image:

```text
acceptance_probability.png
```

This image is the required plot of acceptance probability depending on English level and entry test score.

## What to upload to repository

Upload these files to GitHub:

- `main.py`
- `requirements.txt`
- `README.md`
- `internship_candidates_cefr_final.csv`
- `acceptance_probability.png` after running the code

Then submit the GitHub repository link.

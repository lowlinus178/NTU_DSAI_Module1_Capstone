# Streamlit Local Implementation Guide

This guide provides step-by-step instructions for launching the **Singapore Job Market Analysis Dashboard** in your local machine.

---
## 1. System Requirements

Before you begin, ensure your system meets the following requirements:

### Hardware
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** 2GB free space
- **Processor:** Any modern multi-core processor (Intel i5/AMD Ryzen 5 equivalent or better)

### Software
- **Python:** Version 3.8 or higher (3.10+ recommended)
- **Package Manager:** pip (comes with Python)
- **Terminal/Command Line:** Windows PowerShell, Command Prompt, macOS Terminal, or Linux Terminal

### Browser (for viewing the dashboard)
- Chrome, Firefox, Safari, or Edge (any modern web browser)

## 2. Dataset Information

### Primary Dataset

**File Name:** `SGJobData_cleaned_processed_compressed.csv.gz`

**Description:**
This is the processed and cleaned version of the Singapore job postings dataset containing ~1M+ rows of job listing data.

**Key Columns Used:**
- `title` - Job position title
- `categories` - Job category classification
- `employmentTypes` - Type of employment (Full-time, Part-time, Contract, etc.)
- `salary_minimum` - Minimum salary offered
- `salary_maximum` - Maximum salary offered
- `numberOfVacancies` - Number of positions available
- `metadata_originalPostingDate` - Job posting date
- `metadata_totalNumberJobApplication` - Total applications received
- `metadata_totalNumberOfView` - Total job views
- `postedCompany_name` - Hiring company name



## 3. File Location

Ensure that the following Streamlit python and associated csv gzip-compressed dataset files are downloaded and saved to the same local folder in your local machine:

 - `ntu_dsai_module_1_capstone_streamlit_app_final.py`
 - `SGJobData_cleaned_processed_compressed.csv.gz`


## 4. Launching the Dashboard

### Step 1: Navigate to the local folder where the files are located

Open Git Bash terminal and navigate to the correct local folder via running the command:

cd `folder path`

### Step 2: Run the Streamlit App

Execute the following command:

`streamlit run ntu_dsai_module_1_capstone_streamlit_app_final.py`

### Step 3: Access the Dashboard

After running the command, Streamlit will start a local development server. You'll see output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Open your browser and navigate to:** `http://localhost:8501`

The dashboard should load in your browser. You may need to wait a few seconds for the initial data processing as the dataset is about 55MB.



---


****Last Updated on:  10 Feb 2026****





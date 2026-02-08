# **NTU DSAI MODULE 1 CAPSTONE SUBMISSION**


## **Preamble**

- This document aims to summarise the steps which the DS 4 (Group 7) had taken to conceptualise, clean, and process the SG Job dataset to develop and deliver an interactive Streamlit dashboard for a fictitious business case.



## **Business Case & Objectives**

### ***Scenario:***

- A job market analyst at a public agency has been tasked to scan the Singapore job market to identify potential job categories where there are skills gap and/or perception-related issues for the purpose of more in-depth policy analysis and intervention.

- Specifically, the public agency is seeking to have an overall appreciation of the current Singapore job market in terms of job postings vis-a-vis application volumes to derive some preliminary insights as to:

    1. Which job roles are currently demanded by employers;
    2. Which job roles are not attracting sufficient job applicants; and
    3. Where the potential structural job-skills mismatch are in the prevailing Singapore workforce.   

### ***Target Users/Audience:***

- The preliminary insights derived from the Singapore job market data will be packaged as a dashboard presentation to the Senior Management of the public agency at a **Senior Management Forum chaired by the Permanent Secretary of the parent Ministry overseeing the public agency**.

- The **Chief Executive of the public agency** is seeking additional fiscal resources and mandate from the Permanent Secretary to formally commission a multi-sector Work Group to study and tackle emerging job-skills issues to ensure Singapore's economic sustainability over the longer term.

### ***Business Value Proposition:***

- The dashboard presentation would facilitate the public agency's objective of securing resources and mandate from its parent Ministry to commence a dedicated policy workstream as an integral part of the public agency's workplans for the fiscal year 2026/2027.

- It is envisaged that the dedicated policy workstream would bring a wealth of multi-sectorial inputs from relevant stakeholders to help formulate and implement more market-driven manpower upskilling programmes to secure better job prospects for Singapore job seekers and to bolster the fundamental skills base of the Singapore workforce.



## **Data "Extract-Transform-Load" (ETL) and Exploratory Data Analysis (EDA) Approach**

### ***Tools & Environment:***

- Python 3 with Pandas, NumPy, Matplotlib, Seaborn, and Plotly Express for data manipulation, cleaning, and interactive visualization. Dataset loaded from SGJobData.csv containing ~1.1M+ job postings.  

### ***Data Cleaning Steps:***

- Removed 100% null occupationId column and single-value status_id column;
- Dropped rows with missing values in critical fields, retaining ~95% of raw data (data integrity preserved);
- Standardized column names and handled whitespace inconsistencies.
  
### ***Date & Category Standardisation Steps:***

- Converted three date columns to datetime format;
- Parsed JSON-structured categories field to extract primary job category;
- Standardised salary data by calculating average salary from salary_minimum and salary_maximum ranges (range: $0–$250K+).

### ***Feature Engineering Steps:***

- Created temporal features (month_year, year, month) for trend analysis;
- Engineered salary bands (6 categories: Below 3K to Above 20K) for market segmentation;
- Extracted seniority levels (Senior/Mid-level/Junior) from job titles using regex patterns;
- Calculated demand metrics: applications-per-vacancy ratio and views-per-application ratio to assess market competition.

### ***Key EDA Insights:***

- **Weak Salary-Demand Correlation:** Higher pay doesn't necessarily attract more applicants, signalling potential skills gap and/or perception-related issues, which is a central theme shaping the dashboard's narrative;
- **Market Structural Imbalance:** Average applications per posting vary across job categories, with some roles oversaturated and others critically underserved;
- **Applications-per-Vacancy Tension:** Significant variation across job categories, revealing uneven supply-demand balance and competitive pressures;
- **Temporal Trends:** Monthly postings, applications, and vacancies show seasonal patterns and growth trajectories informing time-series dashboard views.



## **The Dashboard Presentation App - Salient Features & Highlights**

### ***Dashboard Presentation App Technology:***

- A **Streamlit dashboard app** is developed from the cleaned and processed dataset output from the preceding ETL & EDA steps.

### ***Dashboard Structure & Business Case Articulation:***

- To prevent information overload, the dashboard is succinctly structured into 4 easy-to-navigate sections:

    1. **Interactive Visual Explorer section.** This allows the target audience to filter jointly by ***Job Category***, ***Employment Type***, and ***Analysis Period*** to view several headline demand and supply-side statistical metrics. Specifically, the headline statistics capture **Total Job Postings**, **Total Applications**, **Total Vacancies**, **Total Hirers**, **Average Views Per Job**, and **Average Applicants Per Job** metrics. Juxtaposing these headline numbers is a dynamic word cloud which reflects the trending job titles and related keywords derived from the **Title** column of the dataset. Following these is a dynamic time-series chart comparing the associated job postings, vacancies and applicant volume trends;
       
    2. **Static Deep Dive: Contrasting Landscapes of Laggards & Leaders section.** This articulates the key insights in **three sets of visualisations contrasting the bottom versus top 10 job categories grouped by the Average Applicants Per Job** and **Average Applicants Per Vacancy metrics**. These seek to drive home the thematic message that the Singapore job market displays signs of structural demand/supply imbalance across job categories, with some attracting very low applicants while others tend to be over-subscribed. The two visualisations comprise side-by-side (value-axis synchronised) barchart comparison of the bottom versus top 10 job categories in terms of the two key metrics. The job categories shown essentially provide the clues as to the job categories which are demanded by employers (i.e. job categories reflected in both bottom and top 10 job categories), and the "laggard" job categories which garner little applicants (i.e. the bottom 10 job categories). The last visualisation to round up the key message comprises a side-by-side boxplot comparison of the log-transformed average salary distributions by the same bottom 10 versus top 10 job categories. These shows an interesting observation that the median salaries (i.e. the middle line in the boxes) do not vary much across both ends of the market, suggesting that payscales do not quite explain the stark differences in applicant volumes and refutes any initial impression that applicants are strongly attracted to the top 10 job categories by pecuniary factors or better remuneration terms.
 
    3. **Static Pay/Applicant Nexus: Weak Influence of Payscales on Applicant Volumes section.** This penultimate section uses a correlation matrix heatmap to reinforce the earlier observation that payscales do not quite explain the variations in applicant volumes between the two extreme ends of the market. It serves to add credence to the refutation of the initial hypothesis that better payscales or remuneration drive applicants' propensity to gravitate towards the top 10 job categories. Collectively, these provide persuasive visual (though possibly lacking statistical rigour) evidence that there are potentially other non-pecuniary factors like skills and/or perceptions driving applicant behaviour away from certain jobs as opposed to others. Such behaviours are indirectly causing structural misalignments in the Singapore workforce and unmet demands for certain job roles in the Singapore economy.
 
    4. **The Key Takeaways** section. This is self-explanatory, and rounds up the "laggards versus leaders" thematic narrative in a few high-impact points to influence Senior Management's deliberations and discussions on next steps.

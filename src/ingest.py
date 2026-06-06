"""
ingest.py — Data ingestion from Kaggle (offline) and JSearch API (live).

Usage:
    from src.ingest import load_offline_data, fetch_jsearch_jobs, save_raw_data
"""

import os
import json
import time
import random
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.utils import (
    DATA_DIR, RAW_CSV, SAMPLE_PARQUET, JSEARCH_API_KEY,
    clean_text, make_job_id, logger
)

# ─── Schema ───────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "job_id", "title", "company", "location", "city", "country",
    "remote", "salary_min", "salary_max", "salary_midpoint",
    "description", "skills_extracted", "seniority", "employment_type",
    "experience_required", "visa_possible", "date_posted", "source", "url",
]


# ─── Kaggle ingestion ─────────────────────────────────────────────────────────
def load_kaggle_data() -> pd.DataFrame:
    """
    Download and load the TechMap international job postings dataset via kagglehub.
    Falls back to sample data if Kaggle credentials are not set.
    """
    try:
        import kagglehub
        logger.info("Downloading Kaggle dataset via kagglehub...")
        path = kagglehub.dataset_download("techmap/international-job-postings-september-2021")
        logger.info(f"Dataset downloaded to: {path}")

        # Find all CSV files in the downloaded path
        csv_files = list(Path(path).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in Kaggle download.")

        logger.info(f"Found {len(csv_files)} CSV file(s)")

        dfs = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")
                dfs.append(df)
                logger.info(f"  Loaded {len(df):,} rows from {csv_path.name}")
            except Exception as e:
                logger.warning(f"  Skipped {csv_path.name}: {e}")

        if not dfs:
            raise ValueError("No readable CSVs found.")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows loaded from Kaggle: {len(combined):,}")
        return combined

    except Exception as e:
        logger.warning(f"Kaggle load failed ({e}). Generating synthetic dataset.")
        return _generate_synthetic_jobs(n=5000)


def _map_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Kaggle dataset columns to the JobPilot schema.
    Handles different column naming conventions in the TechMap dataset.
    """
    col_map = {
        # Possible Kaggle column names → our schema
        "job_title":       "title",
        "jobtitle":        "title",
        "position":        "title",
        "employer":        "company",
        "company_name":    "company",
        "organization":    "company",
        "job_location":    "location",
        "city_state":      "location",
        "job_description": "description",
        "body":            "description",
        "content":         "description",
        "apply_url":       "url",
        "link":            "url",
        "job_url":         "url",
        "salary":          "salary_raw",
        "salary_range":    "salary_raw",
        "date":            "date_posted",
        "posted_date":     "date_posted",
        "published":       "date_posted",
        "job_type":        "employment_type",
        "contract_type":   "employment_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


# ─── JSearch / RapidAPI live ingestion ────────────────────────────────────────
def fetch_jsearch_jobs(
    query: str = "data scientist",
    num_pages: int = 3,
    country: str = "us",
    date_posted: str = "week",
) -> pd.DataFrame:
    """
    Fetch live job postings from JSearch API (RapidAPI).
    Returns a DataFrame in the JobPilot schema.
    """
    if not JSEARCH_API_KEY:
        logger.warning("JSEARCH_API_KEY not set. Skipping live ingestion.")
        return pd.DataFrame()

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    records = []
    for page in range(1, num_pages + 1):
        params = {
            "query": query,
            "page": str(page),
            "num_pages": "1",
            "country": country,
            "date_posted": date_posted,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            jobs = data.get("data", [])
            logger.info(f"JSearch page {page}: {len(jobs)} jobs fetched")

            for j in jobs:
                records.append({
                    "title":           j.get("job_title", ""),
                    "company":         j.get("employer_name", ""),
                    "location":        f"{j.get('job_city', '')}, {j.get('job_country', '')}",
                    "city":            j.get("job_city", ""),
                    "country":         j.get("job_country", "US"),
                    "remote":          j.get("job_is_remote", False),
                    "description":     j.get("job_description", ""),
                    "employment_type": j.get("job_employment_type", "FULLTIME"),
                    "date_posted":     j.get("job_posted_at_datetime_utc", ""),
                    "url":             j.get("job_apply_link", ""),
                    "salary_min":      j.get("job_min_salary") or 0,
                    "salary_max":      j.get("job_max_salary") or 0,
                    "source":          "jsearch",
                })
            time.sleep(0.5)  # rate limiting
        except Exception as e:
            logger.error(f"JSearch page {page} failed: {e}")
            break

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"JSearch: fetched {len(df):,} total live jobs")
    return df


def fetch_multiple_queries(queries: list[str], pages_per_query: int = 2) -> pd.DataFrame:
    """Fetch jobs for multiple search queries and combine."""
    frames = []
    for q in queries:
        logger.info(f"Fetching: '{q}'")
        df = fetch_jsearch_jobs(query=q, num_pages=pages_per_query)
        if not df.empty:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ─── Synthetic data (fallback) ────────────────────────────────────────────────
def _generate_synthetic_jobs(n: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic job dataset for testing when Kaggle data
    is unavailable. Covers the four persona requirements.
    """
    logger.info(f"Generating {n:,} synthetic job records...")
    random.seed(42)
    np.random.seed(42)

    titles = [
        "Data Scientist", "Senior Data Scientist", "ML Engineer",
        "Senior ML Engineer", "Applied Scientist", "Data Analyst",
        "Senior Data Analyst", "BI Analyst", "Junior Data Analyst",
        "Analytics Engineer", "Data Engineer", "Senior Data Engineer",
        "MLOps Engineer", "ML Platform Engineer", "Research Scientist",
        "AI Engineer", "NLP Engineer", "Computer Vision Engineer",
        "Product Analyst", "Business Intelligence Developer",
        "Staff ML Engineer", "Principal Data Scientist",
    ]
    companies = [
        "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Uber",
        "Airbnb", "Stripe", "Salesforce", "IBM", "Oracle", "Intel", "NVIDIA",
        "Linkedin", "Twitter", "Shopify", "Databricks", "Snowflake", "Palantir",
        "Epic Systems", "Kaiser Permanente", "CVS Health", "UnitedHealth",
        "JPMorgan Chase", "Goldman Sachs", "BlackRock", "Citadel",
        "General Dynamics", "Lockheed Martin", "Raytheon",  # defense (for filter testing)
        "TechStartup Inc", "SmallBiz Analytics", "DataCo",
    ]
    locations = [
        "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
        "Boston, MA", "Chicago, IL", "Remote", "Remote - US",
        "Mountain View, CA", "Menlo Park, CA", "Redmond, WA",
        "London, UK", "Toronto, Canada", "Berlin, Germany",
    ]
    skills_pool = [
        "Python", "SQL", "machine learning", "deep learning", "PyTorch",
        "TensorFlow", "Spark", "Kafka", "Kubernetes", "Docker", "AWS",
        "scikit-learn", "NLP", "computer vision", "pandas", "R",
        "Tableau", "Power BI", "dbt", "Snowflake", "Databricks",
        "MLflow", "Airflow", "Java", "Scala", "Go",
    ]

    records = []
    for i in range(n):
        title = random.choice(titles)
        company = random.choice(companies)
        location = random.choice(locations)
        is_remote = "Remote" in location or random.random() < 0.35
        num_skills = random.randint(3, 10)
        req_skills = random.sample(skills_pool, num_skills)

        seniority = "senior" if any(s in title for s in ["Senior", "Staff", "Principal", "Lead"]) \
                    else "junior" if any(s in title for s in ["Junior", "Associate"]) \
                    else "mid"

        sal_base = {"junior": 80000, "mid": 120000, "senior": 180000}[seniority]
        sal_min = sal_base + random.randint(-15000, 0)
        sal_max = sal_base + random.randint(10000, 50000)

        is_defense = any(d in company for d in ["Dynamics", "Lockheed", "Raytheon"])
        is_contract = random.random() < 0.1
        exp_req = {"junior": random.randint(0, 2), "mid": random.randint(2, 5),
                   "senior": random.randint(5, 10)}[seniority]

        description = (
            f"{company} is looking for a {title} to join our team. "
            f"Required skills: {', '.join(req_skills)}. "
            f"{exp_req}+ years of experience required. "
            f"{'This is a remote position.' if is_remote else f'Based in {location}.'} "
            f"{'Contract only position.' if is_contract else 'Full-time permanent role.'} "
            f"Salary: ${sal_min:,} - ${sal_max:,}. "
            f"{'H-1B visa sponsorship available for qualified candidates.' if not is_defense and random.random() < 0.5 else ''}"
        )

        days_ago = random.randint(0, 90)
        date_posted = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        records.append({
            "title":            title,
            "company":          company,
            "location":         location,
            "description":      description,
            "skills_raw":       ", ".join(req_skills),
            "salary_raw":       f"${sal_min:,} - ${sal_max:,}",
            "employment_type":  "Contract" if is_contract else "Full-time",
            "date_posted":      date_posted,
            "url":              f"https://example.com/jobs/{i+1}",
            "source":           "synthetic",
        })

    df = pd.DataFrame(records)
    logger.info(f"Synthetic dataset: {len(df):,} records")
    return df


# ─── Save helpers ─────────────────────────────────────────────────────────────
def save_raw_data(df: pd.DataFrame, path=None) -> Path:
    """Save raw ingested data as CSV."""
    path = Path(path) if path else RAW_CSV
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} raw records to {path}")
    return path


def load_offline_data(sample: bool = False) -> pd.DataFrame:
    """
    Load offline data. Tries clean parquet first, then raw CSV,
    then falls back to synthetic data.
    """
    from src.utils import CLEAN_PARQUET

    target = SAMPLE_PARQUET if sample else CLEAN_PARQUET
    if target.exists():
        df = pd.read_parquet(target)
        logger.info(f"Loaded {len(df):,} jobs from {target.name}")
        return df

    if RAW_CSV.exists():
        logger.info(f"Loading raw CSV: {RAW_CSV}")
        return pd.read_csv(RAW_CSV, low_memory=False)

    logger.warning("No offline data found. Generating synthetic dataset.")
    return _generate_synthetic_jobs(5000)

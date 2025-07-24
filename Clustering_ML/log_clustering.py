#!/usr/bin/env python3
"""
log_clustering.py

Automated clustering and binning of UVM regression-test logs.
"""
import os
import re
import string
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import hstack
import hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
import matplotlib.pyplot as plt

# =======================
# Constants & File Paths
# =======================
BASE_DIR = os.path.abspath('.')
LOG_FOLDER = os.path.join(BASE_DIR, 'failing_logs_52144')
PARSED_FOLDER = os.path.join(LOG_FOLDER, 'parsed')
# Ensure output directory exists
os.makedirs(PARSED_FOLDER, exist_ok=True)

# Output file paths
STRUCTURED_LOG_FILE     = os.path.join(PARSED_FOLDER, 'uvm_log_structured.csv')
STRUCTURED_FEATURES_FILE= os.path.join(PARSED_FOLDER, 'structured_features.npy')
TFIDF_FEATURES_FILE     = os.path.join(PARSED_FOLDER, 'tfidf_features.npy')
HYBRID_FEATURES_FILE    = os.path.join(PARSED_FOLDER, 'hybrid_features.npy')

# =======================
# Regular Expressions
# =======================
UVM_LOG_PATTERN = re.compile(
    r"(?P<severity>UVM_INFO|UVM_WARNING|UVM_ERROR|UVM_FATAL)\s+"
    r"@\s+(?P<time>[0-9\.]+)us:\s+"
    r"\[(?P<module>[^\]]+)\]\s+(?P<message>.+?)(?:\s+\S+\(\d+\))?$"
)
CONFIG_PATTERN = re.compile(
    r"\+(?P<key>UVM_TESTNAME|uvm_set_type_override|UVM_VERBOSITY)=(?P<value>[\w\.\-]+)"
)
SIM_WARNING_PATTERN = re.compile(
    r"(Null object access|constraint solver failure|Segmentation fault|coverage illegal hit)",
    re.IGNORECASE,
)

# =======================
# Preprocessing Helpers
# =======================
def split_alphanum(text: str) -> str:
    """
    Insert spaces between letters and digits for tokenization.
    """
    text = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)
    return text


def clean_text(text: str) -> str:
    """
    Lowercase, split alphanumeric sequences, and remove punctuation.
    """
    text = split_alphanum(text.lower())
    return text.translate(str.maketrans('', '', string.punctuation))

# =======================
# Data Loading & Parsing
# =======================
def load_logs(log_directory: str) -> list[tuple[str,str]]:
    """
    Load all .log files from `log_directory`.
    Returns a list of (filename, content) tuples.
    """
    logs = []
    for fname in os.listdir(log_directory):
        if fname.endswith('.log'):
            path = os.path.join(log_directory, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                logs.append((fname, f.read()))
    return logs


def extract_log_info(logs: list[tuple[str,str]]) -> pd.DataFrame:
    """
    Parse raw log lines to extract structured entries:
    - UVM messages
    - Config overrides
    - Simulator warnings
    """
    records = []
    for fname, content in logs:
        for line in content.splitlines():
            line = line.strip()
            match = UVM_LOG_PATTERN.match(line)
            if match:
                records.append({
                    'file': fname,
                    'type': 'UVM',
                    'severity': match.group('severity'),
                    'time_us': float(match.group('time')),
                    'module': match.group('module'),
                    'message': match.group('message'),
                })
                continue
            match = CONFIG_PATTERN.search(line)
            if match:
                records.append({
                    'file': fname,
                    'type': 'Config',
                    'severity': '',
                    'time_us': 0.0,
                    'module': 'CONFIG',
                    'message': f"{match.group('key')}={match.group('value')}",
                })
                continue
            match = SIM_WARNING_PATTERN.search(line)
            if match:
                records.append({
                    'file': fname,
                    'type': 'Simulator',
                    'severity': 'SIM_ERROR',
                    'time_us': 0.0,
                    'module': 'SIM',
                    'message': match.group(1),
                })
    return pd.DataFrame(records)

# =======================
# Feature Extraction
# =======================
def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute numeric features per log file:
    - Severity counts
    - Unique modules
    - Error timing metrics
    """
    feature_rows = []
    for fname, group in df.groupby('file'):
        row = defaultdict(float)
        row['file'] = fname
        # Severity counts
        counts = group['severity'].value_counts().to_dict()
        for sev in ['UVM_INFO','UVM_WARNING','UVM_ERROR','UVM_FATAL','SIM_ERROR']:
            row[f'count_{sev}'] = counts.get(sev, 0)
        # Other metrics
        row['unique_modules'] = group['module'].nunique()
        row['total_messages'] = len(group)
        errors = group[group['severity'] == 'UVM_ERROR']
        row['first_error_time'] = errors['time_us'].min() if not errors.empty else 0.0
        row['last_error_time'] = errors['time_us'].max() if not errors.empty else 0.0
        feature_rows.append(row)
    df_struct = pd.DataFrame(feature_rows).set_index('file').fillna(0)
    np.save(STRUCTURED_FEATURES_FILE, df_struct.values)
    return df_struct


def create_documents(df: pd.DataFrame, context_lines: int = 0) -> pd.Series:
    """
    Build a text document per log file containing the first UVM_ERROR
    and optional surrounding lines. Returns a Series: filename -> text.
    """
    docs = {}
    for fname, group in df[df['type']=='UVM'].groupby('file'):
        group = group.reset_index(drop=True)
        idx = group[group['severity']=='UVM_ERROR'].index.min()
        if np.isnan(idx):
            docs[fname] = clean_text(group['message'].iloc[0] if not group.empty else '')
        else:
            end = int(idx + context_lines + 1)
            msgs = group.loc[idx:end, 'message']
            docs[fname] = ' '.join(clean_text(m) for m in msgs)
    return pd.Series(docs)


def compute_tfidf(docs: pd.Series) -> tuple[np.ndarray, list[str]]:
    """
    Compute TF-IDF features for log documents.
    Returns feature matrix and term list.
    """
    stop_words = [
        'packet','header','channel','count','period','difference',
        'recorded','injected','failed','unable','comerr',
        'data','value','word','status','x'
    ]
    vect = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
        min_df=int(0.09*len(docs)),
        max_df=0.6,
        max_features=1500,
        sublinear_tf=True,
        norm='l2'
    )
    X = vect.fit_transform(docs.values)
    np.save(TFIDF_FEATURES_FILE, X.toarray())
    return X, vect.get_feature_names_out().tolist()


def build_hybrid_features(tfidf_mat, struct_df: pd.DataFrame):
    """
    Combine scaled TF-IDF and numeric features into a hybrid matrix.
    """
    tfidf_scaled = StandardScaler(with_mean=False).fit_transform(tfidf_mat)
    struct_scaled = StandardScaler().fit_transform(struct_df.values)
    # Weight ratios
    alpha = np.sqrt(0.85 / tfidf_scaled.shape[1])
    beta  = np.sqrt(0.15 / struct_scaled.shape[1])
    hybrid = hstack([alpha * tfidf_scaled, beta * struct_scaled])
    np.save(HYBRID_FEATURES_FILE, hybrid.toarray())
    return hybrid


def reduce_dimensionality(hybrid, n_components: int = 2) -> np.ndarray:
    """
    Perform truncated SVD for dimensionality reduction.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(hybrid)

# =======================
# Clustering & Evaluation
# =======================
def cluster_hdbscan(reduced: np.ndarray) -> np.ndarray:
    """
    Cluster data using HDBSCAN and return labels.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        alpha=0.8
    )
    return clusterer.fit_predict(reduced)


def cluster_kmeans(reduced: np.ndarray, k: int) -> np.ndarray:
    """
    Cluster data using KMeans with k clusters.
    """
    return KMeans(n_clusters=k, random_state=42).fit_predict(reduced)


def evaluate_clustering(labels: np.ndarray, truth: np.ndarray, reduced: np.ndarray):
    """
    Print ARI, AMI, and silhouette (if applicable).
    """
    ari = adjusted_rand_score(truth, labels)
    ami = adjusted_mutual_info_score(truth, labels)
    print(f"ARI: {ari:.3f}, AMI: {ami:.3f}")
    if len(set(labels[labels != -1])) >= 2:
        sil = silhouette_score(reduced[labels != -1], labels[labels != -1])
        print(f"Silhouette (noise removed): {sil:.3f}")

# =======================
# Main Execution
# =======================
def main():
    # Load and parse logs
    logs       = load_logs(LOG_FOLDER)
    df_logs    = extract_log_info(logs)
    df_logs.to_csv(STRUCTURED_LOG_FILE, index=False)

    # Extract numeric features
    df_struct  = extract_structured_features(df_logs)

    # Build text documents
    docs       = create_documents(df_logs)

    # TF-IDF features
    tfidf_mat, terms = compute_tfidf(docs)

    # Hybrid features
    hybrid     = build_hybrid_features(tfidf_mat, df_struct)

    # Dimensionality reduction
    reduced    = reduce_dimensionality(hybrid)

    # Example ground-truth mapping (customize as needed)
    # desired    = {'file1.log': 0, 'file2.log': 1, ...}
    # truth      = np.array([desired.get(f, -1) for f in docs.index])

    # Perform HDBSCAN
    hdb_labels = cluster_hdbscan(reduced)
    # evaluate_clustering(hdb_labels, truth, reduced)

    # KMeans sweep example (uncomment to use)
    # best_k, best_ari = 2, -1
    # for k in range(2, 11):
    #     km_labels = cluster_kmeans(reduced, k)
    #     ari = adjusted_rand_score(truth, km_labels)
    #     if ari > best_ari:
    #         best_k, best_ari = k, ari
    # print(f"Best KMeans k={best_k}, ARI={best_ari:.3f}")

    # Visualization placeholder
    # (Add plotting code here if desired)

if __name__ == '__main__':
    main()

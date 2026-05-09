# IFEBY310 Big Data Project  
## Literary Big Data Analysis: Romantic vs Realist Authors

This repository contains a Big Data project for the IFEBY310 course.

The project builds an end-to-end Spark and Spark NLP pipeline for analyzing a corpus of nineteenth-century novels. The goal is to compare Romantic and Realist authors using readability and stylometric indicators while also demonstrating practical Spark experience: file formats, staged processing, NLP annotation, and reproducible reporting.

---

## Project Topic

**Literary Big Data Analysis — Romantic vs Realist Authors**

The project analyzes 18 novels by 8 authors:

- Honoré de Balzac
- Alexandre Dumas
- Victor Hugo
- George Sand
- Stendhal
- Charles Dickens
- Gustave Flaubert
- Émile Zola

The corpus contains both French and English novels from the nineteenth century.

---

## Main Objectives

The project aims to:

1. Ingest raw literary text files.
2. Clean Project Gutenberg headers, footers, and metadata.
3. Store the corpus in structured big-data formats.
4. Annotate the corpus with Spark NLP.
5. Compute readability indicators.
6. Compute stylometric indicators.
7. Compare Parquet and ORC storage formats.
8. Produce a reproducible Quarto HTML report.

---

## Pipeline Overview

The pipeline follows these stages:

```text
Raw .txt files
   ↓
Ingestion
   ↓
Cleaned corpus
   ↓
Parquet / ORC storage
   ↓
Chunking for Spark NLP
   ↓
Tokenization, lemmatization, POS tagging
   ↓
Readability analysis
   ↓
Stylometry analysis
   ↓
Quarto report

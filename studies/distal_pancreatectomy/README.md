# Case study: distal pancreatectomy registry (synthetic data)

This folder holds the analysis of a national registry of minimally invasive
distal pancreatectomy that compared higher- and lower-volume centers. Here it
runs on simulated patients. It is meant as a template for a complete,
prespecified analysis; the walkthrough is in
[docs/tutorials/case_study.md](../../docs/tutorials/case_study.md).

The analysis is described in `config/primary_analysis.yaml`. `preprocess.py`
builds the synthetic data and the binary composites (ideal and textbook outcome)
and checks the schema. `export_public_data.py` writes the data set, and
`run_public_analysis.py` runs the analysis and writes the tables and figure to
`results/public/`.

```bash
python -m studies.distal_pancreatectomy.export_public_data
python -m studies.distal_pancreatectomy.run_public_analysis --n-boot 400
```

The hierarchy is 90-day death, Clavien-Dindo grade (none, III, IV), pancreatic
fistula grade (none, B, C), readmission and length of stay, where a difference of
0 or 1 day counts as a tie. Two patients who both died are tied. The sensitivity
analyses change the length-of-stay rule (exact days, at least 3 days apart,
categories <=7, 8-13 and >13 days, or no length of stay) and the order of the
outcomes.

Exposure is the center's mean annual number of minimally invasive pancreatic
resections, more than 10 versus 10 or fewer.

All data in this repository are synthetic, so the estimates only illustrate how
the code works.

# Reference results from the R packages BuyseTest and WINS.
#
# Run from the repository root after `python validation/generate_data.py`:
#   Rscript validation/reference_r.R
# Writes validation/reference/r_reference.json.
#
# Convention mapping (documented in docs/validation.md):
#   BuyseTest counts a pair as favorable when the difference is >= threshold.
#   pywinratio's `margin` ties differences <= margin; its `threshold` argument
#   uses the BuyseTest convention. On integer data, margin m == threshold m + 1.

suppressPackageStartupMessages({
  library(BuyseTest)
  library(WINS)
  library(jsonlite)
})
BuyseTest.options(trace = 0)

data_dir <- file.path("validation", "data")
out_file <- file.path("validation", "reference", "r_reference.json")

buyse_row <- function(fit, scenario, package = "BuyseTest") {
  nb <- confint(fit, statistic = "netBenefit")
  wr <- confint(fit, statistic = "winRatio")
  last <- nrow(nb)
  list(
    scenario = scenario,
    package = paste(package, as.character(packageVersion(package))),
    net_benefit = unname(nb[last, "estimate"]),
    se_net_benefit = unname(nb[last, "se"]),
    wr = unname(wr[last, "estimate"]),
    se_log_wr = unname(wr[last, "se"] / wr[last, "estimate"]),
    p_value_net_benefit = unname(nb[last, "p.value"])
  )
}

results <- list()

## S1: randomized trial, two Gehan time-to-event tiers and a continuous tier
trial <- read.csv(file.path(data_dir, "heart_failure_trial.csv"))
trial$arm <- factor(trial$arm, levels = c("placebo", "treatment"))
fit1 <- BuyseTest(
  arm ~ tte(time_death, status = death) + tte(time_hosp, status = hosp) +
    cont(kccq_change, threshold = 5),
  data = trial, scoring.rule = "Gehan", method.inference = "u-statistic"
)
results[["S1_trial_gehan"]] <- buyse_row(fit1, "S1_trial_gehan")

## S2: observational cohort, binary / ordinal / continuous tiers (lower is better)
cohort <- read.csv(file.path(data_dir, "surgical_cohort.csv"))
cohort$volume <- factor(cohort$volume, levels = c("low", "high"))
f2 <- volume ~ bin(death90, operator = "<0") + cont(clavien, operator = "<0") +
  cont(popf, operator = "<0") + bin(readmission, operator = "<0") +
  cont(los, threshold = 2, operator = "<0")
fit2 <- BuyseTest(f2, data = cohort, method.inference = "u-statistic")
results[["S2_cohort_all_pairs"]] <- buyse_row(fit2, "S2_cohort_all_pairs")

## S3: stratified by period, strata pooled by number of pairs ("Buyse")
f3 <- update(f2, . ~ . + strata(period))
fit3 <- BuyseTest(f3, data = cohort, method.inference = "u-statistic", pool.strata = "Buyse")
results[["S3_cohort_stratified"]] <- buyse_row(fit3, "S3_cohort_stratified")

## S4: matched pairs as one stratum per pair (point estimates)
matched <- read.csv(file.path(data_dir, "surgical_matched.csv"))
matched$volume <- factor(matched$volume, levels = c("low", "high"))
f4 <- update(f2, . ~ . + strata(PAIR_ID))
fit4 <- BuyseTest(f4, data = matched, method.inference = "none", pool.strata = "Buyse")
s4 <- confint(fit4, statistic = "netBenefit")
s4wr <- confint(fit4, statistic = "winRatio")
results[["S4_matched_pairs"]] <- list(
  scenario = "S4_matched_pairs",
  package = paste("BuyseTest", as.character(packageVersion("BuyseTest"))),
  net_benefit = unname(s4[nrow(s4), "estimate"]),
  wr = unname(s4wr[nrow(s4wr), "estimate"])
)

## S5: WINS win statistics for the trial (two Gehan tiers, then KCCQ, no thresholds)
wins_data <- data.frame(
  id = seq_len(nrow(trial)),
  arm = as.character(trial$arm),
  Delta_1 = trial$death, Delta_2 = trial$hosp,
  Y_1 = trial$time_death, Y_2 = trial$time_hosp, Y_3 = trial$kccq_change
)
w <- suppressWarnings(win.stat(
  data = wins_data, ep_type = c("tte", "tte", "continuous"),
  arm.name = c("treatment", "placebo"), priority = c(1, 2, 3), tau = 0,
  np_direction = "larger", method = "unadjusted", pvalue = "two-sided",
  summary.print = FALSE
))
pick <- function(x) as.numeric(x)
results[["S5_trial_wins"]] <- list(
  scenario = "S5_trial_wins",
  package = paste("WINS", as.character(packageVersion("WINS"))),
  wr = pick(w$Win_statistic$Win_Ratio[1]),
  wr_ci = pick(w$Win_statistic$Win_Ratio[2:3]),
  net_benefit = pick(w$Win_statistic$Net_Benefit[1]),
  net_benefit_ci = pick(w$Win_statistic$Net_Benefit[2:3]),
  win_odds = pick(w$Win_statistic$Win_Odds[1]),
  win_odds_ci = pick(w$Win_statistic$Win_Odds[2:3])
)

## S6: censoring time equal to the other patient's event time (Gehan convention)
ties <- data.frame(
  arm = factor(c("C", "C", "C", "T", "T", "T"), levels = c("C", "T")),
  time = c(5, 2, 7, 5, 3, 7),
  status = c(1, 0, 0, 0, 1, 1)
)
fit6 <- BuyseTest(arm ~ tte(time, status = status), data = ties, scoring.rule = "Gehan",
                  method.inference = "none")
results[["S6_event_censoring_ties"]] <- list(
  scenario = "S6_event_censoring_ties",
  package = paste("BuyseTest", as.character(packageVersion("BuyseTest"))),
  wins = unname(fit6@count.favorable[1, 1]),
  losses = unname(fit6@count.unfavorable[1, 1])
)

dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
write_json(list(
  generated = format(Sys.Date()),
  r_version = R.version.string,
  results = unname(results)
), out_file, auto_unbox = TRUE, digits = 12, pretty = TRUE)
cat("wrote", out_file, "\n")

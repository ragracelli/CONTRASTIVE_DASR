#!/bin/bash
# encoding: utf-8

# Very Low – ❄️ Freeze the entire last decoder
bases_very_low=("F03" "M12" "M01")
for speaker in "${bases_very_low[@]}"; do
    python asr_trafo_en_vars_wsl.py "$speaker" original True False True 100 encoder_sgd_epoch100.h5
done

# Low – ❄️ Freeze the entire last decoder
bases_low=("M07" "F02" "M16")
for speaker in "${bases_low[@]}"; do
    python asr_trafo_en_vars_wsl.py "$speaker" original True False True 100 encoder_sgd_epoch100.h5
done

# Mild – ❄️ Freeze only decoder's FFN
bases_mild=("M05" "M11" "F04")
for speaker in "${bases_mild[@]}"; do
    python asr_trafo_en_vars_wsl.py "$speaker" original False True True 100 encoder_sgd_epoch100.h5
done

# High – ❄️ Freeze only decoder's FFN
bases_high=("M09" "M14" "M10" "M08" "F05")
for speaker in "${bases_high[@]}"; do
    python asr_trafo_en_vars_wsl.py "$speaker" original False True True 100 encoder_sgd_epoch100.h5
done

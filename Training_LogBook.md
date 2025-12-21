### 📊 Training Run Summary 2025-12-17 (OVERFITTING)
Run Name: training_run_focal_alpha
Date: December 20, 2025
Status: ⚠️ OVERFITTING DETECTED - Training loss decreasing but validation mIoU dropping

Configuration
Parameter	Value
Model	DFormerv2_Base (HAM decoder)
Dataset	JARVIS/Dformer_format (6424 train / 1606 test)
Loss	Focal Loss (γ=2.0, per-class α)
Learning Rate	3e-5
Batch Size	2
Warm-up	10 epochs
Drop Path Rate	0.1
Image Size	512×768
Eval Frequency	Every 10 epochs
Focal Alpha Weights (AGGRESSIVE - 1000x range)
Class	Name	Pixel %	Alpha
0	battery	71.24%	0.02
1	battery_plate	3.51%	0.40
2	battery_strap	7.86%	0.18
3	bushing	0.21%	6.00
4	cable	0.14%	9.00
5	cable_holder	5.62%	0.25
6	cable_tie	1.05%	1.20
7	Collection	0.30%	4.20
8	UNUSED	0%	0.00
9	connector	2.00%	0.65
10	cooler_pipe_sensor	0.08%	15.00
11	hose_clamp	0.29%	4.30
12	invertor	0.86%	1.45
13	invertor_cover	2.53%	0.50
14	nut	0.09%	14.00
15	pump_bracket	0.40%	3.10
16	pump_valve	0.34%	3.70
17	screw	0.06%	20.00
18	tube	3.42%	0.37
Training Progression
Epoch	Train Loss	Val mIoU	Status
1	0.0381	3.62%	Initial
10	0.0340	46.43%	✅ Best
20	0.0185	21.24%	❌ Overfitting
Per-Class IoU Comparison
Class	Epoch 1	Epoch 10 (Best)	Epoch 20
battery	29.52%	85.80%	75.39%
battery_plate	9.30%	55.57%	16.54%
battery_strap	4.18%	53.70%	18.84%
bushing	1.06%	40.20%	15.36%
cable	0.22%	39.37%	19.38%
cable_holder	1.47%	53.48%	19.56%
cable_tie	2.94%	41.08%	13.42%
Collection	0.45%	31.71%	17.62%
connector	7.53%	56.56%	21.92%
cooler_pipe_sensor	0.06%	42.55%	18.05%
hose_clamp	0.72%	38.40%	18.41%
invertor	1.18%	49.39%	18.95%
invertor_cover	2.61%	59.62%	24.57%
nut	1.85%	39.76%	20.08%
pump_bracket	1.48%	52.36%	23.47%
pump_valve	1.05%	53.16%	19.61%
screw	0.69%	31.65%	19.23%
tube	2.44%	57.88%	23.11%
Diagnosis
Problem: 1000x alpha weight range caused overfitting to rare class training examples
Evidence: Train loss ↓ (0.038→0.018) but val mIoU ↓ (46%→21%)
Best Checkpoint: epoch-10_miou_46.43.pth


### 📊 Training Run 2025-12-20 (CONSERVATIVE WEIGHTS) # 
Planned Changes:

Reduced focal_alpha range: 0.10 - 5.0 (50x range vs. 1000x before)
Increased regularization: drop_path_rate 0.1 → 0.2
Earlier checkpoints: Start saving at epoch 10
Configuration
Parameter	Current Run	Upcoming Run
focal_alpha range	0.02 - 20.0	0.10 - 5.0
drop_path_rate	0.1	0.2
checkpoint_start_epoch	20	10
New Focal Alpha Weights (CONSERVATIVE - 50x range)
Class	Name	Pixel %	Old Alpha	New Alpha
0	battery	71.24%	0.02	0.10
1	battery_plate	3.51%	0.40	0.50
2	battery_strap	7.86%	0.18	0.35
3	bushing	0.21%	6.00	2.20
4	cable	0.14%	9.00	2.70
5	cable_holder	5.62%	0.25	0.40
6	cable_tie	1.05%	1.20	1.00
7	Collection	0.30%	4.20	1.80
8	UNUSED	0%	0.00	0.00
9	connector	2.00%	0.65	0.70
10	cooler_pipe_sensor	0.08%	15.00	3.50
11	hose_clamp	0.29%	4.30	1.85
12	invertor	0.86%	1.45	1.10
13	invertor_cover	2.53%	0.50	0.60
14	nut	0.09%	14.00	3.30
15	pump_bracket	0.40%	3.10	1.60
16	pump_valve	0.34%	3.70	1.70
17	screw	0.06%	20.00	5.00
18	tube	3.42%	0.37	0.55
Expected Behavior
Less overfitting due to gentler class rebalancing
Better generalization with increased dropout (drop_path 0.2)
May need more epochs to converge but should maintain validation performance


2025-12-21 08:14:01,160 - train.py[line:427] - INFO: Epoch 10/150 Iter 3213/3213: lr=2.9999e-05 loss=0.0336 total_loss=0.0288
Validation Iter: 1 / 803
Validation Iter: 401 / 803
Validation Iter: 802 / 803
2025-12-21 09:50:50,415 - train.py[line:568] - INFO: 
================================================================================
2025-12-21 09:50:50,415 - train.py[line:569] - INFO: Epoch 10 - Per-Class IoU Breakdown:
2025-12-21 09:50:50,416 - train.py[line:570] - INFO: ================================================================================
2025-12-21 09:50:50,416 - train.py[line:572] - INFO:   Class  0 - battery                  :  93.43%
2025-12-21 09:50:50,417 - train.py[line:572] - INFO:   Class  1 - battery_plate            :  79.12%
2025-12-21 09:50:50,417 - train.py[line:572] - INFO:   Class  2 - battery_strap            :  77.34%
2025-12-21 09:50:50,418 - train.py[line:572] - INFO:   Class  3 - bushing                  :  54.34%
2025-12-21 09:50:50,418 - train.py[line:572] - INFO:   Class  4 - cable                    :  58.04%
2025-12-21 09:50:50,419 - train.py[line:572] - INFO:   Class  5 - cable_holder             :  74.87%
2025-12-21 09:50:50,419 - train.py[line:572] - INFO:   Class  6 - cable_tie                :  59.92%
2025-12-21 09:50:50,420 - train.py[line:572] - INFO:   Class  7 - Collection               :  47.40%
2025-12-21 09:50:50,420 - train.py[line:572] - INFO:   Class  8 - UNUSED_COLLECTION        :   0.00%
2025-12-21 09:50:50,421 - train.py[line:572] - INFO:   Class  9 - connector                :  75.80%
2025-12-21 09:50:50,421 - train.py[line:572] - INFO:   Class 10 - cooler_pipe_sensor       :  62.19%
2025-12-21 09:50:50,421 - train.py[line:572] - INFO:   Class 11 - hose_clamp               :  55.36%
2025-12-21 09:50:50,422 - train.py[line:572] - INFO:   Class 12 - invertor                 :  68.31%
2025-12-21 09:50:50,422 - train.py[line:572] - INFO:   Class 13 - invertor_cover           :  78.63%
2025-12-21 09:50:50,423 - train.py[line:572] - INFO:   Class 14 - nut                      :  55.12%
2025-12-21 09:50:50,423 - train.py[line:572] - INFO:   Class 15 - pump_bracket             :  69.56%
2025-12-21 09:50:50,424 - train.py[line:572] - INFO:   Class 16 - pump_valve               :  70.61%
2025-12-21 09:50:50,424 - train.py[line:572] - INFO:   Class 17 - screw                    :  50.21%
2025-12-21 09:50:50,424 - train.py[line:572] - INFO:   Class 18 - tube                     :  75.71%
2025-12-21 09:50:50,425 - train.py[line:573] - INFO: ================================================================================
2025-12-21 09:50:50,425 - train.py[line:574] - INFO:   Mean IoU: 63.47%
2025-12-21 09:50:50,426 - train.py[line:575] - INFO:   Mean Acc: 75.41%
2025-12-21 09:50:50,426 - train.py[line:576] - INFO:   Mean F1:  75.43%
2025-12-21 09:50:50,427 - train.py[line:577] - INFO: ================================================================================

## Data processing procedure
1. Collect daily close prices of `$DIS` stock starting from 2008-10-1 and ending at 2020-10-30
2. Batch 2720 days of close prices into 19 day windows
3. Convert price data into daily percentage change and normalize to (-1, 1)
4. Split each 19 day window into two arrays of shape 15 (fifteen days of input) and 4 (four days to predict).
5. Shuffle window batch order
6. Train/test split using a ratio of 0.8  
7. Export as `dataset.npy`

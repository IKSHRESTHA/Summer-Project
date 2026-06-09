# Data

The notebook uses the **Human Mortality Database (HMD)** U.S. period life table — both sexes, 1×1 (single year of age × single calendar year).

## Download

1. Register for a free account at [mortality.org](https://mortality.org).
2. Download the file directly at:

   ```
   https://mortality.org/File/GetDocument/hmd.v6/USA/STATS/bltper_1x1.txt
   ```

   Or navigate: **USA → Period Life Tables → 1×1 → bltper_1x1.txt**

3. Place the file in this `data/` directory as `bltper_1x1.txt`.

## Format

Whitespace-delimited text with a two-line header. Columns: `Year Age mx qx ax lx dx Lx Tx ex`.  
The oldest age group is recorded as `110+`; the loader converts this to `110`.

## Coverage

The HMD USA file spans 1933–2024. The notebook restricts analysis to **1950–2019** (ages 0–99) to avoid pre-war volatility and the COVID-19 structural break.

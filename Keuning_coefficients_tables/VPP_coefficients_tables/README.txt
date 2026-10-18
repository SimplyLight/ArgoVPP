VPP COEFFICIENT TABLES - CSV PACKAGE
====================================

Source
------
Pedro Francisco Gouveia Teixeira, Development of a Velocity Prediction Program for Sailing Yachts,
Appendix IV (page 100) and Appendix VI (page 103), October 2024.

Format
------
- UTF-8
- Comma-separated CSV
- Decimal separator: point
- Tables are normalized: the independent variable is the first column, followed by the coefficients.
- Angles are in degrees in the CSV tables.

Core tables for a Delft/Hazen VPP
---------------------------------
1. IV_1_bare_hull_residuary.csv
2. IV_2_keel_residuary.csv
3. IV_3_wetted_surface_vs_heel.csv
4. IV_4_heeled_residuary_delta.csv
5. IV_5_effective_span.csv
6. VI_1_hazen_1980_sail_coefficients.csv

Alternative aerodynamic tables
-------------------------------
- VI_2 and VI_3: ORC 2023 coefficients reproduced in the thesis.
- VI_4 and VI_5: Poor (1986) / IMS alternative coefficients.

Important correction for Table IV-4
-----------------------------------
The source PDF prints the last two row labels both as 'u5'. The governing equation uses u4 and u5.
In IV_4_heeled_residuary_delta.csv, the first printed 'u5' row is therefore named 'u4_inferred',
and the second printed 'u5' row is named 'u5'. The numerical values are transcribed without alteration.

Interpolation
-------------
Use linear interpolation between adjacent Froude numbers, heel angles, or apparent wind angles.
Avoid uncontrolled extrapolation outside the table ranges.

Scope
-----
This package contains the tabulated coefficients reproduced in the 2024 thesis. It does not attempt
to reproduce every table, correction function or neural-network parameter in the complete ORC VPP 2026.

# Data Dictionary

| Column | Type | Description |
|---|---|---|
| norad_id | integer | Unique identifier assigned to the object by NORAD. *Observed in data: 17,073 unique IDs.* |
| name | string | Name of the satellite or object. |
| object_type | string | Type of the object (e.g., Payload, Debris). *Observed in data: Had ~0.02%-0.04% nulls. Dominant value is PAYLOAD (4M+).* |
| satellite_constellation | string | Name of the satellite constellation this object belongs to, if any. *Observed in data: Dominated by Starlink Gen 1 (2.8M).* |
| altitude_km | float | Estimated current altitude of the object in kilometers. *Observed in data: No invalid/negative values found.* |
| altitude_category | string | Categorization of altitude (e.g., LEO, MEO). |
| orbital_band | string | Specific orbital band classification. |
| congestion_risk | float | Estimated risk of congestion or collision. |
| inclination | float | Orbital inclination angle in degrees. *Observed in data: All values within [0, 180] range.* |
| eccentricity | float | Orbital eccentricity (deviation from a perfect circle). *Observed in data: All values within [0, 1) range.* |
| launch_year_estimate | integer | Estimated year the object was launched. |
| days_in_orbit_estimate | float | Estimated number of days the object has been in orbit. *(EXCLUDED FROM MODEL: Severe upstream data corruption)* |
| orbit_lifetime_category | string | Categorization of expected remaining orbital lifetime. *(EXCLUDED FROM MODEL: Derived from corrupted days_in_orbit_estimate)* |
| mean_motion | float | Number of orbits the object completes per day. *Observed in data: All values > 0.* |
| epoch | string | Timestamp of the orbital state vector. |
| data_source | string | Source of the trajectory data. |
| snapshot_date | string | Date when this snapshot of data was taken. *Observed in data: Avg 246 snapshots per norad_id. Range: 2025-11-04 to 2026-08-15.* |
| country | string | Country of origin or ownership. *Observed in data: Had ~0.02%-0.04% nulls. Dominant value is US.* |
| last_seen | string | Timestamp when the object was last observed. |

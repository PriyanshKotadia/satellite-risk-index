# Data Dictionary

| Column | Type | Description |
|---|---|---|
| norad_id | integer | Unique identifier assigned to the object by NORAD. |
| name | string | Name of the satellite or object. |
| object_type | string | Type of the object (e.g., Payload, Debris). |
| satellite_constellation | string | Name of the satellite constellation this object belongs to, if any. |
| altitude_km | float | Estimated current altitude of the object in kilometers. |
| altitude_category | string | Categorization of altitude (e.g., LEO, MEO). |
| orbital_band | string | Specific orbital band classification. |
| congestion_risk | float | Estimated risk of congestion or collision. |
| inclination | float | Orbital inclination angle in degrees. |
| eccentricity | float | Orbital eccentricity (deviation from a perfect circle). |
| launch_year_estimate | integer | Estimated year the object was launched. |
| days_in_orbit_estimate | float | Estimated number of days the object has been in orbit. |
| orbit_lifetime_category | string | Categorization of expected remaining orbital lifetime. |
| mean_motion | float | Number of orbits the object completes per day. |
| epoch | string | Timestamp of the orbital state vector. |
| data_source | string | Source of the trajectory data. |
| snapshot_date | string | Date when this snapshot of data was taken. |
| country | string | Country of origin or ownership. |
| last_seen | string | Timestamp when the object was last observed. |

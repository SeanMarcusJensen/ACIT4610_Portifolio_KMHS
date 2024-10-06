# ACIT4610 Evolutionary Intelligence : Problem 1

---

## How to run

TBD


## Data Description

Time Range: 01/08/2012 - 09/27/2024

Road Segments:
[1] [ID:151704] WOODHAVEN BOULEVARD between 73rd Avenue and 74th Avenue (Queens)
[2] [ID:36517] YORK AVENUE between East 60th Street and East 61st Street (Manhattan)
[3] [ID:15047] CLOVE ROAD between NECKAR AVENUE and WESER AVENUE (Staten Island)
[4] [ID:15741] BAY STREET between CLINTON STREET and BALTIC STREET (Staten Island)
[5] [ID:69698] GERARD AVENUE between East 150th Street and East 151st Street (Bronx)


### Traffic Data
Time Range: 01/08/2012 - 12/13/2020

| Column Name | Description | API Field Name | Data Type |
|-------------|-------------|----------------|-----------|
| ID | Count ID | id | Text |
| SegmentID | LION Segment ID. Called "GIS ID' in the 2011-12 dataset, "Segment ID" in the 2012-13 dataset. | segmentid | Number |
| Roadway Name | Street name | roadway_name | Text |
| From | Intersecting street name at one end of street | from | Text |
| To | Intersecting street name at other end of street | to | Text |
| Direction | Compass direction | direction | Text |
| Date | Date of the traffic count | date | Floating Timestamp |
| 12:00-1:00 AM | Count for the clock hour | _12_00_1_00_am | Number |
| ... | ... | ... | ... |
| 11:00-12:00AM | Count for the clock hour | _11_00_12_00am | Number |

### Speed Data
Time Range: 09/24/2024 01:04:03 PM - 09/27/2024 12:59:09 AM

| Column Name | Description | API Field Name | Data Type |
|-------------|-------------|----------------|-----------|
| ID | | id | Text |
| SPEED | | speed | Text |
| TRAVEL_TIME | | travel_time | Text |
| STATUS | | status | Text |
| DATA_AS_OF | | data_as_of | Floating Timestamp |
| LINK_ID | | link_id | Text |
| LINK_POINTS | | link_points | Text |
| ENCODED_POLY_LINE | | encoded_poly_line | Text |
| ENCODED_POLY_LINE_LVLS | | encoded_poly_line_lvls | Text |
| OWNER | | owner | Text |
| TRANSCOM_ID | | transcom_id | Text |
| BOROUGH | | borough | Text |
| LINK_NAME | | link_name | Text |

## Project Log

### 2024-07-23

[SEAN]
**Problems met:**
The downloaded data was inconsistent with date ranges and column names.
- Traffic Data had dates ranging from 2012 to 2020
- Speed Data had dates ranging from 2024 to 2024
  
There was also no common columns between the two data sets.

**Solutions:**


**What's been done:**
Loaded data into pandas Data Frames for data discovery.
While doing the initial data analysis it was discovered that the data was inconsistant with the date ranges.

`
Traffic Data Time Range: 01/08/2012 - 12/13/2020
Segment Data Time Range: 09/24/2024 01:04:03 PM - 09/27/2024 12:59:09 AM
`

This is an issue because the traffic volume and the speed between segments is dependant on time. Which currently makes them incomparable.
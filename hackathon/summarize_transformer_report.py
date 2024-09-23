from __future__ import annotations
import bentoml
from transformers import pipeline

from bentoml.io import JSON, Text
import pandas as pd
import random
import numpy as np
import json 


## Serving: 
# cd quickstart
# bentoml serve summarize_transformer_report:generate_report  --timeout 1000

EXAMPLE_INPUT = '{"Hydrogen":12886,"Oxigen":61,"Nitrogen":25041,"Methane":877,"CO":83,"CO2":864,"Ethylene":4,"Ethane":305,"Acethylene":0,"DBDS":45.0,"Power factor":1.0,"Interfacial V":45,"Dielectric rigidity":55,"Water content":0,"Health index":85.5,"Life expectation":19.0,"CO_H2_ratio":0.0064410989,"CH4_H2_ratio":0.0680583579,"C2H4_H2_ratio":0.0003104144,"C2H2_H2_ratio":0.0,"H2_N2_ratio":0.5145960625,"O2_N2_ratio":0.002436005,"H2_CO2_ratio":14.9143518519,"TransformerID":"TX-9069","InstallationDate":"2005-10-02","MaintenanceSchedule":"2021-03-20","ReplacementHistory":"Replaced major components","MaintenanceID":"MNT-24761","MaintenanceType":"Emergency Maintenance","MaintenanceDate":"2023-01-29","TransformerLoad":616.567279246,"Temperature":74.5384218405,"Vibration":5.550855378,"Humidity":44.3397911476,"Precipitation":20.2657668926,"Alerts":null,"IncidentType":"Storm","IncidentDetails":"Severe weather affecting transformer","OperationID":"OP-7106","OperatingTime":38.3200938304,"LoadCondition":"Normal Load","PerformanceMetrics":"Performance stable with a load of 616.5672792460182 KW and temperature 74.53842184051585\\u00b0C"}'

@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 1000},
)
class generate_report:
    def __init__(self) -> None:
        # Load model into pipeline
        self.pipeline = pipeline('summarization')

    @bentoml.api
    def summarize(self, json_string: str = EXAMPLE_INPUT) -> str: 
         # Convert input JSON data into a pandas DataFrame (for compatibility)
        input_df = json.loads(json_string)
        # create a pandas DataFrame from the dictionary
        input_data = pd.DataFrame([input_df])

        # Generate a text-based report using only input data
        report = f"""
            Transformer ID: {input_data['TransformerID'].iloc[0]}
            InstallationDate: {input_data['InstallationDate'].iloc[0]}
            OperatingTime: {input_data['OperatingTime'].iloc[0]} years
            LoadCondition: {input_data['LoadCondition'].iloc[0]}
            PerformanceMetrics: {input_data['PerformanceMetrics'].iloc[0]}

            Condition Summary:
            Health Index: {input_data['Health index'].iloc[0]}
            Predicted Remaining Life Expectancy: {input_data['Life expectation'].iloc[0]} years
            Based on the current dissolved gas analysis (DGA) and health index data, this transformer is expected to last an additional {input_data['Life expectation'].iloc[0]} years. 

            Specific gases dissolved in the transformer insulating oil, could help identify potential faults or degradation. Current levels are:
            - Hydrogen: {input_data['Hydrogen'].iloc[0]} ppm.
            - Oxygen: {input_data['Oxigen'].iloc[0]} ppm.
            - Nitrogen: {input_data['Nitrogen'].iloc[0]} ppm.
            - Methane: {input_data['Methane'].iloc[0]} ppm.
            - CO: {input_data['CO'].iloc[0]} ppm.
            - CO2: {input_data['CO2'].iloc[0]} ppm.
            - Ethylene: {input_data['Ethylene'].iloc[0]} ppm.
            - Ethane: {input_data['Ethane'].iloc[0]} ppm.
            - Acethylene: {input_data['Acethylene'].iloc[0]} ppm.
            - DBDS: {input_data['DBDS'].iloc[0]} ppm.

            Other Key indicators suggest that:
            - Power factor: {input_data['Power factor'].iloc[0]}.
            - Interfacial V: {input_data['Interfacial V'].iloc[0]}.
            - Dielectric rigidity: {input_data['Dielectric rigidity'].iloc[0]}.
            - Water content: {input_data['Water content'].iloc[0]} ppm.

            Calculated Gas ratios are:
            - Carbon Monoxide to Hydrogen Ratio: {input_data['CO_H2_ratio'].iloc[0]}.
            - Methane to Hydrogen Ratio: {input_data['CH4_H2_ratio'].iloc[0]}.
            - Ethylene to Hydrogen Ratio: {input_data['C2H4_H2_ratio'].iloc[0]}.
            - Acetylene to Hydrogen Ratio: {input_data['C2H2_H2_ratio'].iloc[0]}.

            Maintenance Information:
            - Maintenance ID: {input_data['MaintenanceID'].iloc[0]}
            - Maintenance Type: {input_data['MaintenanceType'].iloc[0]} 
            - Last Maintenance Date: {input_data.get('MaintenanceDate', 'N/A')[0]}
            - ReplacementHistory: {input_data.get('ReplacementHistory', 'N/A')[0]}

            Environmental and Operational Factors:
            - Transformer Load (KW): {input_data.get('TransformerLoad', 'N/A')[0]}
            - Temperature (Celsius): {input_data.get('Temperature', 'N/A')[0]}
            - Humidity (%): {input_data.get('Humidity', 'N/A')[0]}
            - Vibration Level: {input_data.get('Vibration', 'N/A')[0]} cycles.
            - Precipitation Level: {input_data.get('Precipitation', 'N/A')[0]} mm.

            Alerts & Incidents:
            - Alerts: {input_data.get('Alerts', 'None')[0]}.
            - Incident Type: {input_data.get('IncidentType', 'None')[0]}.
            - Incident Details: {input_data.get('IncidentDetails', 'None')[0]}.

            Create Best Plan for Maintenance and Risk reduction Plan Recommendation:
            """
        
        result = self.pipeline(report, max_length=600, min_length=200, do_sample=True, truncation=True, num_return_sequences=1)
        return result[0]['summary_text']

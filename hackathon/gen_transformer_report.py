import bentoml
from bentoml.io import JSON, Text
import pandas as pd
import random
import numpy as np

# Define the BentoML Service
svc = bentoml.Service("transformer_report_generator")

# Define the input format for the service
@svc.api(input=JSON(), output=Text())
async def generate_report(input_data):
    # Convert input JSON data into a pandas DataFrame (for compatibility)
    input_df = pd.DataFrame([input_data])

    # Generate a text-based report using only input data
    report = f"""
    Transformer ID: {input_data['TransformerID']}
    Health Index: {input_data['Health index']}
    Predicted Remaining Life Expectancy: {input_data['Life expectation']} years

    Condition Summary:
    Based on the current dissolved gas analysis (DGA) and health index data, this transformer is 
    expected to last an additional {input_data['Life expectation']} years. Key indicators suggest that:
    - Carbon Monoxide to Hydrogen Ratio: {input_data['CO_H2_ratio']}
    - Methane to Hydrogen Ratio: {input_data['CH4_H2_ratio']}
    - Ethylene to Hydrogen Ratio: {input_data['C2H4_H2_ratio']}
    - Acetylene to Hydrogen Ratio: {input_data['C2H2_H2_ratio']}

    Maintenance Information:
    - Maintenance Type: {input_data['MaintenanceType']}
    - Maintenance Schedule: {input_data.get('MaintenanceSchedule', 'N/A')}
    - Last Maintenance Date: {input_data.get('MaintenanceDate', 'N/A')}

    Environmental and Operational Factors:
    - Transformer Load (KW): {input_data.get('TransformerLoad', 'N/A')}
    - Temperature (Celsius): {input_data.get('Temperature', 'N/A')}
    - Humidity (%): {input_data.get('Humidity', 'N/A')}
    - Vibration Level: {input_data.get('Vibration', 'N/A')}
    - Precipitation Level: {input_data.get('Precipitation', 'N/A')}

    Alerts & Incidents:
    - Alerts: {input_data.get('Alerts', 'None')}
    - Incident Type: {input_data.get('IncidentType', 'None')}
    - Incident Details: {input_data.get('IncidentDetails', 'None')}

    Maintenance Recommendation:
    Based on the ratios and performance metrics, it is advised to continue with scheduled maintenance and monitor the transformer closely for any abnormal behavior or environmental changes.
    """

    return report
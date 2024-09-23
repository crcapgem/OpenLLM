import boto3
import json
import pandas as pd 
import numpy as np 

def summarize_text(json_string):
    # Create a Bedrock client
    bedrock = boto3.client('bedrock-runtime')

    # convert to pd
    input_df = json.loads(json_string) 
    input_data = pd.DataFrame([input_df])

    # Generate a text-based report using only input data
    report = f"""
    Transformer ID: {input_data['TransformerID'].iloc[0]}
    InstallationDate: {input_data['InstallationDate'].iloc[0]}
    OperatingTime: {input_data['OperatingTime'].iloc[0]} years.
    LoadCondition: {input_data['LoadCondition'].iloc[0]}
    PerformanceMetrics: {input_data['PerformanceMetrics'].iloc[0]}

    Condition Summary:
    Health Index: {input_data['Health index'].iloc[0]}
    Based on the current dissolved gas analysis (DGA) and health index data, this transformer is expected to last an additional {input_data['Life expectation'].iloc[0]} years. 

    Specific gases dissolved in the transformer insulating oil, could help identify potential faults or degradation. Current levels are:
    - Hydrogen: {input_data['Hydrogen'].iloc[0]}
    - Oxygen (mg/L): {input_data['Oxigen'].iloc[0]}
    - Nitrogen (mg/L): {input_data['Nitrogen'].iloc[0]}
    - Methane (mg/L): {input_data['Methane'].iloc[0]}
    - CO (mg/L): {input_data['CO'].iloc[0]}
    - CO2 (mg/L): {input_data['CO2'].iloc[0]}
    - Ethylene (mg/L): {input_data['Ethylene'].iloc[0]}
    - Ethane (mg/L): {input_data['Ethane'].iloc[0]}
    - Acethylene (mg/L): {input_data['Acethylene'].iloc[0]}
    - DBDS (mg/L): {input_data['DBDS'].iloc[0]}

    Other Key indicators suggest that:
    - Power factor: {input_data['Power factor'].iloc[0]}.
    - Interfacial V: {input_data['Interfacial V'].iloc[0]}.
    - Dielectric rigidity: {input_data['Dielectric rigidity'].iloc[0]}.
    - Water content (mg/L): {input_data['Water content'].iloc[0]}

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
    - Precipitation Level (mm): {input_data.get('Precipitation', 'N/A')[0]}

    Alerts & Incidents:
    - Alerts: {input_data.get('Alerts', 'None')[0]}.
    - Incident Type: {input_data.get('IncidentType', 'None')[0]}.
    - Incident Details: {input_data.get('IncidentDetails', 'None')[0]}.

    Current Status:
    """
    # Prepare the input payload
    payload = {
        "inputText": f"Summarize the following text: {report}",
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1
        }
    }

    # Convert the payload to JSON string
    body = json.dumps(payload)

    # Invoke the model (using Amazon Titan Text model as an example)
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-express-v1",
        contentType="application/json",
        accept="application/json",
        body=body
    )

    # Parse the response
    response_body = json.loads(response['body'].read())
    summary = response_body['results'][0]['outputText']

    output = {"report": report, "gen_ai_summary": summary}

    return output

#### Example usage
#input_json = {"string": "Your long text to be summarized goes here. It can be multiple sentences or paragraphs long."}
#summary = summarize_text(input_json["string"])
#print("Summary:", summary)

# A Godd Condition transformer
""" 
good_condition_transformer = health_index_augdata.iloc[0]
print(good_condition_transformer['TransformerID'])
print(good_condition_transformer['PerformanceMetrics'])

out_summary = summarize_text(good_condition_transformer.to_json())
report_with_summary = out_summary['report'] + out_summary['gen_ai_summary']
print(report_with_summary)

# example save to s3 bucket
folder_name = 'transformer_reports/'
report_filename = bad_condition_transformer['TransformerID'] + "_report.txt"
print(report_filename)

# updload the file
s3.put_object(
    Bucket=bucket_name,
    Key=folder_name + report_filename,
    Body=report_with_summary
)

#### Run an all rows
# save to s3 bucket folder
folder_name = 'transformer_reports/'

# iterate over the rows of health_index_augdata
for index, row in health_index_augdata.iterrows():
    # generate a report for each transformer
    out_summary = summarize_text(row.to_json())
    report_with_summary = out_summary['report'] + out_summary['gen_ai_summary']
    # save the report to s3 bucket
    report_filename = row['TransformerID'] + "_report.txt"
    s3.put_object(
        Bucket=bucket_name,
        Key=folder_name + report_filename,
        Body=report_with_summary
    )
    print(f"{index} ...Report for {row['TransformerID']} saved to S3.")
"""
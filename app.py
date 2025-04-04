from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os
import xlrd
import pickle
import datetime
import calendar

import model
from model import optimal_rf_regressor

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Directory to save uploaded files
PROCESSED_FOLDER = 'processed' #Directory to save processed files

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/')
def upload_form():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        try:
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path) #save the uploaded file.

            df_cxo_file_ois = pd.read_excel(upload_path,sheet_name='OIS')

            def cxo_details(df_cxo_file_ois, sw):
                # if isinstance(row['Unnamed: 1'], str) and value_to_find in row['Unnamed: 1']:# for more generic
                for index, row in df_cxo_file_ois.iterrows():
                    if isinstance(row['Unnamed: 1'], str) and sw in row['Unnamed: 1']:
                        i = index + 1

                #i = sw # Start i at the correct initial row index
                new_swbd_df = pd.DataFrame()
                for index, row in df_cxo_file_ois.loc[i:].iterrows():
                    if pd.isna(row['Unnamed: 1']):  # ['Order Initiation Sheet\n(CXO Handover Document- LV)']): # Explicitly check for NaN in 'Column1'
                        # print(f"NaN found in 'Order Initiation Sheet\n(CXO Handover Document- LV)' at index {index}")
                        break  # Exit loop when encountering NaN in 'Column1'
                    else:
                        new_swbd_df = pd.concat([new_swbd_df, df_cxo_file_ois.loc[[index]]], ignore_index=True)

                else:  # This block executes if the loop completes without encountering a break
                    print("No null values found in 'Unnamed: 1')'. The entire DataFrame has been processed.")
                return new_swbd_df

            swbd_cxo = cxo_details(df_cxo_file_ois, 'Switchboard Details')

            set_columns = pd.DataFrame(columns=['-', 'Switchboard Tag as per PO', 'Greenfield/ R&M', 'Separately Billable?', 'Billable Qty',
                         'Equipment  Number', 'Panel or Item Qty', 'As booked Price in INR', 'As booked Cost in INR',
                         'Margin - OT %', '% ESE Final GM', 'Upstream Margin in % of Sales Value', '% CCO GM',
                         'PCC Panel',
                         'PCC Panel Qty', 'MCC Panel', 'MCC Panel Qty', '300/320mm pnl', 'EQPs'])

            swbd_cxo.columns = set_columns.columns
            swbd_cxo = swbd_cxo.fillna(0)
            # Making default value
            tf900 = 0
            tq3dffxd = 0
            tq3dfdo = 0
            tq3sff = 0
            tq3sfdo = 0

            type_board = 2
            form_of_separation = 1
            busbar_location = 1
            number_of_bus = 2
            system = 2
            fault_level = 2
            number_2_tier = 0
            iqr = 23
            average_cat_lead_time = 35
            no_of_days_MFG = 45
            cxo2din_diff = 120


            #return send_file(processed_path, as_attachment=True,
            #                 mimetype='application/vnd.openxmlformats.spreadsheetml.sheet')
            #                 #download_name=processed_filename,

            if (swbd_cxo['MCC Panel'] == 0).any():
                fixed_drawout = 1
            else:
                for index, row in swbd_cxo.iterrows():
                    if isinstance(row['MCC Panel'], str):
                        if 'FX' in row['MCC Panel']:  # tera
                            # swbd_cxo.loc[index, 'Fixed/Drawout'] = 2
                            # tq3dffxd=swbd_cxo.loc[0,'MCC Panel Qty'] #TQ3DF Fxd
                            fixed_drawout = 2


                        else:
                            # swbd_cxo.loc[index, 'Fixed/Drawout'] = 1
                            tq3dfdo = swbd_cxo.loc[0, 'MCC Panel Qty']  # TQ3DFDO
                            fixed_drawout = 1


                    else:
                        # swbd_cxo.loc[index, 'Fixed/Drawout'] = 1
                        tq3dfdo = swbd_cxo.loc[0, 'MCC Panel Qty']  # TQ3DFDO
                        fixed_drawout = 1

            # swbd_cxo['INFO'] = 0
            for index, row in swbd_cxo.iterrows():
                if isinstance(row['PCC Panel'], str):
                    if 'TS' in row['PCC Panel'] or 'TX' in row['PCC Panel']:
                        info = 2

                        # swbd_cxo.loc[index, 'INFO'] = 2
                    elif 'CDO+' in row['PCC Panel']:
                        info = 3

                        # swbd_cxo.loc[index, 'INFO'] = 3
                    else:
                        info = 1
                        # swbd_cxo.loc[index, 'INFO'] = 1

            cx_date = df_cxo_file_ois.loc[0, 'Date']  # CXO date data
            tf3 = swbd_cxo.loc[0, 'PCC Panel Qty']  # TF3

            rp = swbd_cxo.loc[0, '300/320mm pnl']  # RP

            mccmod = tq3dffxd + tq3dfdo + tq3sff + tq3sfdo
            pcctot = tf3 + tf900 + rp
            total_verticals = mccmod + pcctot

            if mccmod > 0:
                if pcctot > 0:
                    type_board == 3
                elif pcctot == 0:
                    type_board == 1
            else:
                type_board == 2

            product_code=  swbd_cxo.loc[0,'Equipment  Number']
            eqp = swbd_cxo.loc[0, 'EQPs']  # EQP data


            # info=swbd_cxo.loc[0,'INFO'] # INFO

            # fixed_drawout=swbd_cxo.loc[0,'Fixed/Drawout'] # fixed/drawout

            final_tp = swbd_cxo.loc[0, 'As booked Cost in INR']

            # Convert the date string to datetime object
            cx_date = pd.to_datetime(cx_date)

            # Convert to numeric
            yr = cx_date.year  # Access the year
            mn = cx_date.month  # Access the month
            dt = cx_date.day  # Access the day

            new_sample_data = np.array([[yr, mn, dt, tf3, tf900, rp, tq3sff, tq3sfdo, tq3dffxd, tq3dfdo,
                                         total_verticals, info, type_board, fault_level, busbar_location,
                                         form_of_separation,
                                         fixed_drawout, system, number_of_bus, number_2_tier, eqp, final_tp, iqr,
                                         average_cat_lead_time, no_of_days_MFG, cxo2din_diff]])

            #new_sample_data
            predicted_delivery = optimal_rf_regressor.predict(new_sample_data)
            predicted_date = cx_date + datetime.timedelta(days=int(predicted_delivery[0]))
            loading_month = calendar.month_name[predicted_date.month]
            loading_year = predicted_date.year


            return render_template("index.html",product_code=f"{product_code}",loading_month=f"{loading_month }{loading_year}")
        except Exception as e:
            return f"Error processing file: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
import flask
import pickle
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import model_utils
from model_utils import lr_bias_model
from model_utils import lr_bias_mitigated_model
from german_credit_model.bias_model import original_training_dataset


app = flask.Flask(__name__, template_folder='templates')
@app.route('/')

def main():
    return (flask.render_template('index.html'))

@app.route('/report')
def report():
    return (flask.render_template('report.html'))



@app.route('/bias_mitigated_implementation')
def bias_mitigated_implementation():
    return (flask.render_template('bias_mitigated_implementation.html'))

@app.route('/diffrence_show_bias')
def diffrence_show_bias():
    return (flask.render_template('diffrence_show_bias.html'))


@app.route('/EDA_german_datset')
def EDA_german_datset():
    return (flask.render_template('EDA_german_datset.html'))
@app.route('/correlation_analysis_german_db')
def correlation_analysis_german_db():
    return (flask.render_template('correlation_analysis_german_db.html'))



@app.route("/bias_check", methods=['GET', 'POST'])
def bias_check():
  if flask.request.method == 'GET':
      return (flask.render_template('bias_check.html'))
  if flask.request.method =='POST':
    #set form vaule as per test model.
    x = np.zeros(57)
    x[0] = int(flask.request.form['duration'])#month
    x[1] = float(flask.request.form['credit_amount'])
    x[2] = float(flask.request.form['investment_as_income_percentage'])
    x[3] = int(flask.request.form['residence_since'])
    x[4] = True if float(flask.request.form['age']) >= 25 else False
    x[5] = int(flask.request.form['number_of_credits'])
    x[6] = int(flask.request.form['dependent']) #people_liable_for
    x[7] = True if (flask.request.form['status']) == '...< 0 DM' else False
    x[8] = True if (flask.request.form['status']) == '0 <= ...< 200 DM' else False
    x[9] = True if (flask.request.form['status']) == '... >= 200 DM' else False
    x[10] = True if (flask.request.form['status']) == 'no checking account' else False
    x[11] = True if (flask.request.form['credit_history']) == 'No Credits' else False #credit_history=A30 #No Credit
    x[12] = True if (flask.request.form['credit_history']) == 'Paid Credits' else False#credit_history=A31 #Paid Credits
    x[13] = True if (flask.request.form['credit_history']) == 'Existing Credits' else False#credit_history=A32 #Existing Credits
    x[14] = True if (flask.request.form['credit_history']) == 'Delay in Past' else False#credit_history=A33 #Delay in Past
    x[15] = True if (flask.request.form['credit_history']) == 'Critical' else False#credit_history=A34 #critical
    x[16] = True if (flask.request.form['purpose']) == 'new car' else False #purpose=A40
    x[17] = True if (flask.request.form['purpose']) == 'used car' else False #purpose=A41
    x[18] = True if (flask.request.form['purpose']) == 'others' else False #purpose=A410
    x[19] = True if (flask.request.form['purpose']) == 'furniture/equipment' else False #purpose=A42
    x[20] = True if (flask.request.form['purpose']) == 'radio/television' else False #purpose=A43
    x[21] = True if (flask.request.form['purpose']) == 'domestic appliances' else False #purpose=A44
    x[22] = True if (flask.request.form['purpose']) == 'repairs' else False #purpose=A45
    x[23] = True if (flask.request.form['purpose']) == 'education' else False #purpose=A46
    x[24] = True if (flask.request.form['purpose']) == 'retraining' else False #purpose=A48
    x[25] = True if (flask.request.form['purpose']) == 'business' else False #purpose=A49
    x[26] = True if (flask.request.form['saving_accounts']) == '... < 100 DM' else False #savings=A61 saving low
    x[27] = True if (flask.request.form['saving_accounts']) == '100 <= ... < 500 DM' else False #savings=A62 medium
    x[28] = True if (flask.request.form['saving_accounts']) == '500 <= ... < 1000 DM' else False #savings=A63 high
    x[29] = True if (flask.request.form['saving_accounts']) == '.. >= 1000 DM' else False #savings=A64 very high
    x[30] = True if (flask.request.form['saving_accounts']) == 'unknown/ no savings account' else False #savings=A65 no saving
    x[31] = True if (flask.request.form['employment']) == 'unemployed'else False  #employment=A71 Unemployed
    x[32] = True if (flask.request.form['employment']) == '1'else False  #employment=A72 1
    x[33] = True if (flask.request.form['employment']) == '4'else False  #employment=A73 4
    x[34] = True if (flask.request.form['employment']) == '7'else False  #employment=A74 7
    x[35] = True if (flask.request.form['employment']) == '7+'else False  # employment=A75 7+
    x[36] = True if (flask.request.form['debtors']) == 'none'else False #other_debtors=A101
    x[37] = True if (flask.request.form['debtors']) == 'co-applicant'else False #other_debtors=A102
    x[38] = True if (flask.request.form['debtors']) == 'guarantor'else False #other_debtors=A103
    x[39] = True if (flask.request.form['property']) == 'real estate' else False #property=A121 real estate
    x[40] = True if (flask.request.form['property']) == 'life insurance'else False #property=A122 life insurance
    x[41] = True if (flask.request.form['property']) == 'car or other' else False #property=A123 car or other
    x[42] = True if (flask.request.form['property']) == 'no property' else False #property=A124 no
    x[43] = True if (flask.request.form['installment_plans']) == 'bank' else False #installment_plans=A141 bank
    x[44] = True if (flask.request.form['installment_plans']) == 'stors' else False #installment_plans=A142 stors
    x[45] = True if (flask.request.form['installment_plans']) == 'none' else False #installment_plans=A143 none
    x[46] = True if (flask.request.form['housing']) == 'rent' else False #housing=A151 rent
    x[47] = True if (flask.request.form['housing']) == 'own' else False #housing=A152  own
    x[48] = True if (flask.request.form['housing']) == 'free' else False #housing=A153  free
    x[49] = True if (flask.request.form['job']) == 'unemployed/unskilled-non resident' else False  #skill_level=A171 # A171 : unemployed/ unskilled - non-resident
    x[50] = True if (flask.request.form['job']) == 'unskilled-resident' else False  #skill_level=A172# A172 : unskilled - resident
    x[51] = True if (flask.request.form['job']) == 'skilled employee/official' else False  #skill_level=A173 # A173 : skilled employee / official
    x[52] = True if (flask.request.form['job']) == 'highly skilled' else False  #skill_level=A174 # # A174 : management/ self-employed/highly qualified employee/ officer
    x[53] = True if (flask.request.form['telephone']) == 'none' else False  #telephone=A191 none
    x[54] = True if (flask.request.form['telephone']) == 'yes' else False  #telephone=A192 yes
    x[55] = True if (flask.request.form['foreign_worker']) == 'yes' else False  #foreign_worker=A201 yes
    x[56] = True if (flask.request.form['foreign_worker']) == 'no' else False  #foreign_worker=A202 no
    print ("x-------------------", x)
    prediction = model_utils.lr_bias_mitigated_model(x)
    print("prediction-------------", prediction)
    if prediction == 1:
      res = ('🎊🎊Congratulations! your Loan Application has been Approved!🎊🎊')
    else:
      res = ("😔😔Unfortunatly your Loan Application has been Denied😔😔")
    print(res)

    return flask.render_template('bias_check.html',
                                    original_input=x, result=res)

@app.route("/loan_application", methods=['GET', 'POST'])
def loan_application():
  if flask.request.method == 'GET':
      return (flask.render_template('loan_application.html'))
  if flask.request.method =='POST':
    #set form vaule as per test model.
    x = np.zeros(57)
    x[0] = int(flask.request.form['duration'])#month
    x[1] = float(flask.request.form['credit_amount'])
    x[2] = float(flask.request.form['investment_as_income_percentage'])
    x[3] = int(flask.request.form['residence_since'])
    x[4] = True if float(flask.request.form['age']) >= 25 else False
    x[5] = int(flask.request.form['number_of_credits'])
    x[6] = int(flask.request.form['dependent']) #people_liable_for
    x[7] = True if (flask.request.form['status']) == '...< 0 DM' else False
    x[8] = True if (flask.request.form['status']) == '0 <= ...< 200 DM' else False
    x[9] = True if (flask.request.form['status']) == '... >= 200 DM' else False
    x[10] = True if (flask.request.form['status']) == 'no checking account' else False
    x[11] = True if (flask.request.form['credit_history']) == 'No Credits' else False #credit_history=A30 #No Credit
    x[12] = True if (flask.request.form['credit_history']) == 'Paid Credits' else False#credit_history=A31 #Paid Credits
    x[13] = True if (flask.request.form['credit_history']) == 'Existing Credits' else False#credit_history=A32 #Existing Credits
    x[14] = True if (flask.request.form['credit_history']) == 'Delay in Past' else False#credit_history=A33 #Delay in Past
    x[15] = True if (flask.request.form['credit_history']) == 'Critical' else False#credit_history=A34 #critical
    x[16] = True if (flask.request.form['purpose']) == 'new car' else False #purpose=A40
    x[17] = True if (flask.request.form['purpose']) == 'used car' else False #purpose=A41
    x[18] = True if (flask.request.form['purpose']) == 'others' else False #purpose=A410
    x[19] = True if (flask.request.form['purpose']) == 'furniture/equipment' else False #purpose=A42
    x[20] = True if (flask.request.form['purpose']) == 'radio/television' else False #purpose=A43
    x[21] = True if (flask.request.form['purpose']) == 'domestic appliances' else False #purpose=A44
    x[22] = True if (flask.request.form['purpose']) == 'repairs' else False #purpose=A45
    x[23] = True if (flask.request.form['purpose']) == 'education' else False #purpose=A46
    x[24] = True if (flask.request.form['purpose']) == 'retraining' else False #purpose=A48
    x[25] = True if (flask.request.form['purpose']) == 'business' else False #purpose=A49
    x[26] = True if (flask.request.form['saving_accounts']) == '... < 100 DM' else False #savings=A61 saving low
    x[27] = True if (flask.request.form['saving_accounts']) == '100 <= ... < 500 DM' else False #savings=A62 medium
    x[28] = True if (flask.request.form['saving_accounts']) == '500 <= ... < 1000 DM' else False #savings=A63 high
    x[29] = True if (flask.request.form['saving_accounts']) == '.. >= 1000 DM' else False #savings=A64 very high
    x[30] = True if (flask.request.form['saving_accounts']) == 'unknown/ no savings account' else False #savings=A65 no saving
    x[31] = True if (flask.request.form['employment']) == 'unemployed'else False  #employment=A71 Unemployed
    x[32] = True if (flask.request.form['employment']) == '1'else False  #employment=A72 1
    x[33] = True if (flask.request.form['employment']) == '4'else False  #employment=A73 4
    x[34] = True if (flask.request.form['employment']) == '7'else False  #employment=A74 7
    x[35] = True if (flask.request.form['employment']) == '7+'else False  # employment=A75 7+
    x[36] = True if (flask.request.form['debtors']) == 'none'else False #other_debtors=A101
    x[37] = True if (flask.request.form['debtors']) == 'co-applicant'else False #other_debtors=A102
    x[38] = True if (flask.request.form['debtors']) == 'guarantor'else False #other_debtors=A103
    x[39] = True if (flask.request.form['property']) == 'real estate' else False #property=A121 real estate
    x[40] = True if (flask.request.form['property']) == 'life insurance'else False #property=A122 life insurance
    x[41] = True if (flask.request.form['property']) == 'car or other' else False #property=A123 car or other
    x[42] = True if (flask.request.form['property']) == 'no property' else False #property=A124 no
    x[43] = True if (flask.request.form['installment_plans']) == 'bank' else False #installment_plans=A141 bank
    x[44] = True if (flask.request.form['installment_plans']) == 'stors' else False #installment_plans=A142 stors
    x[45] = True if (flask.request.form['installment_plans']) == 'none' else False #installment_plans=A143 none
    x[46] = True if (flask.request.form['housing']) == 'rent' else False #housing=A151 rent
    x[47] = True if (flask.request.form['housing']) == 'own' else False #housing=A152  own
    x[48] = True if (flask.request.form['housing']) == 'free' else False #housing=A153  free
    x[49] = True if (flask.request.form['job']) == 'unemployed/unskilled-non resident' else False  #skill_level=A171 # A171 : unemployed/ unskilled - non-resident
    x[50] = True if (flask.request.form['job']) == 'unskilled-resident' else False  #skill_level=A172# A172 : unskilled - resident
    x[51] = True if (flask.request.form['job']) == 'skilled employee/official' else False  #skill_level=A173 # A173 : skilled employee / official
    x[52] = True if (flask.request.form['job']) == 'highly skilled' else False  #skill_level=A174 # # A174 : management/ self-employed/highly qualified employee/ officer
    x[53] = True if (flask.request.form['telephone']) == 'none' else False  #telephone=A191 none
    x[54] = True if (flask.request.form['telephone']) == 'yes' else False  #telephone=A192 yes
    x[55] = True if (flask.request.form['foreign_worker']) == 'yes' else False  #foreign_worker=A201 yes
    x[56] = True if (flask.request.form['foreign_worker']) == 'no' else False  #foreign_worker=A202 no
    print ("x-------------------", x)
    prediction = model_utils.lr_bias_model(x)
    print("prediction-------------", prediction)
    if prediction == 1:
      res = ('🎊🎊Congratulations! your Loan Application has been Approved!🎊🎊')
    else:
      res = ("😔😔Unfortunatly your Loan Application has been Denied😔😔")
    print(res)

    return flask.render_template('loan_application.html',
                                    original_input=x, result=res)

if __name__ == '__main__':
    app.run(debug=True)
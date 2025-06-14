import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

risk = ctrl.Antecedent(np.arange(0.01, 1, 0.001), 'risk')
risk["High"] = fuzz.trapmf(risk.universe, (0.01, 0.01, 0.45, 0.50)) #Caido
risk["Medium"] = fuzz.trapmf(risk.universe, (0.4, 0.45, 0.70, 0.75)) # Mantenido
risk["Low"] = fuzz.trapmf(risk.universe, (0.7, 0.80, 1, 1)) # Normalizado

a_days = ctrl.Antecedent(np.arange(0, 230, 1), 'a_days')
a_days["Low"] = fuzz.trapmf(a_days.universe, [0, 1, 40, 50])
a_days["Medium"] = fuzz.trapmf(a_days.universe, [40, 46, 70, 80])
a_days["Medium-High"] = fuzz.trapmf(a_days.universe, [65, 71, 130, 140])
a_days["High"] = fuzz.trapmf(a_days.universe, [120, 131, 160, 170])
a_days["Very High"] = fuzz.trapmf(a_days.universe, [155, 160, 230, 230])

total_balance_CDC = ctrl.Antecedent(np.linspace(0, 350_000_000, 5000), "total_balance_CDC")
total_balance_CDC["Very Low"] = fuzz.trapmf(total_balance_CDC.universe, [0, 0, 500_000, 1_000_000])
total_balance_CDC["Low"] = fuzz.trapmf(total_balance_CDC.universe, [800_000, 1_250_000, 4_000_000, 5_000_000])
total_balance_CDC["Medium Low"] = fuzz.trapmf(total_balance_CDC.universe, [4_500_000, 5_500_000, 8_000_000, 10_000_000])
total_balance_CDC["Medium"] = fuzz.trapmf(total_balance_CDC.universe, [8_500_000, 10_500_000, 18_000_000, 20_000_000])
total_balance_CDC["High"] = fuzz.trapmf(total_balance_CDC.universe, [18_500_000, 20_500_000, 28_000_000, 30_000_000])
total_balance_CDC["Very High"] = fuzz.trapmf(total_balance_CDC.universe, [27_000_000, 31_000_000, 350_000_000, 350_000_000])

total_balance_TDC = ctrl.Antecedent(np.linspace(0, 200_000_000, 5000), "total_balance_TDC")
total_balance_TDC["Very Low"] = fuzz.trapmf(total_balance_TDC.universe, [0, 0, 500_000, 1_000_000])
total_balance_TDC["Low"] = fuzz.trapmf(total_balance_TDC.universe, [800_000, 1_100_000, 2_350_000, 2_500_000])
total_balance_TDC["Medium Low"] = fuzz.trapmf(total_balance_TDC.universe, [2_000_000, 2_700_000, 4_500_000, 5_000_000])
total_balance_TDC["Medium"] = fuzz.trapmf(total_balance_TDC.universe, [4_300_000, 5_300_000, 7_300_000, 7_500_000])
total_balance_TDC["High"] = fuzz.trapmf(total_balance_TDC.universe, [6_800_000, 8_000_000, 9_500_000, 10_000_000])
total_balance_TDC["Very High"] = fuzz.trapmf(total_balance_TDC.universe, [9_000_000, 10_500_000, 200_000_000, 200_000_000])

overdue_balance = ctrl.Antecedent(np.linspace(0, 35_000_000, 5000), "overdue_balance")
overdue_balance["Low"] = fuzz.trapmf(overdue_balance.universe, [0, 0, 300_000, 400_000])
overdue_balance["Medium Low"] = fuzz.trapmf(overdue_balance.universe, [350_000, 400_000, 700_000, 800_000])
overdue_balance["Medium"] = fuzz.trapmf(overdue_balance.universe, [750_000, 900_000, 1_800_000, 2_000_000])
overdue_balance["High"] = fuzz.trapmf(overdue_balance.universe, [1_800_000, 2_000_000, 4_500_000, 5_000_000])
overdue_balance["Very High"] = fuzz.trapmf(overdue_balance.universe, [4_800_000, 5_000_000, 35_000_000, 35_000_000])

overdue_installment = ctrl.Antecedent(np.linspace(0, 6_000_000, 5000), "overdue_installment")
overdue_installment["Low"] = fuzz.trapmf(overdue_installment.universe, [0, 0, 100_000, 250_000])
overdue_installment["Medium Low"] = fuzz.trapmf(overdue_installment.universe, [200_000, 300_000, 450_000, 500_000])
overdue_installment["Medium"] = fuzz.trapmf(overdue_installment.universe, [450_000, 550_000, 750_000, 800_000])
overdue_installment["High"] = fuzz.trapmf(overdue_installment.universe, [750_000, 850_000, 1_100_000, 1_250_000])
overdue_installment["Very High"] = fuzz.trapmf(overdue_installment.universe, [1_200_000, 1_300_000, 6_000_000, 6_000_000])

prioritization = ctrl.Antecedent(np.arange(1, 6, 1), "prioritization")
prioritization["High"] = fuzz.trimf(prioritization.universe, [1, 1, 2])
prioritization["Medium"] = fuzz.trimf(prioritization.universe, [2, 3, 4])
prioritization["Low"] = fuzz.trimf(prioritization.universe, [3, 4, 5])

payment_habit = ctrl.Antecedent(np.arange(0, 13, 1), "payment_habit")
payment_habit["Bad"] = fuzz.trapmf(payment_habit.universe, [0, 0, 1, 2])
payment_habit["Regular"] = fuzz.trapmf(payment_habit.universe, [1, 2, 3, 4])
payment_habit["Good"] = fuzz.trapmf(payment_habit.universe, [3, 4, 6, 13])

total_contacts = ctrl.Antecedent(np.arange(0, 50, 1), "total_contacts")
total_contacts["Low"] = fuzz.trapmf(total_contacts.universe, [0, 0, 2, 3])
total_contacts["Medium"] = fuzz.trapmf(total_contacts.universe, [2, 3, 5, 6])
total_contacts["High"] = fuzz.trapmf(total_contacts.universe, [5, 6, 10, 50])

total_agreements = ctrl.Antecedent(np.arange(0, 30, 1), "total_agreements")
total_agreements["Low"] = fuzz.trapmf(total_agreements.universe, [3, 4, 6, 30])
total_agreements["Medium"] = fuzz.trapmf(total_agreements.universe, [2, 2, 3, 4])
total_agreements["High"] = fuzz.trapmf(total_agreements.universe, [0, 0, 1, 2])

client_risk = ctrl.Consequent(np.arange(0, 6, 0.1), "client_risk")
client_risk["Low"] = fuzz.trapmf(client_risk.universe, [0, 0, 1, 1.8])
client_risk["Mild"] = fuzz.trapmf(client_risk.universe, [1, 1.5, 2.5, 3])
client_risk["Medium"] = fuzz.trapmf(client_risk.universe, [2, 2.8, 3.5, 4])
client_risk["High"] = fuzz.trapmf(client_risk.universe, [3.5, 4, 4.5, 5])
client_risk["Critical"] = fuzz.trapmf(client_risk.universe, [4.5, 5, 6, 6])


rules = [
    ctrl.Rule(risk["Low"] & (a_days["Low"] | a_days["Medium"]), client_risk["Low"]),
    ctrl.Rule(risk["Low"] & (overdue_installment["Low"] | overdue_installment["Medium Low"]), client_risk["Low"]),
    ctrl.Rule(risk["Low"] & payment_habit["Good"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & (overdue_balance["Low"] | overdue_balance["Medium Low"]), client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_contacts["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_balance_TDC["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_balance_CDC["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & prioritization["High"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_agreements["High"], client_risk["Low"]),

    ctrl.Rule(risk["Low"] & total_balance_CDC["Medium Low"] & overdue_installment["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_balance_TDC["Medium"] & overdue_installment["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & prioritization["High"] & total_contacts["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & (total_contacts["Low"] | total_agreements["High"]) & prioritization["Medium"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & overdue_balance["Medium Low"] & a_days["Medium"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_balance_TDC["Very High"] & total_balance_CDC["Very High"] & prioritization["High"], client_risk["Low"]),

    ctrl.Rule(risk["Low"] & a_days["Low"] & overdue_installment["Low"] & payment_habit["Good"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & (overdue_balance["Low"] | overdue_installment["Low"]) & total_agreements["High"] & prioritization["Medium"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & total_balance_CDC["Medium"] & total_contacts["Low"] & payment_habit["Good"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & a_days["Medium"] & overdue_installment["Low"] & prioritization["High"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & overdue_balance["Medium Low"] & total_contacts["Low"] & payment_habit["Good"], client_risk["Low"]),

    ctrl.Rule(risk["Low"] & a_days["Medium"] & (total_contacts["Low"] | total_agreements["High"]) & overdue_installment["Low"] & prioritization["High"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & overdue_balance["Medium Low"] & a_days["Medium"] & prioritization["High"] & payment_habit["Good"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & overdue_installment["Low"] & total_contacts["Low"] & prioritization["Medium"] & total_balance_TDC["Medium Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & payment_habit["Good"] & a_days["Low"] & total_agreements["High"] & overdue_installment["Low"], client_risk["Low"]),
    ctrl.Rule(risk["Low"] & prioritization["High"] & overdue_balance["Medium Low"] & overdue_installment["Low"] & total_contacts["Low"], client_risk["Low"]),
#
    ctrl.Rule(risk["Medium"] & a_days["Low"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & overdue_installment["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & total_contacts["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium"] & total_balance_TDC["Medium Low"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium"] & total_balance_CDC["Medium Low"], client_risk["Mild"]),
    
    ctrl.Rule(risk["Low"] & overdue_installment["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & total_contacts["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & total_contacts["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & a_days["Low"] & total_agreements["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & overdue_installment["Medium"] & a_days["Medium-High"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & total_balance_TDC["Medium"] & prioritization["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & total_balance_CDC["Medium"] & prioritization["Medium"], client_risk["Mild"]),
    
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & overdue_installment["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & overdue_installment["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & a_days["Medium"] & total_contacts["Low"] & total_balance_TDC["Low"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & a_days["Medium"] & total_contacts["Low"] & total_balance_CDC["Low"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & a_days["Low"] & prioritization["Medium"] & overdue_installment["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & total_balance_CDC["Medium Low"] & total_contacts["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & total_balance_TDC["Medium Low"] & total_contacts["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & overdue_installment["Low"] & total_agreements["Medium"] & a_days["Low"], client_risk["Mild"]),
    
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & overdue_installment["Medium"] & total_contacts["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & overdue_installment["Medium"] & total_contacts["Medium"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & a_days["Medium-High"] & overdue_installment["Medium"] & total_balance_CDC["Medium"] & prioritization["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & a_days["Medium-High"] & overdue_installment["Medium"] & total_balance_TDC["Medium"] & prioritization["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_agreements["High"] & a_days["Low"] & total_balance_TDC["Low"] & overdue_installment["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & total_agreements["High"] & a_days["Low"] & total_balance_CDC["Low"] & overdue_installment["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Low"] & overdue_installment["Medium"] & a_days["Medium-High"] & total_contacts["Low"] & prioritization["High"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium-High"] & total_contacts["Medium"] & total_balance_CDC["Medium Low"] & overdue_installment["Medium"], client_risk["Mild"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium-High"] & total_contacts["Medium"] & total_balance_TDC["Medium Low"] & overdue_installment["Medium"], client_risk["Mild"]),
#
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_contacts["High"] & a_days["Medium"], client_risk["Medium"]),

    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_contacts["High"] & (a_days["Medium"] | a_days["Medium-High"]), client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["High"] & a_days["High"] & total_agreements["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["High"] & a_days["High"] & total_agreements["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & prioritization["Medium"] & total_agreements["High"], client_risk["Medium"]),

    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["High"] & a_days["Medium-High"] & total_agreements["Medium"] & prioritization["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["High"] & a_days["Medium-High"] & total_agreements["Medium"] & prioritization["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_contacts["High"] & a_days["Medium-High"] & total_agreements["Low"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["High"] & total_agreements["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["High"] & total_agreements["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["Medium"] & prioritization["Low"], client_risk["Medium"]),

    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & a_days["Medium"] & total_contacts["High"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & a_days["Medium"] & total_contacts["High"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["High"] & a_days["Medium-High"] & total_agreements["Medium"] & total_contacts["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["High"] & a_days["Medium-High"] & total_agreements["Medium"] & total_contacts["Medium"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["Medium-High"] & total_contacts["High"] & prioritization["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["Medium-High"] & total_contacts["High"] & prioritization["Medium"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & a_days["Medium"] & total_agreements["Low"] & total_contacts["High"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & a_days["Medium"] & total_agreements["Low"] & total_contacts["High"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["High"] & a_days["Medium-High"] & total_agreements["Low"] & prioritization["Low"], client_risk["Medium"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["High"] & a_days["Medium-High"] & total_agreements["Low"] & prioritization["Low"], client_risk["Medium"]),

#
    ctrl.Rule(risk["Medium"] & a_days["Medium-High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & overdue_balance["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & a_days["High"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["High"] & prioritization["Low"], client_risk["High"]),
    ctrl.Rule(risk["Medium"] & total_balance_TDC["High"] & prioritization["Low"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & overdue_balance["Medium"], client_risk["High"]),
    
    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium"] & a_days["Medium-High"], client_risk["High"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium"] & a_days["Medium-High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & overdue_balance["Medium Low"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & a_days["High"] & prioritization["Medium"] & total_contacts["High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["High"] & overdue_balance["High"] & total_agreements["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["High"] & overdue_balance["High"] & total_agreements["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & a_days["Medium-High"] & prioritization["Low"] & overdue_installment["Medium Low"], client_risk["High"]),

    ctrl.Rule(risk["Medium"] & total_balance_TDC["Medium Low"] & a_days["High"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["Medium"] & total_balance_CDC["Medium Low"] & a_days["High"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & overdue_balance["Medium Low"] & prioritization["Medium"] & total_contacts["High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & a_days["High"] & total_balance_CDC["High"] & total_agreements["Medium"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & a_days["High"] & total_balance_TDC["High"] & total_agreements["Medium"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["High"] & overdue_balance["High"] & prioritization["Low"] & a_days["Medium-High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["High"] & overdue_balance["High"] & prioritization["Low"] & a_days["Medium-High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & total_agreements["Low"] & total_balance_TDC["Medium"] & prioritization["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & total_agreements["Low"] & total_balance_CDC["Medium"] & prioritization["Medium"], client_risk["High"]),

    ctrl.Rule(risk["Medium"] & a_days["High"] & overdue_balance["Medium Low"] & prioritization["Low"] & overdue_installment["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["Medium"] & overdue_balance["Medium Low"] & a_days["Medium-High"] & total_contacts["High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["Medium"] & overdue_balance["Medium Low"] & a_days["Medium-High"] & total_contacts["High"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["High"] & overdue_balance["High"] & a_days["High"] & overdue_installment["Medium"] & prioritization["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["High"] & overdue_balance["High"] & a_days["High"] & overdue_installment["Medium"] & prioritization["Medium"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & total_agreements["Low"] & total_balance_TDC["Medium Low"] & a_days["Medium-High"] & prioritization["Low"], client_risk["High"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & total_agreements["Low"] & total_balance_CDC["Medium Low"] & a_days["Medium-High"] & prioritization["Low"], client_risk["High"]),
    ctrl.Rule(risk["Medium"] & a_days["Medium-High"] & overdue_balance["Medium"] & overdue_installment["Medium Low"] & total_agreements["Medium"], client_risk["High"]),
#

    ctrl.Rule(risk["High"] & a_days["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & overdue_balance["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_installment["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["Very High"] & overdue_balance["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["Very High"] & overdue_balance["Very High"], client_risk["Critical"]),

    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_installment["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & prioritization["Low"] & overdue_balance["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & total_balance_CDC["Very High"] & overdue_balance["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & total_balance_TDC["Very High"] & overdue_balance["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & overdue_installment["Very High"] & total_contacts["High"], client_risk["Critical"]),

    ctrl.Rule(risk["High"] & a_days["Very High"] & prioritization["Low"] & overdue_installment["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & overdue_balance["Very High"] & overdue_installment["Very High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_contacts["High"] & total_agreements["High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_balance["Very High"] & overdue_installment["Very High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & total_contacts["High"] & prioritization["Low"], client_risk["Critical"]),

    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_balance["Very High"] & overdue_installment["Very High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["Very High"] & a_days["Very High"] & overdue_installment["Very High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["Very High"] & a_days["Very High"] & overdue_installment["Very High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_balance["Very High"] & total_contacts["High"] & prioritization["Low"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_CDC["Very High"] & total_contacts["High"] & prioritization["Low"] & overdue_installment["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & total_balance_TDC["Very High"] & total_contacts["High"] & prioritization["Low"] & overdue_installment["Very High"], client_risk["Critical"]),
    ctrl.Rule(risk["High"] & a_days["Very High"] & overdue_balance["Very High"] & total_contacts["High"] & total_agreements["High"], client_risk["Critical"]),
]

control_system = ctrl.ControlSystem(rules)
simulation = ctrl.ControlSystemSimulation(control_system)


def evaluar_cliente_fuzzy(inputs: dict) -> dict:
    sim = ctrl.ControlSystemSimulation(control_system)

    try:
        for key, value in inputs.items():
            sim.input[key] = value

        sim.compute()

        crisp = sim.output['client_risk']
        grados = {
            label: fuzz.interp_membership(client_risk.universe, mf.mf, crisp)
            for label, mf in client_risk.terms.items()
        }
        label_final = max(grados, key=grados.get)

        return {
            "valor_crisp": round(crisp, 3),
            "segmento_fuzzy": label_final
        }

    except Exception as e:
        return {
            "valor_crisp": None,
            "segmento_fuzzy": "Sin evaluación"
        }



from joblib import Parallel, delayed

def evaluar_batch_fuzzy(df_inputs: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    resultados = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluar_cliente_fuzzy)(row.to_dict())
        for _, row in df_inputs.iterrows()
    )
    return pd.DataFrame(resultados)

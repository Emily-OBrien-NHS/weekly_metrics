import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

#===========================================================================
#                        Read in Data from Stored Proc
#===========================================================================
#Make connection, run stored proc and read in the results to a dataframe. 
#close the connection.
def GetData():
    sdmart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
                                'trusted_connection=yes&driver=ODBC+Driver+17'\
                                '+for+SQL+Server')
    conn = sdmart_engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute('SET NOCOUNT ON; EXECUTE sp_er_1151')
    results = cursor.fetchall()
    df = pd.DataFrame.from_records(results,
                                columns=[col[0] for col in cursor.description])
    cursor.close()
    conn.close()
    sdmart_engine.dispose()
    #Filter to last 6 months of data
    df = df.loc[pd.to_datetime(df['dte'])
                > pd.to_datetime(date.today() - relativedelta(months=+2))].copy()
    #Remove unnecesary and weekly metrics
    remove_metrics = ['#NOF - % within 36hrs',
                    '#NOF - Discharges',
                    '#NOF - Mean LoS days (inc ED & Community stay)',
                    '#NOF - Mean time to surgery (hrs)',
                    'ED Ambulance mean TTIA',
                    'Inpatient CT - Activity',
                    'Inpatient CT - ROTTs',
                    'Inpatient CT - Within 4hrs',
                    'Medicine - % of discharges with amber box the day before',
                    'Surgery - % of discharges with amber box the day before',
                    'Medicine - Amber Box Patients (as at 23:55)',
                    'Surgery - Amber Box Patients (as at 23:55)',
                    'MTU mean LoS (hours)',
                    'Saturday discharges as % of weekday discharges',
                    'SDEC Patients through AAU',
                    'Sunday discharges as % of weekday discharges',
                    'Patients through MTU',
                    'Weekend discharges as % of weekday discharges']
    df = df.loc[~(df['metric'].isin(remove_metrics))
                & ~(df['metric'].str.lower().str.contains('weekly'))].copy()
    df['data'] = np.where(df['metric'].str.contains('%'), df['data']*100, df['data'])
    #pivot to make a column per data
    pivot = (df.drop_duplicates(subset=['dte', 'metric'])
             .pivot(index='dte', columns='metric', values='data'))

    #Orignal
    original = pivot.loc[pivot.index
                        < pd.to_datetime(date.today()
                                        - relativedelta(days=+7))].copy()
    #Recent
    recent = pivot.loc[pivot.index
                    >= pd.to_datetime(date.today()
                                        - relativedelta(days=+7))].copy()
    return df, pivot, original, recent


#===========================================================================
#                       outlier weeks & recent trend
#===========================================================================
def OutliersAndRecent(original, recent):
    metrics = original.columns
    all_outliers = []
    recent_trend = []
    for metric in metrics:
        #Get series for each metric, calculate LQ, UQ and IQR
        og_col = original[metric].copy()
        LQ = og_col.quantile(0.25)
        UQ = og_col.quantile(0.75)
        IQR = UQ - LQ
        upper = UQ + 1.5*IQR
        lower = LQ - 1.5*IQR
        #Turn into dataframe, add blank type column
        col = pd.DataFrame(recent[metric].copy())
        col['type'] = ''
        #Find the outliers and label them as high or low
        col.loc[col[metric] < lower, 'type'] = 'Low'
        col.loc[col[metric] > upper, 'type'] = 'High'
        #Filter the outliers, add the metric and add to the list to turn into a df
        outliers = col.loc[col['type'] != ''].copy().reset_index()
        outliers['metric'] = metric
        all_outliers += outliers.values.tolist()

        #recent trend
        if recent[metric].sum() > 0:
            gt_pvalue = stats.mannwhitneyu(recent[metric].to_numpy(),
                                        original[metric].to_numpy(),
                                        alternative='greater',
                                        nan_policy='omit')[1]
            lt_pvalue = stats.mannwhitneyu(recent[metric].to_numpy(),
                                        original[metric].to_numpy(),
                                        alternative='less',
                                        nan_policy='omit')[1]
            og_mean = original[metric].mean()
            l7d_mean = recent[metric].mean()
            if gt_pvalue < 0.05:
                output = 'High'
                recent_trend.append([metric, output, og_mean, l7d_mean])
            elif lt_pvalue < 0.05:
                output = 'Low'
                recent_trend.append([metric, output, og_mean, l7d_mean])

    #create dataframe of all outliers.
    outliers = (pd.DataFrame(all_outliers,
                            columns=['Date', 'Data', 'Type', 'Metric'])
                            [['Metric', 'Date', 'Data', 'Type']])
    outliers['Data'] = outliers['Data'].round(2)
    #create dataframe of recent trends.
    recent_trend = pd.DataFrame(recent_trend,
                                columns=['Metric', 'Last 7 Day Trend',
                                         'Historical Mean', 'Last 7 Day Mean'])
    return outliers, recent_trend

#===========================================================================
#                            correlations
#===========================================================================
#Find correlations of entire data, then compare to correlations of just the last
#week, if anything significantly different, flag.

def Correlations(original, recent):
    #Get the correlatins of the historical and recent data
    og_corr = original.corr(method='pearson').unstack()
    og_corr.index.names = ['Metric 1', 'Metric 2']
    og_corr = pd.DataFrame(og_corr, columns=['Original'])
    recent_corr = recent.corr(method='pearson').unstack()
    recent_corr.index.names = ['Metric 1', 'Metric 2']
    recent_corr = pd.DataFrame(recent_corr, columns=['Recent'])
    #Join and compare
    correlations = og_corr.join(recent_corr).reset_index()
    correlations['pair'] = [sorted(i) for i in correlations[['Metric 1', 'Metric 2']].values.tolist()]
    correlations = correlations.drop_duplicates(subset='pair')
    #Filter out where correlation is 1 or -1 (round for float point error)
    correlations = correlations.loc[
                   (correlations['Metric 1'] != correlations['Metric 2'])
                    & (abs(correlations['Original']).round(7) != 1)
                    & (abs(correlations['Recent']).round(7) != 1),
                    ['Metric 1', 'Metric 2', 'Original',
                     'Recent']].reset_index(drop=True).copy()
    critical_value = abs(stats.norm.ppf(0.01))
    sig_diff_test = []
    for i in range(len(correlations)):
        row = correlations.iloc[i]
        data1, data2 = row[['Metric 1', 'Metric 2']]
        #get the two data series
        og = original[[data1, data2]].dropna(how='any')
        rec = recent[[data1, data2]].dropna(how='any')
        #transform the r/coeficient values into z scores (fishers r to z transformation)
        n1 = len(og)
        n2 = len(rec)    
        r1, r2 = row[['Original', 'Recent']]
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)
        #determine the observed z test statistic:
        if (n1 > 3) and (n2 > 3):
            z_obs = ((z1 - z2) / (((1/(n1-3)) + (1/(n2-3)))**0.5))
        else:
            z_obs = np.nan
        sig_diff_test.append([round(z_obs, 2), round(abs(z_obs), 2),
                              abs(z_obs) > critical_value])
        #If observed is less than the critical value, then the difference is significant
        #https://www.statisticssolutions.com/comparing-correlation-coefficients/#:~:text=The%20way%20to%20do%20this,the%20observed%20z%20test%20statistic.
    correlations[['Z Obs Score', 'Absolute Score', 'Significant Difference']] = sig_diff_test
    #Filter to only those where the correlation has changed significantly
    correlations = (correlations.loc[correlations['Significant Difference']].copy()
            .sort_values(by='Absolute Score', ascending=False)
            .reset_index(drop=True).reset_index())
    #Melt so one metric column to create relationships in BI
    correlations = correlations.melt(id_vars=['index', 'Z Obs Score', 'Absolute Score'],
                                     value_vars=['Metric 1', 'Metric 2']
                                     ).sort_values(by='index')
    correlations.columns = ['index', 'Obs Score', 'Absolute Score', 'Metric No.', 'Metric']
    return correlations

def Forecasts(pivot):
    #Take a copy of the data
    pivot_ = pivot.copy()
    #Remove weekly metrics
    value_counts = pivot_.count()
    drop_cols = value_counts[value_counts <= 10].index
    pivot_ = pivot_.drop(drop_cols, axis=1)
    #To ensure we keep the key metrics, fill their nans in then drop all
    #other nan rows
    key_metrics = ['ED - Patients awaiting admission at 8AM',
                   'Ambulance handovers - Hours lost >15 minutes',
                   'ED mean LoS (mins)',
                   'No right to reside - % of occupied beds']
    pivot_[key_metrics] = pivot_[key_metrics].fillna(0)
    pivot_ = pivot_.dropna(axis=1)
    other_metrics = [i for i in pivot_.columns if i not in key_metrics]
    #Split to current position and past data
    curr_pos = pivot_.loc[pivot_.index == pd.to_datetime(date.today()
                                        - relativedelta(days=+1))].copy()
    past_pos = pivot_.loc[pivot_.index < pd.to_datetime(date.today()
                                        - relativedelta(days=+1))].copy()
    X_pred = curr_pos[other_metrics]
    #Work out which metrics have a high recent value
    high = []
    low = []
    for metric in pivot_.columns:
        gt_pvalue = stats.mannwhitneyu(curr_pos[metric].to_numpy(),
                                       past_pos[metric].to_numpy(),
                                       alternative='greater',
                                       nan_policy='omit')[1]
        lt_pvalue = stats.mannwhitneyu(curr_pos[metric].to_numpy(),
                                       past_pos[metric].to_numpy(),
                                       alternative='less',
                                       nan_policy='omit')[1]
        if gt_pvalue < 0.05:
            high.append(metric)
        elif lt_pvalue < 0.05:
            low.append(metric)

    results = []
    #For each key, fit a model for n days
    for key in key_metrics:
        lag = 1
        for _ in range(5):
            past_pos[key] = past_pos[key].shift(-1)
            past_pos = past_pos.dropna()
            X_fit = past_pos[other_metrics]
            y_fit = past_pos[key]
            # Apply SelectKBest with chi2
            select_k_best = SelectKBest(score_func=f_regression, k=5)
            X_train_k_best = select_k_best.fit_transform(X_fit, y_fit.astype('float'))
            features = X_fit.columns[select_k_best.get_support()].tolist()
            #Filter to selected features and split into train/test
            X = X_fit[features].copy()
            X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=0.2, random_state=42)
            #Fit model and predict n days time
            lin = LinearRegression()
            lin.fit(X_train, y_train)
            y_pred = lin.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            prediction = lin.predict(X_pred[features])[0]

            #Work out if this is a high/low forecast
            gt_pvalue = stats.mannwhitneyu(prediction, y_fit, 
                                           alternative='greater',
                                           nan_policy='omit')[1]
            lt_pvalue = stats.mannwhitneyu( prediction, y_fit,
                                           alternative='less',
                                           nan_policy='omit')[1]
            if gt_pvalue < 0.05:
                high_or_low = 'high'
            elif lt_pvalue < 0.05:
                high_or_low = 'low'
            else:
                high_or_low = ''

            #Append results to list and increase lag
            results.append([key, lag,
                  (date.today() + relativedelta(days=+lag)).strftime('%Y-%m-%d'),
                  round(prediction, 2), high_or_low, features] + features)
            lag += 1

    forecasts = pd.DataFrame(results, columns=['Metric', 'Days Time', 'Date',
                                    'Prediction', 'High or Low', 'All Causes',
                                    'Cause 1',
                                    'Cause 2', 'Cause 3', 'Cause 4', 'Cause 5'])
    forecasts['High Causes'] = ((forecasts['All Causes']
                                .apply(lambda x: [i for i in x if i in high]))
                                .astype(str).str.replace('[' , '')
                                .str.replace(']', ''))
    forecasts['Low Causes'] = ((forecasts['All Causes']
                                .apply(lambda x: [i for i in x if i in low]))
                                .astype(str).str.replace('[' , '')
                                .str.replace(']', ''))
    return forecasts

def main():
    full_data, pivot, original, recent = GetData()
    outliers, recent_trend = OutliersAndRecent(original, recent)
    correlation = Correlations(original, recent)
    forecasts = Forecasts(pivot)
    metrics = pd.DataFrame(full_data['metric'].drop_duplicates().reset_index(drop=True))
    dates = pd.DataFrame({'Date': (full_data['dte'].drop_duplicates().tolist()
                                   + [date.today().strftime('%Y-%m-%d')]
                                   + forecasts['Date'].drop_duplicates().tolist())}).drop_duplicates()
    dates['Date'] = pd.to_datetime(dates['Date'])

    #add in period column for plots
    full_data['dte'] = full_data['dte'].astype(str)
    full_data['period'] = np.where(pd.to_datetime(full_data['dte'])
                                   < pd.to_datetime(date.today() - relativedelta(days=+7)),
                                   'Previous', 'Last 7 Days')
    
    return full_data, outliers, recent_trend, correlation, forecasts, metrics, dates
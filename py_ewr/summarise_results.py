from typing import List, Dict
from itertools import chain
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import dask.dataframe as dd
import dask.array as da

from . import data_inputs, evaluate_EWRs
#--------------------------------------------------------------------------------------------------
nproc = 14

def get_frequency(events: list) -> int:
    '''Returns the frequency of years they occur in.
    
    Args:
        events (list): a list of years with events (0 for no event, 1 for event)
    Results:
        int: Frequency of years with events
    
    '''
    if events.count() == 0:
        result = 0
    else:
        result = (int(events.sum())/int(events.count()))*100
    return int(round(result, 0))


def get_ewr_columns(ewr:str, cols:List) -> List:
    """Filter the columns of a particular ewr code in a list of 
    column names.

    Args:
        ewr (str): Ewr code
        cols (List): list of columns to search ewr pattern

    Returns:
        List: List of columns that matches the ewr code
    """
    return [c for c in cols if ewr in c]


def get_columns_attributes(cols: List)-> List:
    """Takes a list of columns with the pattern EwrCode_Attribute
    and relates them returning only the Attribute name.

    Args:
        cols (List): DataFrame columns names as a list

    Returns:
        List: List of the column attribute stripped out of the ewr code
    """
    return [c.split("_")[-1] for c  in cols]

def get_ewrs(pu_df: pd.DataFrame)-> List:
    """Take a DataFrame with the location results and by searching its 
    column return a list with the unique name of ewrs on the results.

    Args:
        pu_df (pd.DataFrame): DataFrame with with location stats results

    Returns:
        List: Returns a list os unique ewrs present in the location results
    """
    cols = pu_df.columns.to_list()
    ewrs_set = set(["_".join(c.split("_")[:-1]) for c  in cols])
    return list(ewrs_set)

def pu_dfs_to_process(detailed_results: Dict)-> List[Dict]:
    """Take the detailed results dictionary of the ewr calculation run,
    either observed or scenario and unpack items into a list of items.
    Each item is a dictionary with the following keys.
                { "scenario" : scenario_name,
                  "gauge" : gauge_id,
                  "pu" : pu_name,
                  "pu_df : DataFrame}

    Args:
        detailed_results (Dict): Dictionary with the following structure
        { "scenario_name/or observed": {"gaugeID": {"pu_name": pu_DateFrame}
        
            }

        } 
        It packs in a dictionary all the gauge ewr calculation for the scenario
        or observed dates run.

    Returns:
        List[Dict]: list of dict with the items to be processed
    """
    items_to_process = []
    for scenario in detailed_results:
        for gauge in detailed_results[scenario]:
            for pu in detailed_results[scenario][gauge]:
                item = {}
                item["scenario"] = scenario
                item["gauge"] = gauge
                item["pu"] = pu
                item["pu_df"] = detailed_results[scenario][gauge][pu]
                items_to_process.append(item)
    return items_to_process


def process_df(scenario:str, gauge:str, pu:str, pu_df: pd.DataFrame)-> pd.DataFrame:
    """Process all the pu_dfs into a tidy format

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        pu_df (pd.DataFrame): DataFrame to be transformed

    Returns:
        pd.DataFrame: DataFrame with all processed pu_dfs into one.
    """
    ewrs = get_ewrs(pu_df)
    returned_dfs = []
    for ewr in ewrs:
        columns_ewr = get_ewr_columns(ewr, pu_df.columns.to_list())
        ewr_df = pu_df[columns_ewr]
        column_attributes = get_columns_attributes(ewr_df.columns.to_list())
        ewr_df.columns = column_attributes
        ewr_df = ewr_df.reset_index().rename(columns={"index":'Year'})
        ewr_df["ewrCode"] = ewr
        ewr_df["scenario"] = scenario
        ewr_df["gauge"] = gauge
        ewr_df["pu"] = pu
        ewr_df = ewr_df.loc[:,~ewr_df.columns.duplicated()]
        returned_dfs.append(ewr_df)
    #return pd.concat(returned_dfs, ignore_index=True)
    return dd.concat(returned_dfs)


def process_df_results(results_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing and concatenating the processed dfs into one
    single dataframe with the results of all ewr calculations.

    Args:
        results_to_process (List[Dict]): List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr results
    """
    returned_dfs = []
    for item in results_to_process:
        try:
            transformed_df = process_df(**item)
            returned_dfs.append(transformed_df)
        except Exception as e:
            print(f"Could not process due to {e}")
    return dd.concat(returned_dfs)

def get_events_to_process(gauge_events: dict)-> List:
    """Take the detailed gauge events results dictionary of the ewr calculation run,
    and unpack items into a list of items.
    Each item is a dictionary with the following keys.
                { "scenario" : scenario_name,
                  "gauge" : gauge_id,
                  "pu" : pu_name,
                  "ewr": ewr_code
                  "ewr_events" : yearly_events_dictionary}

    Args:
        gauge_events (dict): Gauge events captured by the ewr calculations.
        Dictionary with the following structure
        {'observed': {'419001': {'Keepit to Boggabri': {'CF1_a': ({2010: [],
                2011: [],
                2012: [],
                2013: [],
                2014: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},)}
                                }
                     }
        }
        It packs in a dictionary all the gauge ewr yearly events and threshold flows.

    Returns:
        List: list of dict with the items to be processed
    """
    items_to_process = []
    for scenario in gauge_events:
        for gauge in gauge_events[scenario]:
            for pu in gauge_events[scenario][gauge]:
                for ewr in gauge_events[scenario][gauge][pu]:
                    try:
                        item = {}
                        item["scenario"] = scenario
                        item["gauge"] = gauge
                        item["pu"] = pu
                        item["ewr"] = ewr
                        item["ewr_events"],  = gauge_events[scenario][gauge][pu][ewr]
                        items_to_process.append(item)
                    except Exception as e:
                        print(f"fail to process events for {scenario}-{pu}-{ewr}-{gauge} with error {e}")
                        continue
    return items_to_process


def count_events(yearly_events:dict)-> int:
    """count the events in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        int: count of length of all events in the collection of years
    """
    return sum([len(events) for events in yearly_events.values()])


def sum_events(yearly_events:dict)-> int:
    """sum the total event days in a collection of years

    Args:
        yearly_events (dict): ewr yearly events dictionary of lists of lists

    Returns:
        int: count of total days of all events in the collection of years
    """

    flattened_events = [list(chain(*events)) for events in yearly_events.values()]
    return len(list(chain(*flattened_events)))


def process_yearly_events(scenario:str, gauge:str, pu:str, ewr:str, ewr_events: Dict)-> pd.DataFrame:
    """process each item for the gauge and return the statistics in a DataFrame

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        ewr (str): DataFrame to be transformed
        ewr_events (Dict): Dict with all yearly events list with date and flow/level 

    Returns:
        pd.DataFrame: DataFrame with events statistics
    """
    row_data = defaultdict(list)
    yearly_events = ewr_events
    total_events = count_events(yearly_events)
    total_event_days = sum_events(yearly_events)
    average_event_length = total_event_days/total_events if total_events else 0
    row_data['scenario'].append(scenario)
    row_data['gauge'].append(gauge)
    row_data['pu'].append(pu)
    row_data['ewrCode'].append(ewr)
    row_data['totalEvents'].append(total_events)
    row_data['totalEventDays'].append(total_event_days)
    row_data['averageEventLength'].append(average_event_length)
    
    return pd.DataFrame(row_data)

def process_ewr_events_stats(events_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing of yearly events and concatenate into a
    single dataframe with the results of all ewr calculations.

    Args:
        events_to_process (List[Dict]): List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr events stats results
    """
    returned_dfs = []
    for item in events_to_process:
        row_data = process_yearly_events(**item)
        returned_dfs.append(row_data)
    return dd.concat(returned_dfs)

def process_all_yearly_events(scenario:str, gauge:str, pu:str, ewr:str, ewr_events: Dict)-> pd.DataFrame():
    """process each item for the gauge and return all events. Each event is a row with a start and end date
    duration and event length

    Args:
        scenario (str): scenario name metadata
        gauge (str): gauge name metadata
        pu (str): planning unit name metadata
        ewr (str): DataFrame to be transformed
        ewr_events (Dict): Dict with all yearly events list with date and flow/level
        

    Returns:
        pd.DataFrame: DataFrame with all events of Pu-ewr-gauge combination
    """
    df_data = defaultdict(list)
    for year in ewr_events:
        for i, ev in enumerate(ewr_events[year]):
            start_date, _ = ev[0]
            end_date, _ = ev[-1]
            df_data["scenario"].append(scenario)
            df_data["gauge"].append(gauge)
            df_data["pu"].append(pu)
            df_data["ewr"].append(ewr)
            df_data["waterYear"].append(year)
            df_data["startDate"].append(start_date )
            df_data["endDate"].append(end_date)
            df_data["eventDuration"].append((end_date - start_date).days + 1)
            df_data["eventLength"].append(len(ev))          
    
    return pd.DataFrame(df_data)

def process_all_events_results(results_to_process: List[Dict])-> pd.DataFrame:
    """Manage the processing of yearly events and concatenate into a
    single dataframe with the results of all ewr calculations.

    Args:
        results_to_process (List[Dict]):List with all items to process.

    Returns:
        pd.DataFrame: Single DataFrame with all the ewr events
    """
    returned_dfs = []
    for item in results_to_process:
        try:
            df = process_all_yearly_events(**item)
            returned_dfs.append(df)
        except Exception as e:
            print(f"could not process due to {e}")
            continue
    return dd.concat(returned_dfs).compute()

def fill_empty(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x if x != '' else '0')
    return df

def join_ewr_parameters(cols_to_add:List, left_table:pd.DataFrame, left_on:List, 
                        selected_columns:List = None, renamed_columns:List = None,
                        parameter_sheet_path:str = None)-> pd.DataFrame:
    
    EWR_table, bad_EWRs = data_inputs.get_EWR_table(parameter_sheet_path)

    EWR_table = fill_empty(EWR_table, ['TargetFrequency','MaxInter-event'])

    columns_right_table = ['Gauge','PlanningUnitName','Code']

    columns_right_table += cols_to_add

    EWR_table = EWR_table[columns_right_table]
    
    output_table = left_table.merge(EWR_table, 
                                                  'left',
                                                  left_on=left_on, 
                                                  right_on=['Gauge','PlanningUnitName','Code'])

    if selected_columns:
        output_table = output_table[selected_columns]

    if renamed_columns:    
        output_table.columns = renamed_columns

    return output_table

def sum_0(series:pd.Series) -> int:
    '''
    Custom agg function for counting occurences of 0's in a series
    
    Args:
        series (pd.Series): pandas series of 0s and 1s

    Results:
        int: sum of 0s
    '''
    return series[series==0].count()

    # return series.value_counts()[0]


def summarise(input_dict:Dict , events:Dict, parameter_sheet_path:str = None)-> pd.DataFrame:
    """orchestrate the processing of the pu_dfs items and the gauge events and join
    in one summary DataFrame and join with EWR parameters for comparison

    Args:
        input_dict (Dict): DataFrame result by yearly with statistics for the ewr calculations.
        events (Dict): Gauge events captured by the ewr calculations.

    Returns:
        pd.DataFrame: Summary statistics for all ewr calculation for the whole period of the run
    """
    print(str(datetime.now()) + " starting ewr results summarise")
    to_process = pu_dfs_to_process(input_dict)
    yearly_ewr_results = process_df_results(to_process)
    yearly_ewr_results = yearly_ewr_results.compute()
    print(str(datetime.now()) + " dfs concatenated")
    
    # aggregate by "gauge","pu","ewrCode"
    final_summary_output = (yearly_ewr_results
    .groupby(["scenario","gauge","pu","ewrCode"])
    .agg( EventYears = ("eventYears", sum),
          Frequency = ("eventYears", get_frequency),
          AchievementCount = ("numAchieved", sum),
          AchievementPerYear = ("numAchieved", 'mean'),
          EventCount = ("numEvents",sum),
          EventCountAll = ("numEventsAll",sum),
          EventsPerYear = ("numEvents",'mean'),
          EventsPerYearAll = ("numEventsAll",'mean'),
          ThresholdDays = ("totalEventDays", sum),
        #   InterEventExceedingCount = ("rollingMaxInterEventAchieved", sum_0),#"maxInterEventDaysAchieved"
          NoDataDays =  ("missingDays" , sum),
          TotalDays = ("totalPossibleDays" , sum),
          )
    )
    print(str(datetime.now()) + " groupby done")
    # summarize gauge events
    
    events_to_process = get_events_to_process(events)
    ewr_event_stats = process_ewr_events_stats(events_to_process)
    ewr_event_stats = ewr_event_stats.compute()
    print(str(datetime.now()) + " event dfs concatenated")
    
    # join summary with gauge events
    
    final_summary_output = final_summary_output.merge(ewr_event_stats, 
                                                      'left',
                                                      left_on=['scenario', 'gauge','pu','ewrCode'], 
                                                      right_on=['scenario', 'gauge','pu',"ewrCode"])
    print(str(datetime.now()) + " summary/gauge merge done")
    # Join Ewr parameter to summary

    final_merged = join_ewr_parameters(cols_to_add=['TargetFrequency','MaxInter-event','Multigauge'],
                                left_table=final_summary_output,
                                left_on=['gauge','pu','ewrCode'],
                                selected_columns=["scenario",'gauge',
                                                    'pu', 
                                                    'ewrCode',
                                                    'Multigauge',
                                                    'EventYears',
                                                    'Frequency',
                                                    'TargetFrequency',
                                                    'AchievementCount',
                                                    'AchievementPerYear',
                                                    'EventCount',
                                                    'EventCountAll',
                                                    'EventsPerYear',
                                                    'EventsPerYearAll',
                                                    'averageEventLength',
                                                    'ThresholdDays',
                                                    # 'InterEventExceedingCount',
                                                    'MaxInter-event',
                                                    'NoDataDays',
                                                    'TotalDays'],
                                renamed_columns=['Scenario','Gauge', 'PlanningUnit', 'EwrCode', 'Multigauge','EventYears', 'Frequency', 'TargetFrequency',
                                    'AchievementCount', 'AchievementPerYear', 'EventCount', 'EventCountAll','EventsPerYear', 'EventsPerYearAll',
                                    'AverageEventLength', 'ThresholdDays', #'InterEventExceedingCount',
                                    'MaxInterEventYears', 'NoDataDays', 'TotalDays'],
                                    parameter_sheet_path=parameter_sheet_path)

    print(str(datetime.now()) + " final merge done")
    
    return final_merged

def filter_duplicate_start_dates(df: pd.DataFrame) -> pd.DataFrame:
    '''
    For those events that are recorded on a rolling basis at the end of the year - remove the duplicates.
    TODO: Make this filtering process more robust. Currently its just keeping the last one because
    this will be the longest event, but if for some reason the dataframe is reordered this will be
    tripped up.

    Args:
        events (pd.DataFrame): all events dataframe

    Results:
        pd.DataFrame: Updated all_events dataframe with duplicates removed

    '''

    df.drop_duplicates(subset = ['scenario', 'gauge', 'pu', 'ewr', 'startDate'], keep='last', inplace=True)

    return df


def events_to_interevents(df_events: pd.DataFrame, scenarios={}) -> pd.DataFrame:

    df_events = df_events.drop(['eventDuration', 'eventLength', 'Multigauge'], axis=1)

    def get_start_date(row):
        y = row['scenario'].values[0]
        # Get start and end date form scenario df
        flow_data = scenarios[y]
        date0 = flow_data.index[0]

        start_date = date(date0.year, date0.month, date0.day)

        return start_date

    def get_end_date(row):
        y = row['scenario'].values[0]
        # Get start and end date form scenario df
        flow_data = scenarios[y]
        date1 = flow_data.index[-1]

        end_date = date(date1.year, date1.month, date1.day)

        return end_date

        # get the scenario start and end dates

    df_starts = df_events.groupby(['scenario', 'gauge', 'pu', 'ewr']).apply(lambda row: get_start_date(row))
    df_ends = df_events.groupby(['scenario', 'gauge', 'pu', 'ewr']).apply(lambda row: get_end_date(row))

    df_ends.name = "endDate"
    df_starts.name = "startDate"

    df_ends = df_ends.to_frame().reset_index()
    df_starts = df_starts.to_frame().reset_index()

    scenario_start_end = df_starts.merge(df_ends, on=['scenario', 'gauge', 'pu', 'ewr'])

    ewr_list = []

    # split df into groups,
    for index, row in scenario_start_end.iterrows():
        # select data, convert to dask array
        data = df_events[((df_events['scenario'] == row['scenario'])
                          & (df_events['gauge'] == row['gauge'])
                          & (df_events['pu'] == row['pu'])
                          & (df_events['ewr'] == row['ewr']))].to_numpy()

        # get scenario start and end date
        scenario_start = scenario_start_end[((scenario_start_end['scenario'] == data[0, [0]][0])
                                             & (scenario_start_end['gauge'] == data[0, [1]][0])
                                             & (scenario_start_end['pu'] == data[0, [2]][0])
                                             & (scenario_start_end['ewr'] == data[0, [3]][0]))]['startDate'].values[0]

        scenario_end = scenario_start_end[((scenario_start_end['scenario'] == data[0, [0]][0])
                                           & (scenario_start_end['gauge'] == data[0, [1]][0])
                                           & (scenario_start_end['pu'] == data[0, [2]][0])
                                           & (scenario_start_end['ewr'] == data[0, [3]][0]))]['endDate'].values[0]

        start_list = [data[0, [0]][0], data[0, [1]][0], data[0, [2]][0], data[0, [3]][0], data[0, [4]][0], np.NaN,
                      scenario_start]
        end_list = [data[0, [0]][0], data[0, [1]][0], data[0, [2]][0], data[0, [3]][0], data[0, [4]][0], scenario_end,
                    np.NaN]

        # offset dates + 1 day for start dates, -1 day for end dates
        # new end dates
        data[:, [5]] = np.subtract(data[:, [5]], timedelta(days=1))
        # new start dates
        data[:, [6]] = np.add(data[:, [6]], timedelta(days=1))

        # insert new end at top, new starts at bottom
        data = np.vstack(([start_list], data, [end_list]))

        # shift new end date down 1
        data[:, [6]] = np.roll(data[:, [6]], 1)

        # drop row with NaN
        data = np.delete(data, 0, 0)

        # calculate inter event lenght
        inter_event = np.subtract(data[:, [5]], data[:, [6]])

        # add day to inter event lenght
        inter_event = np.add(inter_event, timedelta(days=1))

        # merge inter event into rest of data
        data = np.hstack((data, inter_event))

        ewr_list.append(data)

    all_interEvents = da.vstack(tuple(ewr_list))

    all_interEvents = dd.from_dask_array(all_interEvents, columns=['scenario', 'gauge', 'pu', 'ewr', 'wateryear',
                                                                   'endDate', 'startDate', 'interEventLength'])

    all_interEvents = all_interEvents.compute()
    # after conversion apply conversion for intereventlength to int
    all_interEvents['interEventLength'] = all_interEvents['interEventLength'].dt.days

    # after conversion back to df - drop 0 lenght inter events
    all_interEvents = all_interEvents[all_interEvents['interEventLength'] != 0]

    return all_interEvents


def filter_successful_events(all_events: pd.DataFrame) -> pd.DataFrame:
    '''
    Filters out unsuccessful events, returns successful events - those meeting min spell

    Args:
        all_events (pd.DataFrame): dataframe with events

    Returns:
        pd.DataFrame: Dataframe with only successful events
    
    '''

    all_events = dd.from_pandas(all_events, npartitions=nproc)
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()

    def get_min_spell(row):
        minSpell = int(data_inputs.ewr_parameter_grabber(EWR_table, row['gauge'], row['pu'], row['ewr'], 'MinSpell'))
        return minSpell

    min_spell = all_events.apply(lambda row: get_min_spell(row), axis=1, meta=pd.Series(dtype="int64"))
    min_spell.name = "MinSpell"

    # join min spell data to all_events
    all_events = all_events.join(min_spell)

    # Filter out the events that fall under the minimum spell length
    all_successfulEvents = all_events[all_events['eventDuration'] >= all_events['MinSpell']]
    all_successfulEvents = all_successfulEvents.drop('MinSpell', axis=1)

    return all_successfulEvents.compute()

def get_rolling_max_interEvents(df:pd.DataFrame, start_date: date, end_date: date, yearly_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Determines the rolling maximum interevent period for each year.
    Args:
        yearly_df (pd.DataFrame): used to get list of all EWRs.
    Results:
        pd.DataFrame: 
    
    '''

    s = 'TEMPORARY_ID_SPLIT'

    df['ID'] = df['scenario']+s+df['gauge']+s+df['pu']+s+df['ewr']
    yearly_df['ID'] = yearly_df['scenario']+s+yearly_df['gauge']+s+yearly_df['pu']+s+yearly_df['ewrCode']
    unique_ID = list(OrderedDict.fromkeys(yearly_df['ID']))
    master_dict = dict()
    unique_years = list(range(min(yearly_df['Year']),max(yearly_df['Year'])+1,1))
    # Load in EWR table to variable to access start and end dates of the EWR
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()
    for unique_EWR in unique_ID:
        df_subset = df[df['ID'].str.fullmatch(unique_EWR)]
        yearly_df_subset = yearly_df[yearly_df['ID'].str.fullmatch(unique_EWR)]
        # Get EWR characteristics for current EWR
        scenario = unique_EWR.split('TEMPORARY_ID_SPLIT')[0]
        gauge = unique_EWR.split('TEMPORARY_ID_SPLIT')[1]
        pu = unique_EWR.split('TEMPORARY_ID_SPLIT')[2]
        ewr = unique_EWR.split('TEMPORARY_ID_SPLIT')[3]

        # if merged ewr skip
        if '/' in ewr:
            continue
        
        # Construct dictionary to save results to:
        if scenario not in master_dict:
            master_dict[scenario] = {}
        if gauge not in master_dict[scenario]:
            master_dict[scenario][gauge] = {}
        if pu not in master_dict[scenario][gauge]:
            master_dict[scenario][gauge][pu] = {}
        if ewr not in master_dict[scenario][gauge][pu]:
            master_dict[scenario][gauge][pu][ewr] = evaluate_EWRs.construct_event_dict(unique_years)
        # Pull EWR start and end date from EWR dataset and clean TODO: functionalise this
        EWR_info = {}
        EWR_info['start_date'] = data_inputs.ewr_parameter_grabber(EWR_table, gauge, pu, ewr, 'StartMonth')
        EWR_info['end_date'] = data_inputs.ewr_parameter_grabber(EWR_table, gauge, pu, ewr, 'EndMonth')
        if '.' in EWR_info['start_date']:
            EWR_info['start_day'] = int(EWR_info['start_date'].split('.')[1])
            EWR_info['start_month'] = int(EWR_info['start_date'].split('.')[0])
        else:
            EWR_info['start_day'] = None
            EWR_info['start_month'] = int(EWR_info['start_date'])

        if '.' in EWR_info['end_date']:  
            EWR_info['end_day'] = int(EWR_info['end_date'].split('.')[1])
            EWR_info['end_month'] = int(EWR_info['end_date'].split('.')[0])
        else:
            EWR_info['end_day'] = None
            EWR_info['end_month'] =int(EWR_info['end_date'])        
        # if ewr == "LF2_WP":
        # if unique_EWR == "big10602.bmdTEMPORARY_ID_SPLIT425010TEMPORARY_ID_SPLITMurray River - Lock 10 to Lock 9TEMPORARY_ID_SPLITLF2_WP":

        # Iterate over the interevent periods for this EWR
        for i, row in df_subset.iterrows():
            # Get the date range:
            period = pd.period_range(df_subset.loc[i, 'startDate'],df_subset.loc[i, 'endDate'])
            # Save to pd.df for function compatibility
            dates_df = pd.DataFrame(index = period)
            # Convert year to water year using the existing function            
            period_wy = evaluate_EWRs.wateryear_daily(dates_df, EWR_info)
            # Iterate over the years:
            for YEAR in period_wy:
                master_dict[scenario][gauge][pu][ewr][YEAR].append(np.sum(period_wy<=YEAR))
        # Iterate over the water years, keep only the maximum values from each year:
        for yr, interevents in master_dict[scenario][gauge][pu][ewr].items():
            master_dict[scenario][gauge][pu][ewr].update({yr: max(interevents, default=0)})
    
    df.drop(['ID'], axis=1, inplace=True)
    yearly_df.drop(['ID'], axis=1, inplace=True)

    return master_dict

def add_interevent_to_yearly_results(yearly_df: pd.DataFrame, yearly_dict:Dict) -> pd.DataFrame:
    '''
    Adds a column to the yearly results summary with the maximum rolling interevent period.

    Args:
        yearly_df (pd.DataFrame): Yearly results dataframe summary
        yearly_dict (dict): Rolling maximum annual interevent period for every EWR
    Returns:
        pd.DataFrame: Yearly results dataframe summary with the new column
    '''
    yearly_df['rollingMaxInterEvent'] = None
    # iterate yearly df, but ignore merged ewrs
    for i, row in yearly_df[~yearly_df['ewrCode'].str.contains('/', regex=False)].iterrows():
        scenario = yearly_df.loc[i, 'scenario']
        gauge = yearly_df.loc[i, 'gauge']
        pu = yearly_df.loc[i, 'pu']
        ewr = yearly_df.loc[i, 'ewrCode']
        year = yearly_df.loc[i, 'Year']
        value_to_add = yearly_dict[scenario][gauge][pu][ewr][year]
        yearly_df.loc[i, 'rollingMaxInterEvent'] = value_to_add
    
    return yearly_df

def add_interevent_check_to_yearly_results(yearly_df: pd.DataFrame) -> pd.DataFrame:
    '''
    For each EWR, check to see if the rolling max interevent achieves the minimum requirement.

    Args:
        yearly_df (pd.DataFrame): 
    Results:
        pd.DataFrame: yearly_ewr_results dataframe with the new column
    
    '''

    yearly_df['rollingMaxInterEventAchieved'] = None

    # Load in EWR table to variable to access start and end dates of the EWR
    EWR_table, bad_EWRs = data_inputs.get_EWR_table()

    # Get EWR characteristics for current EWR
    for i, row in yearly_df.iterrows():
        gauge = yearly_df.loc[i, 'gauge']
        pu = yearly_df.loc[i, 'pu']
        ewr = yearly_df.loc[i, 'ewrCode']

        if '/' in ewr:
            yearly_df.loc[i, 'rollingMaxInterEventAchieved'] = None
            continue

        max_interevent_target = int(float(data_inputs.ewr_parameter_grabber(EWR_table, gauge, pu, ewr, 'MaxInter-event'))*365)
        
        interevent_value = yearly_df.loc[i, 'rollingMaxInterEvent']
        
        if interevent_value > max_interevent_target:
            result = 0
        else:
            result = 1
        yearly_df.loc[i, 'rollingMaxInterEventAchieved'] = result
    
    return yearly_df



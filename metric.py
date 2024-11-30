import numpy as np
import pandas as pd
import json

APOLOGETIC_SUBSTRING = ["sorry",  
                     "apologize", 
                     "cannot", 
                     "could not",
                     "couldn't",
                     "can’t", 
                     "unable",
                     "unfortunately",
                     "not successful",
                     "lacks"]

API_LIST = ["GorillaFileSystem", "MathAPI", "MessageAPI", 
            "TwitterAPI", "TicketAPI", "TradingBot", 
            "TravelAPI", "VehicleControlAPI"]

def average_num_steps(df):
    """
    Calculate the mean number of steps across each turn.
    """
    return pd.Series(df["turn_responses"]).apply(lambda x: x["num_steps"]).mean()

def average_num_tools(df):
    """
    Calculate the mean number of tools used across each turn, ignoring the last step message in each turn.
    """
    turn_responses = df["turn_responses"]
    num_tools = []
    for turn_response in turn_responses:
        turn_num_tools = [step_response["num_tools"] for step_response in turn_response["step_responses"][:-1]]
        if len(turn_num_tools) > 0:
            num_tools.append(np.mean(turn_num_tools))
    return np.mean(num_tools)

def apologetic(df):
    """
    Determine if the model returned an unsuccessful execution message in any turn.
    Checked by verifying whether the final step response (summary) in any turn contains any of the APOLOGETIC_SUBSTRINGS.
    """ 
    turn_responses = df["turn_responses"]
    for turn_response in turn_responses:
        turn_summary = str(turn_response["step_responses"][-1]['assistant_response']['content'])
        if turn_summary is None or any([phrase.lower() in turn_summary.lower() for phrase in APOLOGETIC_SUBSTRING]):
            return True
    return False    

def apologetic_rate(df):
    """
    Returns the proportion of turns containing apologetic messages.
    """ 
    apologetic_turn_count = 0
    turn_responses = df["turn_responses"]
    for turn_response in turn_responses:
        turn_summary = str(turn_response["step_responses"][-1]['assistant_response']['content'])
        if turn_summary is None or any([phrase.lower() in turn_summary.lower() for phrase in APOLOGETIC_SUBSTRING]):
            apologetic_turn_count += 1
    return apologetic_turn_count / len(turn_responses)

def no_func_call_rate(df):
    """
    Returns the proportion of turns with no function calls.
    """
    turn_responses = df["turn_responses"]
    no_func_call_count = 0
    for turn_idx in range(len(turn_responses)):
        for entry in df['turn_responses'][turn_idx]['step_responses']:
            if 'handler_response' in entry.keys():
                break
            else:
                no_func_call_count += 1
    return no_func_call_count / len(turn_responses)

def num_tool_errors(df):
    """
    Return the total number of tool errors across all turns.
    """
    turn_responses = df["turn_responses"]
    num_errors = 0
    for turn_idx in range(len(turn_responses)):
        for step in df['turn_responses'][turn_idx]['step_responses']:
            for tool_response in step['tool_response']:
                try:
                    if "error" in json.loads(tool_response['content']).keys():
                        num_errors += 1
                except:
                    pass
    return num_errors

def num_turn_with_errors(df):
    """
    Returns the number of turns with errors.
    """
    def has_error_in_tool_response(tool_response):
        try:
            return "error" in json.loads(tool_response['content']).keys()
        except:
            return False
    turn_responses = df["turn_responses"]
    num_turns_with_error = 0
    for turn_idx in range(len(turn_responses)):
        if any(
            has_error_in_tool_response(tool_response)
            for step in df['turn_responses'][turn_idx]['step_responses'] 
            for tool_response in step['tool_response'] 
        ):
            num_turns_with_error += 1
    return num_turns_with_error

def check_api_state_mismatch(df):
    """
    For each API not present, returns np.nan.
    For each API present, returns False if its states match across all turns; otherwise, returns True (a mismatch exists).
    """
    state = {api: np.nan for api in API_LIST}
    api_list = [entry['class_name'] for entry in df['turn_responses'][0]['end_of_turn_state']]
    for api in api_list:
        state[api] = False
    if len(df['ground_truth_log']) != len(df['turn_responses']) + 1:
        for api in api_list:
            state[api] = True
    else:
        for turn_idx, turn_response in enumerate(df['turn_responses']):
            gt_states = df['ground_truth_log'][turn_idx + 1]
            turn_states = turn_response['end_of_turn_state']
            for gt_state, turn_state in zip(gt_states, turn_states):
                if gt_state != turn_state:
                    state[gt_state['class_name']] = True
    return pd.Series(state.values())

def force_terminated(df):
    """
    Returns True if force_terminated
    """
    if "multi_turn:force_terminated" in df['error_type']:
        return True
    return False

def state_inconsistent(df):
    """
    Returns True if model state is inconsistent with ground truth state
    """
    if "multi_turn:instance_state_mismatch" in df['error_type']:
        return True
    return False

def response_inconsistent(df):
    """
    Returns True if response_inconsistent
    """
    if "multi_turn:execution_response_mismatch" in df['error_type']:
        return True
    return False

def task_process_rate(df):
    """
    Returns the index of the earliest turn with an error divided by the total number of turns.  
    Measures how early the model makes mistakes.
    """
    earlier_error_turn_index = 0
    for turn_idx in range(len(df['turn_responses'])):
        for pred, truth in zip(df['turn_responses'][turn_idx]['end_of_turn_state'], df['ground_truth_log'][turn_idx+1]):
            if pred != truth:
                return turn_idx / len(df['turn_responses'])
    return float("inf") 

def average_turn_success_rate(df):
    """
    Calculates the average turn success rate, which is the proportion of turns where 
    the predicted end-of-turn state matches the ground truth state.
    """
    score = 0
    for turn_idx in range(len(df['turn_responses'])):
        if all(
            pred == truth 
            for pred, truth in zip(df['turn_responses'][turn_idx]['end_of_turn_state'], df['ground_truth_log'][turn_idx + 1])
        ):
            score += 1 
    return score / len(df['turn_responses'])

def soft_average_turn_success_rate(df):
    """
    Calculates a modified average turn success rate, where the score for each correct turn is adjusted
    based on the proximity of the last incorrect turn. The further a correct turn is from the last incorrect one,
    the lower its contribution to the score.
    """
    score = 0
    last_incorrect_turn = float("inf")
    for turn_idx in range(len(df['turn_responses'])):
        if all(
            pred == truth 
            for pred, truth in zip(df['turn_responses'][turn_idx]['end_of_turn_state'], df['ground_truth_log'][turn_idx + 1])
        ):
            if turn_idx < last_incorrect_turn:
                score += 1
            else:
                score += 1 - np.exp(-(turn_idx - last_incorrect_turn))
        else:
            last_incorrect_turn = turn_idx
    return score / len(df['turn_responses'])
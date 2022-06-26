import numpy as np
import pandas as pd  
from tqdm import tqdm, trange 

import requests 
import json 
import time
from collections.abc import Iterable
from datetime import datetime

def get_auth_keys(file_path: str, 
    api_key_key: str = "api_key", 
    api_key_secret_key: str = "api_key_secret", 
    bearer_token_key: str = "bearer_token") -> (str, str, str):
    """
    Returns the authentication keys for the Twitter API 
    from a json file. Returns the api_key, api_key_secret, and 
    the bearer_token. 

    Expected json format: 
    {
        "api_key": "", 
        "api_key_secret": "", 
        "bearer_token": ""
    }

    parameters:
    ----------
    file_path - relative or absolute file path to json file with auth tokens
    api_key_key - key for the api_key in the json file
    api_key_secret_key - key for the api_key_secret in the json file
    bearer_token_key - key for the bearer_token in the json file
    """
    with open(file_path, "r") as f: 
        auth = json.load(f)

        api_key = auth['api_key']
        api_key_secret = auth['api_key_secret']
        bearer_token = auth['bearer_token']
    
    return (api_key, api_key_secret, bearer_token)

def create_param_dict(
        expansions: str = "author_id,in_reply_to_user_id,geo.place_id", 
        tweet_fields: str = "id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source", 
        user_fields: str = "id,name,username,created_at,description,public_metrics,verified", 
        place_fields: str = "full_name,id,country,country_code,geo,name,place_type"
        ) -> dict:
    """
    Creates the parameter dictionary for the Twitter API to simplify 
    request process. Currently only supports getting tweets form 
    their id. See 

    parameters:
    ----------
    expansions: https://developer.twitter.com/en/docs/twitter-api/expansions
    tweet_fields: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
    user_fields: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object
    place_fields: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/place-object 
    """ 
    params = {
        'expansions': expansions,
        'tweet.fields': tweet_fields,
        'user.fields': user_fields,
        'place.fields': place_fields  
    } #params

    return params

def create_header_dict(bearer_token: str) -> dict:
    """
    Creates the headers for the Twitter API request. Will not 
    check if your bearer token is valid.

    parameters:
    ----------
    bearer_token: str
    """
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers

def get_tweet_from_id(tweet_ids: str,
    bearer_token: str,
    params: dict,
    headers: dict) -> requests.models.Response:
    """
    Returns a tweet from the Twitter API given a tweet id. If a tweet does not 
    exist it will return an error. Does not account for rate limits, check response 
    headers. 

    parameters:
    ----------
    tweet_id - id of the tweet to retrieve 
    bearer_token - bearer token for the Twitter API
    params - parameter dictionary for the Twitter API
    headers - headers for the Twitter API
    """
    end_point = "https://api.twitter.com/2/tweets/"
    params["ids"] = tweet_ids
    response = requests.get(end_point, headers=headers, params=params)

    return response

def check_rate_limit(response: requests.models.Response) -> dict:
    """
    Checks the current status of the rate limit. Returns two values from 
    the response header: 
        1. "x-rate-limit-remaining" - number of requests remaining
        2. "x-rate-limit-reset" - unix timestamp until the rate limit resets

    parameters:
    ----------
    response - response object from the Twitter API

    returns:
    ----------
    dictionary with two keys:
        remaining:int - number of requests remaining
        reset_time:int - unix timestamp until the rate limit resets
    """
    remaining = response.headers["x-rate-limit-remaining"]
    reset = response.headers["x-rate-limit-reset"]

    rate_limit = {
        "remaining": remaining, 
        "reset_time": reset}

    return rate_limit

def get_tweets_from_id(tweet_ids: Iterable, 
    bearer_token: str,
    params: dict,
    headers: dict,
    save_folder: str = "../../raw_tweets/"): 
    """
    Retrieves tweets from the Twitter API given a list of tweet ids. Keeps 
    track of rate_limit and saves the raw response json to a file in the 
    specified save_folder. 

    Twitter API allows for up to 100 tweets to be retrieved at a time.

    parameters:
    ----------
    tweet_ids: iterable - list of tweet ids to retrieve
    save_folder: str - folder to save the raw json response
    """
    num_batches = len(tweet_ids) // 100 + 1
    for i in trange(num_batches):
        batch_ids = tweet_ids[i*100:(i+1)*100]
        id_string = ",".join([str(id) for id in batch_ids])

        response = get_tweet_from_id(id_string, bearer_token, params, headers)
        rate_limit = check_rate_limit(response)

        if rate_limit["remaining"] == str(0):
            print(f"Timed out, sleeping until {datetime.fromtimestamp(int(rate_limit['reset_time']))}")
            time.sleep(float(rate_limit["reset_time"]) - time.time() + 10)
            response = get_tweet_from_id(id_string, bearer_token, params, headers)
        
        with open(save_folder + f"batch{i}" + ".json", "w") as dump_file:
            json.dump(response.json(), dump_file, indent=4)
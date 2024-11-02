# -*- coding: utf-8 -*-
"""Copy of TDSproject1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ieep2bbZrfHIGLJtXQP7Mzmt8i_g2CwE
"""

# GIT_HUB_API is stored as secret in google colab!

from google.colab import userdata
api_key=userdata.get('GIT_HUB_API')

import requests
import csv


TOKEN = api_key
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"Bearer {TOKEN}"
}
def search_users(city):
    users = []
    page = 1
    while page <= 100:  # pagination because github shows only 30 results per page
        url = f"https://api.github.com/search/users?q=location:{city}+followers:>200&page={page}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            data = response.json()
            users.extend(data.get('items', []))
            if len(data.get('items', [])) < 30:  # Less than 30 means no more pages
                break
            page += 1
        else:
            print(f"Error fetching users: {response.status_code} - {response.text}") #error handling
            break
    return users


def get_user_details(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching user details for {username}: {response.status_code} - {response.text}")  #error handling
        return {}

def get_user_repos(username):
    repos = []
    page = 1
    while True:  # Continue until there are no more pages
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            data = response.json()
            repos.extend(data)

            # If less than 100 were returned, we've reached the last page
            if len(data) < 100:
                break
            page += 1
        else:
            print(f"Error fetching repositories for {username}: {response.status_code} - {response.text}")
            break

    return repos  # Return all collected repositories
     # Sort by 'pushed_at' and get the most recent 500
    sorted_repos = sorted(repos, key=lambda r: r.get('pushed_at', '1970-01-01'), reverse=True)
    return sorted_repos[:500]  # Limit to 500 most recently pushed repositories



city = "Paris"
users = search_users(city)

# Prepare the user CSV file
with open('github_usersPARIS.csv', mode='w', newline='', encoding='utf-8') as user_file:
    user_fieldnames = [
        'login', 'name', 'company', 'location', 'email',
        'hireable', 'bio', 'public_repos', 'followers',
        'following', 'created_at'
    ]
    user_writer = csv.DictWriter(user_file, fieldnames=user_fieldnames)
    user_writer.writeheader()  # Write the header row

    # Prepare the repository CSV file
    with open('github_reposPARIS.csv', mode='w', newline='', encoding='utf-8') as repo_file:
        repo_fieldnames = [
            'login', 'repo_full_name', 'repo_created_at',
            'stargazers_count', 'watchers_count', 'language',
            'has_projects', 'has_wiki', 'license_name'
        ]
        repo_writer = csv.DictWriter(repo_file, fieldnames=repo_fieldnames)
        repo_writer.writeheader()  # Write the header row

        for user in users: #gather user info
            username = user.get('login', 'Unknown User')
            user_details = get_user_details(username)

            # Clean up company names

            COM= user_details.get('company')
            if COM is not None:
              COM = user_details.get('company', '').strip().upper()
              if COM.startswith('@'):
                COM = COM[1:]
              else:
                COM = COM
            else:
              COM = ''
            user_data = {
                'login': user_details.get('login', ''),
                'name': user_details.get('name', ''),
                'company': COM,
                'location': user_details.get('location', ''),
                'email': user_details.get('email', ''),
                'hireable': 'true' if user_details.get('hireable', False) else 'false', #set hireable to true and false
                'bio': user_details.get('bio', ''),
                'public_repos': user_details.get('public_repos', 0),
                'followers': user_details.get('followers', 0),
                'following': user_details.get('following', 0),
                'created_at': user_details.get('created_at', '')
            }

            # Write user data to user CSV
            user_writer.writerow(user_data)

            # Fetch user repositories
            repos = get_user_repos(username)

            for repo in repos:
                # Gather repository info
                repo_data = {
                    'login': username,  # Include the user's login for reference
                    'repo_full_name': repo.get('full_name', ''),
                    'repo_created_at': repo.get('created_at', ''),
                    'stargazers_count': repo.get('stargazers_count', 0),
                    'watchers_count': repo.get('watchers_count', 0),
                    'language': repo.get('language', ''),
                    'has_projects': 'true' if repo.get('has_projects', False) else 'false',
                    'has_wiki': 'true' if repo.get('has_wiki', False) else 'false',
                    'license_name': repo.get('license')['name'] if repo.get('license') else 'NONE'

                }

                # Write repository data to repo CSV
                repo_writer.writerow(repo_data)
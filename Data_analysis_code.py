import pandas as pd
import statsmodels.api as sm

# Load the data from CSV files
users_df = pd.read_csv('github_usersPARIS.csv')  # User data
repos_df = pd.read_csv('github_reposPARIS.csv')  # Repository data

# Group by language and calculate the average stars
average_stars = repos_df.groupby('language')['stargazers_count'].mean()

# Identify the language with the highest average stars
highest_avg_language = average_stars.idxmax()
highest_avg_value = average_stars.max()

print(f"The language with the highest average number of stars per repository is '{highest_avg_language}' with an average of {highest_avg_value:.2f} stars.")


# Calculate leader strength
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])

# Sort by leader strength in descending order and get the top 5
top_leaders = users_df.sort_values(by='leader_strength', ascending=False).head(5)

# Get the logins of the top 5 leaders, comma-separated
top_logins = ', '.join(top_leaders['login'].tolist())

print(f"Top 5 users in terms of leader strength: {top_logins}")




# Calculate the correlation between followers and public repositories
correlation = users_df['followers'].corr(users_df['public_repos'])

print(f"The correlation between the number of followers and the number of public repositories is: {correlation:.3f}")




# Prepare the independent (X) and dependent (Y) variables
X = users_df['public_repos']  # Independent variable
Y = users_df['followers']      # Dependent variable

# Add a constant to the independent variable for the intercept
X = sm.add_constant(X)

# Perform linear regression
model = sm.OLS(Y, X).fit()

# Get the summary of the regression
summary = model.summary()
additional_followers_per_repo = model.params['public_repos']

print(summary)
print(f"Estimated additional followers per additional public repository: {additional_followers_per_repo:.3f}")



# Convert boolean columns to integers (1 for True, 0 for False)
repos_df['has_projects'] = repos_df['has_projects'].astype(int)
repos_df['has_wiki'] = repos_df['has_wiki'].astype(int)

# Calculate the correlation between has_projects and has_wiki
correlation = repos_df['has_wiki'].corr(repos_df['has_projects'])
print(correlation)
print(f"The correlation between enabling projects and enabling wiki is: {correlation:.3f}")



users_df['hireable'] = users_df['hireable'].astype(bool)

# Calculate average following count for hireable and non-hireable users
average_following_hireable = users_df[users_df['hireable']]['following'].mean()
average_following_non_hireable = users_df[~users_df['hireable']]['following'].mean()

difference = average_following_hireable - average_following_non_hireable

print(f"Difference in average following: {difference:.3f}")



# Filter out users without bios
users_with_bios = users_df[users_df['bio'].notnull()].copy()  # .copy() to avoid the warning

# Calculate the length of each bio in characters
users_with_bios['bio_length'] = users_with_bios['bio'].str.len()

# Prepare the independent (X) and dependent (Y) variables
A = users_with_bios['bio_length']  # Independent variable
B = users_with_bios['followers']    # Dependent variable

# Add a constant to the independent variable for the intercept
A = sm.add_constant(A)

# Perform linear regression
model = sm.OLS(B, A).fit()

# Get the slope (coefficient) for bio length
slope = model.params['bio_length']

# Print the result rounded to 3 decimal places
print(f"Regression slope of followers on bio length: {slope:.3f}")




# Convert the created_at column to datetime
repos_df['repo_created_at'] = pd.to_datetime(repos_df['repo_created_at'])

# Filter for repositories created on weekends (Saturday=5, Sunday=6)
repos_df['weekday'] = repos_df['repo_created_at'].dt.weekday
weekend_repos = repos_df[repos_df['weekday'].isin([5, 6])]

# Group by user and count repositories
top_weekend_users = weekend_repos['login'].value_counts().head(5)

# Prepare the result as a comma-separated string
top_users_logins = ', '.join(top_weekend_users.index)

print(f"Top 5 users who created the most repositories on weekends: {top_users_logins}")




# Calculate the total number of users and those with emails for hireable and non-hireable users
total_hireable = users_df[users_df['hireable']].shape[0]
total_non_hireable = users_df[~users_df['hireable']].shape[0]

# Count users with emails
users_with_email_hireable = users_df[users_df['hireable']]['email'].notnull().sum()
users_with_email_non_hireable = users_df[~users_df['hireable']]['email'].notnull().sum()

# Calculate fractions
fraction_hireable = users_with_email_hireable / total_hireable if total_hireable > 0 else 0
fraction_non_hireable = users_with_email_non_hireable / total_non_hireable if total_non_hireable > 0 else 0

# Calculate the difference
difference = fraction_hireable - fraction_non_hireable

# Print the result rounded to 3 decimal places
print(f"Difference in email sharing: {difference:.3f}")



# Drop rows with missing names
users_df = users_df[users_df['name'].notnull()]

# Extract surnames by splitting names and getting the last word
users_df['surname'] = users_df['name'].str.strip().str.split().str[-1]

# Count occurrences of each surname
surname_counts = users_df['surname'].value_counts()

# Find the maximum count
max_count = surname_counts.max()

# Get all surnames with the maximum count
most_common_surnames = surname_counts[surname_counts == max_count].index.tolist()

# Sort surnames alphabetically
most_common_surnames.sort()

# Prepare the result as a comma-separated string
result = ', '.join(most_common_surnames)

print(f"Most common surname(s): {result}")
print(surname_counts)



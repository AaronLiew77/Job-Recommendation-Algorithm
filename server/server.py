from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# Initialize Supabase client
url = "https://rpwiwppxsdyipvfrehql.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwd2l3cHB4c2R5aXB2ZnJlaHFsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNTEzODkxMSwiZXhwIjoyMDMwNzE0OTExfQ.GDjTVcPqV1iwFyoNu1jWq05z_JbCCAaNeHHjVerxVsM"
supabase: Client = create_client(url, key)

# Retrieve job data from Supabase
def get_jobs_from_supabase():
    response = supabase.table('jobs_posted').select('job_title, job_description').execute()
    data = response.data
    return pd.DataFrame(data)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variable to store job data
df_gigs = get_jobs_from_supabase()

# Function to update job data
def update_job_data():
    global df_gigs, tfidf, tfidf_matrix, cosine_sim
    df_gigs = get_jobs_from_supabase()
    tfidf, tfidf_matrix = update_tfidf(df_gigs)
    cosine_sim = update_cosine_sim(tfidf_matrix)
    print("Job data updated.")

# TF-IDF vectorization
def update_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['job_description'])
    return tfidf, tfidf_matrix

# Cosine Similarity matrix
def update_cosine_sim(tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Initialize TF-IDF and Cosine Similarity
tfidf, tfidf_matrix = update_tfidf(df_gigs)
cosine_sim = update_cosine_sim(tfidf_matrix)

# Scheduler to update job data periodically
scheduler = BackgroundScheduler(timezone=pytz.utc)  # Use UTC timezone
scheduler.add_job(update_job_data, 'interval', minutes=1)  # Update every 1 minutes
scheduler.start()

# Define API endpoint for job recommendations
@app.route("/api/job_recommendations", methods=['POST'])
def get_job_recommendations():
    # Extract job title from request data
    data = request.get_json()
    title_to_search = data.get('job_title')

    # Function to get recommendations
    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = df_gigs.index[df_gigs['job_title'] == title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Exclude the first one, which is the job itself
        recommendations = [df_gigs.iloc[i]['job_title'] for i, _ in sim_scores]
        return recommendations

    # Get recommendations for the given job title
    recommendations = get_recommendations(title_to_search)

    return jsonify({'job_recommendations': recommendations})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
